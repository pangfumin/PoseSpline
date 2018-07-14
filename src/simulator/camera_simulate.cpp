#include <algorithm>
#include "simulator/camera_simulate.hpp"
#include "okvis_util/size.hpp"
#include "okvis_util/random.hpp"
#include "okvis_cv/cameras/camera_util.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#define ZE_USE_OPENCV

CameraSimulator::CameraSimulator(const std::shared_ptr<Trajectory>& trajectory,
                                 const okvis::cameras::NCameraSystem& nCameraSystem,
                const CameraSimulatorOptions cameraSimulatorOptions ):
trajectory_(trajectory), nCameraSystem_(nCameraSystem),
options_(cameraSimulatorOptions), num_landmarks_(0),
landmark_preAlooc_num_(20*cameraSimulatorOptions.min_num_keypoints_per_frame) {
    landmarks_W_.conservativeResize(Eigen::NoChange, landmark_preAlooc_num_);
    trajectory_itr_ = trajectory_->begin();
    initializeMap();
}

bool CameraSimulator::hasNextMeasurement() {
    return trajectory_itr_ != trajectory_->end();
}

okvis::MultiFramePtr CameraSimulator::getNextMeasurement() {
    Time ts = trajectory_itr_->first;
    Pose<double> T_W_B = trajectory_itr_->second;
    size_t  num_camera = nCameraSystem_.numCameras();
    okvis::MultiFramePtr multiFramePtr = std::make_shared<okvis::MultiFrame>(nCameraSystem_,ts);
    for (size_t cam_idx = 0 ; cam_idx < num_camera; cam_idx ++) {
        CameraMeasurements measurements = visibleLandmarks(cam_idx, T_W_B, 0u, num_landmarks_);

        for (int i = 0; i < measurements.keypoints_.cols(); i ++) {
            cv::KeyPoint kp(static_cast<float>(measurements.keypoints_.col(i)[0])
                    , static_cast<float>(measurements.keypoints_.col(i)[1]),1.0);

            multiFramePtr->appendKeypoint(cam_idx,kp);
            multiFramePtr->setLandmarkId(cam_idx,i,
                                         static_cast<uint64_t>(measurements.global_landmark_ids_[i]));
        }
        //multiFramePtr->resetKeypoints(cam_idx, keypoints);

#ifdef ZE_USE_OPENCV
        if (cam_idx == 0) {
            cv::Mat img_0(nCameraSystem_.cameraGeometry(cam_idx)->imageHeight(),
                          nCameraSystem_.cameraGeometry(cam_idx)->imageWidth(), CV_8UC1, cv::Scalar(0));


            for (int i = 0; i <  multiFramePtr->numKeypoints(cam_idx); ++i)
            {
                Eigen::Vector2d  keypoint;
                multiFramePtr->getKeypoint(cam_idx,i,keypoint);

                cv::Point p(keypoint(0), keypoint(1));
                cv::circle(img_0, p, 1, cv::Scalar(255), 1);

                char name[5];
                sprintf(name, "%d", multiFramePtr->landmarkId(cam_idx,i));
                cv::putText(img_0, name, p, cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar( 200));
            }
            cv::imshow("img_0", img_0);
            cv::waitKey(1);
        }
        if (cam_idx == 1) {
            cv::Mat img_0(nCameraSystem_.cameraGeometry(cam_idx)->imageHeight(),
                          nCameraSystem_.cameraGeometry(cam_idx)->imageWidth(), CV_8UC1, cv::Scalar(0));


            for (int i = 0; i <  multiFramePtr->numKeypoints(cam_idx); ++i)
            {
                Eigen::Vector2d  keypoint;
                multiFramePtr->getKeypoint(cam_idx,i,keypoint);

                cv::Point p(keypoint(0), keypoint(1));
                cv::circle(img_0, p, 1, cv::Scalar(255), 1);

                char name[5];
                sprintf(name, "%d", multiFramePtr->landmarkId(cam_idx,i));
                cv::putText(img_0, name, p, cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar( 200));
            }
            cv::imshow("img_1", img_0);
            cv::waitKey(1);
        }
#endif
    }

    trajectory_itr_ ++;
    return multiFramePtr;
}

void CameraSimulator::initializeMap() {
    size_t num_frames = trajectory_->size();
    size_t  num_camera = nCameraSystem_.numCameras();

    for (size_t i = 0 ; i < num_frames; i++) {
        Pose<double> T_W_B = trajectory_->at(i).second;
        for (size_t cam_idx = 0 ; cam_idx < num_camera; cam_idx ++) {
            CameraMeasurements measurements = visibleLandmarks(cam_idx, T_W_B, 0u, num_landmarks_);



            int num_visible = measurements.keypoints_.cols();
            if (num_visible >= options_.min_num_keypoints_per_frame)
            {
                continue;
            }

            // Initialize new random visible landmarks.
            int32_t num_new_landmarks = std::max(0u, options_.min_num_keypoints_per_frame - num_visible);
            CHECK_GE(num_new_landmarks, 0);

            Positions p_C;
            std::tie(std::ignore, std::ignore, p_C) =
                    generateRandomVisible3dPoints(
                            nCameraSystem_,cam_idx, num_new_landmarks,
                            10u, options_.min_depth_m, options_.max_depth_m);
//
//            DEBUG_CHECK_LE(static_cast<int>(num_landmarks + num_new_landmarks),
//                           landmarks_W_.cols());

            if (num_landmarks_ + num_new_landmarks > landmark_preAlooc_num_) {
                landmarks_W_.conservativeResize(Eigen::NoChange ,num_landmarks_ + landmark_preAlooc_num_);
            }

            Positions p_W =  (T_W_B* (*nCameraSystem_.T_SC(cam_idx))).transformVector3d( p_C);
            landmarks_W_.middleCols(num_landmarks_,  num_new_landmarks)
                    = p_W;

            num_landmarks_ += num_new_landmarks;
            //std::cout<< "num_landmarks_: "<<num_landmarks_ << std::endl;
        }
    }

    landmarks_W_ = landmarks_W_.middleCols(0u, num_landmarks_);
    std::cout<< " - Init map : " << std::endl;
    std::cout<< " - Trajectory    : " << trajectory_->size() <<  std::endl;
    std::cout<< " - Camera        : " << nCameraSystem_.numCameras() <<  std::endl;
    std::cout<< " - Landmarks     : " << num_landmarks_ <<  std::endl;

}



// -----------------------------------------------------------------------------
CameraMeasurements CameraSimulator::visibleLandmarks(
        const uint32_t cam_idx,
        const Pose<double>& T_W_B,
        const uint32_t lm_min_idx,
        const uint32_t lm_max_idx)
{


    const uint32_t num_landmarks = lm_max_idx - lm_min_idx;
    if (num_landmarks == 0)
    {
        return CameraMeasurements();
    }

    const Size2u image_size ( nCameraSystem_.cameraGeometry(cam_idx)->imageWidth(),
                              nCameraSystem_.cameraGeometry(cam_idx)->imageHeight());
    const auto lm_W = landmarks_W_.middleCols(lm_min_idx, num_landmarks);
    auto T_C_W = (T_W_B * (*nCameraSystem_.T_SC(cam_idx))).inverse();

    const auto lm_C = T_C_W.transformVector3d(lm_W);
    Eigen::Matrix2Xd  imagePoints(2, num_landmarks);
    std::vector<okvis::cameras::CameraBase::ProjectionStatus>  stati;
    nCameraSystem_.cameraGeometry(cam_idx)->projectBatch(lm_C, &imagePoints, &stati);


    std::vector<uint32_t> visible_indices;
    for (uint32_t i = 0u; i < num_landmarks; i++) {
        if (stati[i] == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
            uint32_t global_id = lm_min_idx + i;
            visible_indices.push_back(global_id);
        }

    }

     // Copy visible indices into Camera Measurements struct:
     CameraMeasurements m;
     m.keypoints_.resize(Eigen::NoChange, visible_indices.size());
     m.global_landmark_ids_.resize(visible_indices.size());
     for (size_t i = 0; i < visible_indices.size(); ++i)
     {
         m.keypoints_.col(i) = imagePoints.col(visible_indices[i]);
         m.global_landmark_ids_[i] = visible_indices[i];
     }

    return m;
}

