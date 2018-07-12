#include <algorithm>
#include "simulator/camera_simulate.hpp"
#include "okvis_util/size.hpp"
#include "okvis_util/random.hpp"

#include <glog/logging.h>
CameraSimulator::CameraSimulator(const std::shared_ptr<Trajectory>& trajectory,
                                 const okvis::cameras::NCameraSystem& nCameraSystem,
                const CameraSimulatorOptions cameraSimulatorOptions ):
trajectory_(trajectory), nCameraSystem_(nCameraSystem), options_(cameraSimulatorOptions) {

}

void CameraSimulator::initializeMap() {
    int64_t num_frames = trajectory_->size();
    int  num_camera = nCameraSystem_.numCameras();
    uint32_t  num_landmarks = landmarks_W_.cols();
    for (size_t i = 0 ; i < num_frames; i++) {
        Pose<double> T_W_B = trajectory_->at(i).second;
        for (size_t cam_idx = 0 ; cam_idx < num_camera; cam_idx ++) {
            CameraMeasurements measurements = visibleLandmarks(cam_idx, T_W_B, 0u, num_landmarks);
            int num_visible = measurements.keypoints_.cols();
            if (num_visible >= options_.min_num_keypoints_per_frame)
            {
                continue;
            }

            // Initialize new random visible landmarks.
            int32_t num_new_landmarks = std::max(0u, options_.min_num_keypoints_per_frame - num_visible);
            CHECK_GE(num_new_landmarks, 0);

            Positions p_C;
//            std::tie(std::ignore, std::ignore, p_C) =
//                    generateRandomVisible3dPoints(
//                            rig_->at(cam_idx), num_new_landmarks,
//                            10u, options_.min_depth_m, options_.max_depth_m);
//
//            DEBUG_CHECK_LE(static_cast<int>(num_landmarks + num_new_landmarks),
//                           landmarks_W_.cols());

        }
    }

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
    Eigen::Matrix2Xd  imagePoints;
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

// -----------------------------------------------------------------------------
Keypoints CameraSimulator::generateRandomKeypoints(
        const Size2u size,
        const uint32_t margin,
        const uint32_t num_keypoints)
{
//    DEBUG_CHECK_GT(size.width(), margin + 1u);
//    DEBUG_CHECK_GT(size.height(), margin + 1u);

    Keypoints kp(2, num_keypoints);
    for(uint32_t i = 0u; i < num_keypoints; ++i)
    {
        kp(0,i) = sampleUniformRealDistribution<real_t>(false, margin, size.width() - 1 - margin);
        kp(1,i) = sampleUniformRealDistribution<real_t>(false, margin, size.height() - 1 - margin);
    }
    return kp;
}

// -----------------------------------------------------------------------------
Keypoints CameraSimulator::generateUniformKeypoints(
        const Size2u size,
        const uint32_t margin,
        const uint32_t num_cols)
{
//    DEBUG_CHECK_GT(size.width(), margin + 1u);
//    DEBUG_CHECK_GT(size.height(), margin + 1u);
    const uint32_t num_rows = num_cols * size.height() / size.width();

    // Compute width and height of a cell:
    real_t w = (static_cast<real_t>(size.width() - 0.01)  - 2.0 * margin) / (num_cols - 1);
    real_t h = (static_cast<real_t>(size.height() - 0.01) - 2.0 * margin) / (num_rows - 1);

    // Sample keypoints:
    Keypoints kp(2, num_rows * num_cols);
    for (uint32_t y = 0u; y < num_rows; ++y)
    {
        for (uint32_t x = 0u; x < num_cols; ++x)
        {
            uint32_t i = y * num_cols + x;
            kp(0,i) = margin + x * w;
            kp(1,i) = margin + y * h;
        }
    }
    return kp;
}

// -----------------------------------------------------------------------------
std::tuple<Keypoints, Bearings, Positions> CameraSimulator::generateRandomVisible3dPoints(
        const okvis::cameras::NCameraSystem& cam,
        const uint32_t num_points,
        const uint32_t margin,
        const real_t min_depth,
        const real_t max_depth)
{
//    Keypoints px = generateRandomKeypoints(cam.size(), margin, num_points);
//    Bearings f = cam.backProjectVectorized(px);
//    Positions pos = f;
//    for(uint32_t i = 0u; i < num_points; ++i)
//    {
//        pos.col(i) *= sampleUniformRealDistribution<real_t>(false, min_depth, max_depth);
//    }
//    return std::make_tuple(px, f, pos);
}

