#include "gtest/gtest.h"
#include "simulator/camera_simulate.hpp"
#include "common/csv_trajectory.hpp"
#include "estimator/VioParametersReader.hpp"
#include "frontend/triangulate/ProbabilisticStereoTriangulator.hpp"
#include "okvis_cv/cameras/PinholeCamera.hpp"
#include "okvis_cv/cameras/RadialTangentialDistortion.hpp"
#include <glog/logging.h>
using namespace ze;

//google::InitGoogleLogging();

TEST(TestSimulator, cameraSimulator) {
    std::string dataset = "/home/pang/software/PoseSpline/data/MH_01_easy";
    ze::EurocResultSeries eurocDataReader;
    eurocDataReader.load(dataset + "/state_groundtruth_estimate0/data.csv");
    eurocDataReader.loadIMU(dataset + "/imu0/data.csv");
    ze::TupleVector data = eurocDataReader.getVector();
    Buffer<real_t, 7> &poseBuffer = eurocDataReader.getBuffer();
    LOG(INFO) << "Get data size: " << data.size(); // @200Hz


// read configuration file
    std::string configFilename = dataset +  "/config_fpga_p2_euroc.yaml";

    okvis::VioParametersReader vio_parameters_reader(configFilename);
    okvis::VioParameters parameters;
    vio_parameters_reader.getParameters(parameters);

    std::shared_ptr<okvis::cameras::NCameraSystem> ncamera
            = std::make_shared<okvis::cameras::NCameraSystem>(parameters.nCameraSystem);


    size_t start = data.size()*59/100;
    size_t end  = data.size()*6/10 ;
    std::shared_ptr<Trajectory> trajectory = std::make_shared<Trajectory >();
    for (size_t i = start; i < end; i++) {
        ze::TrajectoryEle  p0 = data.at(i);
        uint64_t time = std::get<0>(p0);
        Eigen::Vector3d t = std::get<1>(p0);
        Quaternion q = std::get<2>(p0);

        Pose<double> pose(t, HamiltonToJPL(q));
        trajectory->push_back(std::make_pair(NanosecondsToTime(time), pose));
    }

    std::cout<< "trajectory size: " << trajectory->size()<< std::endl;

    CameraSimulatorOptions cameraSimulatorOptions;
    CameraSimulator cameraSimulator(trajectory,*ncamera,cameraSimulatorOptions);


//    okvis::triangulation::ProbabilisticStereoTriangulator<
//            okvis::cameras::PinholeCamera<
//                    okvis::cameras::RadialTangentialDistortion> > probabilisticStereoTriangulator;

    //std::vector<okvis::MultiFramePtr> frames;
    while (cameraSimulator.hasNextMeasurement()) {
        okvis::MultiFramePtr multiFramePtr = cameraSimulator.getNextMeasurement();
        //frames.push_back(multiFramePtr);
        for (int cam_id  = 0; cam_id < multiFramePtr->numFrames(); cam_id ++) {
            for(size_t kp_id = 0; kp_id < multiFramePtr->numKeypoints(cam_id); kp_id++) {
                Eigen::Vector2d kp;
                if( multiFramePtr->getKeypoint(cam_id, kp_id,kp)){
                    uint64_t landmark_id = multiFramePtr->landmarkId(cam_id, kp_id);
                    Position landmark = cameraSimulator.getLandmark(landmark_id);
                    Pose<double> T_W_C = multiFramePtr->getCameraPose(cam_id);
                    Position C_landmark = T_W_C.inverse().transformVector3d(landmark);
                    Eigen::Vector2d  imagePoint;
                    multiFramePtr->geometry(cam_id)->project(C_landmark, &imagePoint);

                    EXPECT_TRUE((kp - imagePoint).norm() < 0.001);

                    size_t query_kp_id;
                    EXPECT_TRUE(multiFramePtr->isLandmarkSeenBy(cam_id,landmark_id,query_kp_id) && query_kp_id == kp_id);


                }
            }
        }


    }

    //
//    okvis::MultiFramePtr multiFramePtr1 = frames.front();
//    std::map<uint64_t, int> landmarks_set;
//    size_t kp_num_A = multiFramePtr1->numKeypoints(0);
//    size_t kp_num_B = multiFramePtr1->numKeypoints(1);
//    for (int i = 0 ; i < kp_num_B; i ++) {
//        landmarks_set[multiFramePtr1->landmarkId(1,i)] = i;
//    }
//    std::cout<< "set B: " << landmarks_set.size() << std::endl;
//
//    Pose<double> T_CaCb = multiFramePtr1->T_SC(0)->inverse() * (*multiFramePtr1->T_SC(1));
//    Eigen::Matrix<double,6,6> UOplus;
//    probabilisticStereoTriangulator.resetFrames(multiFramePtr1, multiFramePtr1, 0,
//                                                 1, T_CaCb, UOplus);
//    for (int idA = 0; idA < kp_num_A; idA++) {
//        //std::set<uint64_t>::iterator itr = landmarks_set.find(multiFramePtr1->landmarkId(0,idA));
//        if(landmarks_set.count(multiFramePtr1->landmarkId(0,idA))) {
//            int idB = landmarks_set[multiFramePtr1->landmarkId(0,idA)];
//            uint64_t landmark_id = multiFramePtr1->landmarkId(0,idA);
//
//            EXPECT_TRUE(multiFramePtr1->landmarkId(0,idA) == multiFramePtr1->landmarkId(1,idB));
//
//            Eigen::Vector4d point_in_A;
//            Eigen::Matrix3d outPointUOplus_A;
//            bool canBeInitial = false;
//            probabilisticStereoTriangulator.stereoTriangulate(idA, idB, point_in_A,outPointUOplus_A,canBeInitial);
//
//            //std::cout<< "idA-idB: "<< idA << "  " << idB <<" "<<landmark_id<< " "<< canBeInitial <<std::endl;
//        }
//    }


}

