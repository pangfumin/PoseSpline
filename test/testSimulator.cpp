#include "gtest/gtest.h"
#include "simulator/camera_simulate.hpp"
#include "common/csv_trajectory.hpp"
#include "estimator/VioParametersReader.hpp"
#include <glog/logging.h>
using namespace ze;

TEST(TestSimulator, cameraSimulator) {

    //google::InitGoogleLogging();
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
    size_t start = data.size()*5/10;
    size_t end  = data.size()*6/10 ;
    std::shared_ptr<Trajectory> trajectory = std::make_shared<Trajectory >();
    for (size_t i = start; i < end; i++) {
        ze::TrajectoryEle  p0 = data.at(i);
        uint64_t time = std::get<0>(p0);
        Eigen::Vector3d t = std::get<1>(p0);
        Quaternion q = std::get<2>(p0);

//        std::cout<< "time 1: " << time<< std::endl;
//        std::cout<< "time 2: " << NanosecondsToTime(time) << std::endl;

        Pose<double> pose(t, HamiltonToJPL(q));
        trajectory->push_back(std::make_pair(NanosecondsToTime(time), pose));
    }

    std::cout<< "trajectory size: " << trajectory->size()<< std::endl;

    CameraSimulatorOptions cameraSimulatorOptions;
    CameraSimulator cameraSimulator(trajectory,*ncamera,cameraSimulatorOptions);

    while (cameraSimulator.hasNextMeasurement()) {
        cameraSimulator.getNextMeasurement();
    }

}