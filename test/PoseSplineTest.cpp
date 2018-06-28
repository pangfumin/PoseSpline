#include <iostream>
#include <fstream>

#include "common/csv_trajectory.hpp"
#include "splines/bspline.hpp"
#include "pose-spline/Quaternion.hpp"
#include "pose-spline/QuaternionSpline.hpp"
#include <pose-spline/QuaternionSplineUtility.hpp>
#include <pose-spline/PoseSpline.hpp>


using namespace ze;

std::pair<double,Quaternion>  getSample(ze::TupleVector& data, unsigned int i){
    ze::TrajectoryEle  p0 = data.at(i);
    Quaternion q = std::get<2>(p0);
    return std::make_pair(std::get<0>(p0)*1e-9,q);
};

std::pair<double,Eigen::Vector3d>  getPositionSample(ze::TupleVector& data, unsigned int i){
    ze::TrajectoryEle  p0 = data.at(i);
    Eigen::Vector3d t = std::get<1>(p0);
    return std::make_pair(std::get<0>(p0)*1e-9,t);
};


/**
 TODO:: fixme , input Hamilton quaternion into quaternion spline 
       which will mistake 
*/

int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    std::string dataset = "/media/pang/Plus/dataset/MH_01_easy";
    ze::EurocResultSeries eurocDataReader;
    eurocDataReader.load(dataset + "/mav0/state_groundtruth_estimate0/data.csv");
    eurocDataReader.loadIMU(dataset+ "/mav0/imu0/data.csv");

    ze::TupleVector  data = eurocDataReader.getVector();
    Buffer<real_t, 7>& poseBuffer = eurocDataReader.getBuffer();
    LOG(INFO)<<"Get data size: "<<data.size(); // @200Hz
    std::vector<std::pair<int64_t ,Eigen::Vector3d>> linearVelocities = eurocDataReader.getLinearVelocities();
    std::vector<Vector3> gyroBias = eurocDataReader.getGyroBias();
    LOG(INFO)<<"Get velocities size: "<<linearVelocities.size(); // @200Hz
    LOG(INFO)<<"Get gyro_bias  size: "<<gyroBias.size(); // @200Hz

    std::vector<int64_t> imu_ts = eurocDataReader.getIMUts();
    LOG(INFO)<<"Get IMU ts  size: "<<imu_ts.size(); // @200Hz

    std::vector<Vector3> gyroMeas = eurocDataReader.getGyroMeas();
    LOG(INFO)<<"Get gyro Meas  size: "<<gyroMeas.size(); // @200Hz

    int start  = 1;
    int end = data.size()/100;

    //ze::QuaternionSpline qspline(4,0.1);
    ze::PoseSpline poseSpline(4, 0.1);
    std::vector<std::pair<double,Pose<double>>> samples, queryMeas;

    for(uint i = start; i <end; i++){

        Pose<double> pose(getPositionSample( data,  i).second,getSample( data,  i).second);
        queryMeas.push_back(std::pair<double,Pose<double>>(getPositionSample( data,  i).first, pose ) );
        if(i % 4  == 0){
            samples.push_back(std::pair<double,Pose<double>>(getPositionSample( data,  i).first, pose ));

        }
    }

    poseSpline.initialPoseSpline(samples);

    /*
     *  Test: qspline.evalQuatSpline
     */

    for(auto i: queryMeas){

        if(poseSpline.isTsEvaluable(i.first)){
            Pose<double> query = poseSpline.evalPoseSpline(i.first);
//
//            std::cout <<"Gt:    "<<i.second.r().transpose() << " " << i.second.q().transpose()<<std::endl;
//            std::cout <<"Query: "<<query.r().transpose()<<" "<< query.q().transpose()<< std::endl << std::endl;
        }

    }

    LOG(INFO)<<" - QuaternionSpline initialization passed!";

    /*
     * Check PoseSpline linear velocity
     *
     * Passed!
     */

    std::vector<std::pair<double,Eigen::Vector3d>> velocitySamples;
    for(uint i = start; i <end; i++){

        std::pair<double,Eigen::Vector3d> velocity;
        velocity.first = (double)linearVelocities.at(i).first* 1e-9;
        velocity.second = linearVelocities.at(i).second;

        if(poseSpline.isTsEvaluable(velocity.first)){
            Eigen::Vector3d query = poseSpline.evalLinearVelocity(velocity.first);

            std::cout <<"Gt:    "<< velocity.second.transpose()<<std::endl;
            std::cout <<"Query: "<< query.transpose()<< std::endl << std::endl;
        }
    }

    return 0;
}
