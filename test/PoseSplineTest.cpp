#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "common/csv_trajectory.hpp"

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



TEST(PoseSplineTest, initialAndEvaluate){
    //google::InitGoogleLogging(argv[0]);

    std::string dataset = "/home/pang/software/PoseSpline/data/MH_01_easy";
    ze::EurocResultSeries eurocDataReader;
    eurocDataReader.load(dataset + "/state_groundtruth_estimate0/data.csv");
    eurocDataReader.loadIMU(dataset+ "/imu0/data.csv");
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

    std::vector<Vector3> accelMeas = eurocDataReader.getAccelMeas();
    LOG(INFO)<<"Get accel Meas  size: "<<accelMeas.size(); // @200Hz

    int start  = 1;
    int end = data.size()/5;

    PoseSpline poseSpline(0.1);
    std::vector<std::pair<double,Pose<double>>> samples, queryMeas;

    for(uint i = start; i <end; i++){
        std::pair<double,Quaternion> sample = getSample( data,  i);
        Eigen::Quaterniond QuatHamilton(sample.second(3),sample.second(0),sample.second(1),sample.second(2));
        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
        Quaternion QuatJPL = rotMatToQuat(R);
        std::pair<double,Quaternion> sampleJPL = std::make_pair(sample.first, QuatJPL);

        Pose<double> pose(getPositionSample( data,  i).second,sampleJPL.second);
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
            EXPECT_TRUE((i.second.r() -query.r()).squaredNorm() < 0.01);
            EXPECT_TRUE((i.second.q() -query.q()).squaredNorm() < 0.0001);
        }

    }

    //LOG(INFO)<<" - QuaternionSpline initialization passed!";

    /*
     * Check PoseSpline linear velocity
     *
     * Passed!
     */

    std::vector<std::pair<double,Eigen::Vector3d>> velocitySamples;
    int inlier_cnt = 0;
    for(uint i = start; i <end; i++){
        std::pair<double,Eigen::Vector3d> velocity;
        velocity.first = (double)linearVelocities.at(i).first* 1e-9;
        velocity.second = linearVelocities.at(i).second;

        if(poseSpline.isTsEvaluable(velocity.first)){
            Eigen::Vector3d query = poseSpline.evalLinearVelocity(velocity.first);

            // std::cout <<"Gt:    "<< velocity.second.transpose()<<std::endl;
            // std::cout <<"Query: "<< query.transpose()<< std::endl << std::endl;
            inlier_cnt += (velocity.second -query).squaredNorm() < 0.01; 
        }
    }
    EXPECT_TRUE(inlier_cnt / (double)(end - start) > 0.97);


    // std::ofstream ofs_debug("/home/pang/debug.txt");


    std::map<int64_t,Eigen::Vector3d> accelMap = eurocDataReader.getAccelMeasMap();
    //LOG(INFO)<<"accelMap size: "<<accelMap.size();

    std::map<int64_t,Eigen::Vector3d>::iterator search;
    inlier_cnt = 0;
    for(uint i = start; i <end; i++){

        ze::TrajectoryEle  p0 = data.at(i);
        int64_t ts = std::get<0>(p0);
        search = accelMap.find(ts);

        if(search != accelMap.end() && poseSpline.isTsEvaluable(ts*1e-9)){
            Eigen::Vector3d evalAccel = poseSpline.evalLinearAccelerator(ts*1e-9);

            //std::cout<<"Found!"<<std::endl;
            //ofs_debug<< search->second.transpose()<<" "<< evalAccel.transpose()<<std::endl;
            inlier_cnt += (search->second - evalAccel).norm() < 1.0; 
        }else{
            //std::cout<<"Not found!"<<std::endl;
        }
    }
    std::cout<< "inlier_cnt / (double)(end - start): " << inlier_cnt / (double)(end - start)<< std::endl;
    EXPECT_TRUE(inlier_cnt / (double)(end - start) > 0.75);


    // ofs_debug.close();
}
