#include <iostream>
#include <fstream>

#include <gtest/gtest.h>
#include "PoseSpline/QuaternionSpline.hpp"
#include <PoseSpline/QuaternionSplineUtility.hpp>
#include <PoseSpline/PoseSpline.hpp>
#include "csv.h"
#include "PoseSpline/Time.hpp"



struct StampedPose{
    uint64_t timestamp_;
    Eigen::Vector3d t_;
    Eigen::Quaterniond q_;
    Eigen::Vector3d v_;
};

struct StampedImu{
    uint64_t timestamp_;
    Eigen::Vector3d accel_;
    Eigen::Vector3d gyro_;
};
class TestSample {
public:


    void readStates(const std::string& states_file) {
        io::CSVReader<11> in(states_file);
        in.read_header(io::ignore_extra_column, "#timestamp",
                       "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
                       "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
                       "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]");
        std::string vendor; int size; double speed;
        int64_t timestamp;

        double p_RS_R_x, p_RS_R_y, p_RS_R_z;
        double q_RS_w, q_RS_x, q_RS_y, q_RS_z;
        double v_RS_R_x, v_RS_R_y, v_RS_R_z;
        int cnt  =0 ;

        states_vec_.clear();
        while(in.read_row(timestamp,
                          p_RS_R_x, p_RS_R_y, p_RS_R_z,
                          q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                          v_RS_R_x, v_RS_R_y, v_RS_R_z)){
            // do stuff with the data

            StampedPose pose;
            pose.timestamp_ = timestamp;
            pose.t_ = Eigen::Vector3d(p_RS_R_x, p_RS_R_y, p_RS_R_z);
            pose.v_ = Eigen::Vector3d(v_RS_R_x, v_RS_R_y, v_RS_R_z);
            pose.q_ = Eigen::Quaterniond(q_RS_w, q_RS_x, q_RS_y, q_RS_z);
            states_vec_.push_back(pose);
            cnt ++;

        }

        std::cout << "Load states: " << states_vec_.size() << std::endl;
    }

    void readImu(const std::string& IMU_file) {
        io::CSVReader<7> in(IMU_file);
        in.read_header(io::ignore_extra_column, "#timestamp [ns]",
                       "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
                       "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]");
        std::string vendor; int size; double speed;
        int64_t timestamp;

        double w_RS_S_x, w_RS_S_y, w_RS_S_z;
        double a_RS_S_x, a_RS_S_y, a_RS_S_z;
        int cnt  =0 ;

        imu_vec_.clear();
        while(in.read_row(timestamp,
                          w_RS_S_x, w_RS_S_y, w_RS_S_z,
                          a_RS_S_x, a_RS_S_y, a_RS_S_z)){
            // do stuff with the data

            StampedImu imu;
            imu.timestamp_ = timestamp;
            imu.accel_ = Eigen::Vector3d(a_RS_S_x, a_RS_S_y, a_RS_S_z);
            imu.gyro_ = Eigen::Vector3d(w_RS_S_x, w_RS_S_y, w_RS_S_z);
            imu_vec_.push_back(imu);
            cnt ++;

        }

        std::cout << "Load imu: " << imu_vec_.size() << std::endl;
    }

    std::vector<StampedPose> states_vec_;
    std::vector<StampedImu> imu_vec_;

};

TEST( Spline , initialization){
    //google::InitGoogleLogging(argv[0]);

//    std::string dataset = "/home/pang/software/PoseSpline/data/MH_01_easy";
//    ze::EurocResultSeries eurocDataReader;
//    eurocDataReader.load(dataset + "/state_groundtruth_estimate0/data.csv");
//    eurocDataReader.loadIMU(dataset+ "/imu0/data.csv");
//    ze::TupleVector  data = eurocDataReader.getVector();
//    Buffer<real_t, 7>& poseBuffer = eurocDataReader.getBuffer();
//    LOG(INFO)<<"Get data size: "<<data.size(); // @200Hz
//    std::vector<std::pair<int64_t ,Eigen::Vector3d>> linearVelocities = eurocDataReader.getLinearVelocities();
//    std::vector<Vector3> gyroBias = eurocDataReader.getGyroBias();
//    LOG(INFO)<<"Get velocities size: "<<linearVelocities.size(); // @200Hz
//    LOG(INFO)<<"Get gyro_bias  size: "<<gyroBias.size(); // @200Hz
//
//    std::vector<int64_t> imu_ts = eurocDataReader.getIMUts();
//    LOG(INFO)<<"Get IMU ts  size: "<<imu_ts.size(); // @200Hz
//
//    std::vector<Vector3> gyroMeas = eurocDataReader.getGyroMeas();
//    LOG(INFO)<<"Get gyro Meas  size: "<<gyroMeas.size(); // @200Hz
//
//    std::vector<Vector3> accelMeas = eurocDataReader.getAccelMeas();
//    LOG(INFO)<<"Get accel Meas  size: "<<accelMeas.size(); // @200Hz
//
//    int start  = 1;
//    int end = data.size()/10;
//
//    PoseSpline poseSpline(0.1);
//    std::vector<std::pair<double,Pose<double>>> samples, queryMeas;
//
//    for(uint i = start; i <end; i++){
//        std::pair<double,Quaternion> sample = getSample( data,  i);
//        Eigen::Quaterniond QuatHamilton(sample.second(3),sample.second(0),sample.second(1),sample.second(2));
//        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
//        Quaternion QuatJPL = rotMatToQuat(R);
//        std::pair<double,Quaternion> sampleJPL = std::make_pair(sample.first, QuatJPL);
//
//        Pose<double> pose(getPositionSample( data,  i).second,sampleJPL.second);
//        queryMeas.push_back(std::pair<double,Pose<double>>(getPositionSample( data,  i).first, pose ) );
//        if(i % 4  == 0){
//            samples.push_back(std::pair<double,Pose<double>>(getPositionSample( data,  i).first, pose ));
//
//        }
//    }
//
//    poseSpline.initialPoseSpline(samples);
//
//    /*
//     *  Test: qspline.evalQuatSpline
//     */
//
//    for(auto i: queryMeas){
//
//        if(poseSpline.isTsEvaluable(i.first)){
//            Pose<double> query = poseSpline.evalPoseSpline(i.first);
////
////            std::cout <<"Gt:    "<<i.second.r().transpose() << " " << i.second.q().transpose()<<std::endl;
////            std::cout <<"Query: "<<query.r().transpose()<<" "<< query.q().transpose()<< std::endl << std::endl;
//        }
//
//    }
//
//    LOG(INFO)<<" - QuaternionSpline initialization passed!";
//
//    /*
//     * Check PoseSpline linear velocity
//     *
//     * Passed!
//     */
//
//    std::vector<std::pair<double,Eigen::Vector3d>> velocitySamples;
//    for(uint i = start; i <end; i++){
//
//        std::pair<double,Eigen::Vector3d> velocity;
//        velocity.first = (double)linearVelocities.at(i).first* 1e-9;
//        velocity.second = linearVelocities.at(i).second;
//
//        if(poseSpline.isTsEvaluable(velocity.first)){
//            Eigen::Vector3d query = poseSpline.evalLinearVelocity(velocity.first);
//
//            std::cout <<"Gt:    "<< velocity.second.transpose()<<std::endl;
//            std::cout <<"Query: "<< query.transpose()<< std::endl << std::endl;
//        }
//    }
//
//
//    std::ofstream ofs_debug("/home/pang/debug.txt");
//
//
//    std::map<int64_t,Eigen::Vector3d> accelMap = eurocDataReader.getAccelMeasMap();
//    LOG(INFO)<<"accelMap size: "<<accelMap.size();
//
//    std::map<int64_t,Eigen::Vector3d>::iterator search;
//    for(uint i = start; i <end; i++){
//
//        ze::TrajectoryEle  p0 = data.at(i);
//        int64_t ts = std::get<0>(p0);
//        search = accelMap.find(ts);
//
//        if(search != accelMap.end() && poseSpline.isTsEvaluable(ts*1e-9)){
//            Eigen::Vector3d evalAccel = poseSpline.evalLinearAccelerator(ts*1e-9);
//
//            std::cout<<"Found!"<<std::endl;
//            ofs_debug<< search->second.transpose()<<" "<< evalAccel.transpose()<<std::endl;
//
//        }else{
//            std::cout<<"Not found!"<<std::endl;
//        }
//
//    }
//
//    ofs_debug.close();
//
//
//
//    return 0;

    std::string pose_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
    std::string imu_meas_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/imu0/data.csv";

    TestSample testSample;
    testSample.readStates(pose_file);
    testSample.readImu(imu_meas_file);

    int start  = 0;
    int end = testSample.states_vec_.size()/5;

    PoseSpline poseSpline(0.1);
    std::vector<std::pair<double,Pose<double>>> samples, queryMeas;
    std::vector<std::pair<double,Eigen::Vector3d>> queryVelocityMeas;


    //
    for(uint i = start; i <end; i++){
        StampedPose stampedPose = testSample.states_vec_.at(i);

        Eigen::Quaterniond QuatHamilton(stampedPose.q_);
        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
        Quaternion QuatJPL = rotMatToQuat(R);

        Pose<double> pose(stampedPose.t_, QuatJPL);
        queryMeas.push_back(std::pair<double,Pose<double>>(Time(stampedPose.timestamp_).toSec(), pose ) );
        queryVelocityMeas.push_back(std::pair<double, Eigen::Vector3d>(Time(stampedPose.timestamp_).toSec(), stampedPose.v_ ));
//        std::cout << Time(stampedPose.timestamp_).toNSec() << std::endl;

        if(i % 5  == 0){
            samples.push_back(std::pair<double,Pose<double>>(Time(stampedPose.timestamp_).toSec(), pose ) );
        }

    }

    poseSpline.initialPoseSpline(samples);

    /*
     *  Test: pose spline evalPoseSpline
     */

    for(auto pair : queryMeas){
        if(poseSpline.isTsEvaluable(pair.first)){
            Pose<double> query = poseSpline.evalPoseSpline(pair.first);
//            std::cout <<"Gt:    "<<pair.second.r().transpose() << " " << pair.second.q().transpose()<<std::endl;
//            std::cout <<"Query: "<<query.r().transpose()<<" "<< query.q().transpose()<< std::endl << std::endl;

            GTEST_ASSERT_LT((pair.second.r() - query.r()).norm(), 5e-2);
            GTEST_ASSERT_LT((pair.second.q() - query.q()).norm(), 5e-2);
        }

    }

    /*
     * Check PoseSpline linear velocity
     *
     * Passed!
     */

    for(auto pair : queryVelocityMeas){
        if(poseSpline.isTsEvaluable(pair.first)){
            Eigen::Vector3d query = poseSpline.evalLinearVelocity(pair.first);

//            std::cout <<"Gt:    "<< pair.second.transpose()<<std::endl;
//            std::cout <<"Query: "<< query.transpose()<< std::endl << std::endl;
//            ofs_debug << pair.second.transpose() << " " << query.transpose() << std::endl;
            GTEST_ASSERT_LT((pair.second - query).norm(), 0.1);

        }
    }


//    std::ofstream ofs_debug("/home/pang/debug.txt");

    for(uint i = start; i <end; i++){
        StampedImu stampedImu = testSample.imu_vec_.at(i);
        auto ts = Time(stampedImu.timestamp_).toSec();
        if(poseSpline.isTsEvaluable(ts)){
            Eigen::Vector3d evalAccel = poseSpline.evalLinearAccelerator(ts);

            // Note: the accelerator is noisy,
//            std::cout <<"Gt:    "<< stampedImu.accel_.transpose()<<std::endl;
//            std::cout <<"Query: "<< evalAccel.transpose()<< std::endl << std::endl;

//            ofs_debug << stampedImu.accel_.transpose() << " " << evalAccel.transpose() << std::endl;

        }
    }
//        ofs_debug.close();

//



}
