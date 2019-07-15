#include <iostream>
#include <fstream>

#include "PoseSpline/BSplineBase.hpp"
#include "PoseSpline/QuaternionSpline.hpp"
#include <PoseSpline/QuaternionSplineUtility.hpp>
#include <gtest/gtest.h>
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



TEST( Spline , quaternionSplineInitialization) {
////
////    /*
////     *  Test getOmegaFromTwoQuaternion
////     *  Passed!
////     */
////
////
////    for(uint i = 1; i < data.size(); i++){
////        std::pair<double, Quaternion> q1 = getSample(data,i);
////        std::pair<double, Quaternion> q0 = getSample(data,i-1);
////        double dt = q1.first - q0.first;
////
////        Eigen::Vector3d omega = getOmegaFromTwoQuaternion(q0.second,q1.second,dt);
////        //ofs_debug << omega.transpose()<<std::endl;
////
////    }
////
////    /*
////     *  Test QSUtility::w_a
////     *  Passed!
////     */
////
////    for(uint i = 1; i < data.size()-1; i++){
////        std::pair<double, Quaternion> q1 = getSample(data,i);
////        std::pair<double, Quaternion> q0 = getSample(data,i-1);
////        std::pair<double, Quaternion> q2 = getSample(data,i+1);
////        double dt = q2.first - q0.first;
////
////        Quaternion dotQ0 = (q2.second - q0.second)/dt;
////
////        Eigen::Vector3d omega = QSUtility::w(q0.second,dotQ0);
////
////        //ofs_debug << omega.transpose()<<std::endl;
////
////    }
////
////
////    /*
////     * Test QuaternionSpline evalQuatSplineDerivate
////     * Passed!
////     */
////
////
////    for(uint i = 1; i < data.size()-1; i++){
////
////        std::pair<double, Quaternion> q1 = getSample(data,i);
////        if(qspline.isTsEvaluable(q1.first)){
////            std::pair<double, Quaternion> q0 = getSample(data,i-1);
////            std::pair<double, Quaternion> q2 = getSample(data,i+1);
////            double dt20 = q2.first - q0.first;
////            double dt10 = q1.first - q0.first;
////            double dt21 = q2.first - q1.first;
////
////            // num-diff 1-order
////            Quaternion num_dotQ1 = (q2.second - q0.second)/dt20;
////
////            Quaternion num_dotQ10 = (q1.second - q0.second)/dt10;
////            Quaternion num_dotQ21 = (q2.second - q1.second)/dt21;
////
////            // num-diff 2-order
////            Quaternion num_dot_dot_Q1 = (num_dotQ21 - num_dotQ10)/(dt20/2.0);
////
////
////            Quaternion Q1, dotQ1, dot_dotQ1;
////            qspline.evalQuatSplineDerivate(q1.first,Q1.data(),dotQ1.data(),dot_dotQ1.data());
////
////
////            if(qspline.isTsEvaluable(q2.first)){
////
////                Quaternion dotQ2 = qspline.evalDotQuatSpline(q2.first);
////                Quaternion another_num_dot_dot_Q1 = (dotQ2 - dotQ1)/(dt21); // a little delay
////
////
////
////                ofs_debug << q0.second.transpose()<<" "<<Q1.transpose()<<" ";
////                ofs_debug << num_dotQ1.transpose()<<" "<<dotQ1.transpose()<<" ";
////                ofs_debug << num_dot_dot_Q1.transpose()<<" "<<another_num_dot_dot_Q1.transpose()<<" "<<dot_dotQ1.transpose()<<std::endl;
////
////
//////                std::cout << q0.second.transpose()<<" "<<Q1.transpose()<<" ";
//////                std::cout << num_dotQ1.transpose()<<" "<<dotQ1.transpose()<<" ";
//////                std::cout << num_dot_dot_Q1.transpose()<<" "<<dot_dotQ1.transpose()<<std::endl;
////
////            }
////
////
////
////
////            //std::cout << num_dotQ0.transpose()<<" "<<dotQ0.transpose()<<std::endl;
////
////        }
////    }
////
////
////
////
////
////
//    ofs_debug.close();
////

    std::string pose_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
    std::string imu_meas_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/imu0/data.csv";

    TestSample testSample;
    testSample.readStates(pose_file);
    testSample.readImu(imu_meas_file);

    int start  = 0;
    int end = testSample.states_vec_.size()/5;

    QuaternionSpline qspline(0.1);
    std::vector<std::pair<double,Quaternion>> samples, queryMeas;

    for(uint i = start; i < end; i++){
        StampedPose stampedPose = testSample.states_vec_.at(i);

        Eigen::Quaterniond QuatHamilton(stampedPose.q_);
        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
        Quaternion QuatJPL = rotMatToQuat(R);

        queryMeas.push_back(std::pair<double,Quaternion>(Time(stampedPose.timestamp_).toSec(), QuatJPL ) );

        if(i % 5  == 0){
            samples.push_back(std::pair<double,Quaternion>(Time(stampedPose.timestamp_).toSec(), QuatJPL ) );
        }
    }

    qspline.initialQuaternionSpline(samples);

    for(auto pair : queryMeas) {
        if (qspline.isTsEvaluable(pair.first)) {
            auto query = qspline.evalQuatSpline(pair.first);

            GTEST_ASSERT_LT((pair.second - query).norm(), 5e-2);
        }
    }

//    std::ofstream ofs_debug("/home/pang/debug.txt");

    for(uint i = start; i < end; i++){
        StampedImu stampedImu = testSample.imu_vec_.at(i);
        auto ts = Time(stampedImu.timestamp_).toSec();
        if(qspline.isTsEvaluable(ts)){
            Eigen::Vector3d query = qspline.evalOmega(ts);

//            std::cout <<"Gt:    "<< stampedImu.gyro_.transpose()<<std::endl;
//            std::cout <<"Query: "<< query.transpose()<< std::endl << std::endl;

//            ofs_debug << stampedImu.gyro_.transpose() << " " << query.transpose() << std::endl;

            std::cout << i << "/" << end << std::endl;
        }
    }

//        ofs_debug.close();

}
