
#include <iostream>
#include <fstream>

//#include <gtest/gtest.h>
#include "PoseSpline/QuaternionSpline.hpp"
#include <PoseSpline/QuaternionSplineUtility.hpp>
#include <PoseSpline/PoseSpline.hpp>
#include <PoseSpline/VectorSpaceSpline.hpp>
#include "csv.h"
#include "PoseSpline/Time.hpp"
#include "extern/project_error.h"
#include "PoseSpline/PoseLocalParameter.hpp"
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

double uniform_rand(double lowerBndr, double upperBndr)
{
    return lowerBndr + ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}



int main() {
    std::string pose_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
    std::string imu_meas_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/imu0/data.csv";

//    TestSample testSample;
//    testSample.readStates(pose_file);
//    testSample.readImu(imu_meas_file);
//
    int image_width = 640;
    int image_height = 480;
    double focal = 500;
    double fx = 200;
    double fy = 200;
    double cx = image_width/2;
    double cy = image_height/2;

//
//    PoseSpline poseSpline(1.0);
//
//    // simulate
//    int num_landmarks = 10000;
//    std::vector<Eigen::Vector3d> landmarks(num_landmarks);
//    for (auto i : landmarks) {
//        i(0) = uniform_rand(-10, 10);
//        i(1) = uniform_rand(-10, 10);
//        i(2) = uniform_rand(-10, 10);
//    }
//
//    std::vector<StampedPose> T_WC_vec;
//
//    int sample_start = 1000;
//
//    for (int i = sample_start;i < testSample.states_vec_.size(); i+= 400) {
//        StampedPose stampedPose = testSample.states_vec_.at(i);
//        poseSpline.addControlPointsUntil(Time(stampedPose.timestamp_).toSec());
//        T_WC_vec.push_back(stampedPose);
//    }
//
//    std::cout<< poseSpline.getControlPointNum() << std::endl;
////    std::cout << i << std::endl
//    int num_pose = T_WC_vec.size();
//    std::cout<< num_pose << std::endl;
//
//    std::cout << testSample.states_vec_.size() << std::endl;
//
//
//    std::cout << "simulating ..." << std::endl;
//    typedef std::vector<std::pair<int, Eigen::Vector2d>> Observations;
//    std::vector<Observations> observation_per_landmark;
//    for (auto pt : landmarks) {
//        Observations obs;
//        for (int i = 0 ; i < T_WC_vec.size(); i++) {
//            auto stampedPose = T_WC_vec.at(i);
//            Pose<double> T_WC(stampedPose.t_, stampedPose.q_);
//            Eigen::Vector3d Cp = T_WC.inverse()*pt;
//            if(Cp(2) < 0) continue;
//            Eigen::Vector2d bearing(Cp(0)/Cp(2), Cp(1)/Cp(2));
//            Eigen::Vector2d uv(fx*bearing(0) + cx, fy*bearing(1) + cy);
//            if (uv(0) > 0 && uv(0) < image_width && uv(1) > 0 && uv(1) < image_height) {
//                obs.push_back(std::make_pair(i, bearing));
//
////
////                Eigen::Vector2d residual;
////
////                ProjectError projectError(Eigen::Vector3d(bearing(0), bearing(1), 1.0));
////                double* parameters[2] = {T_WC.parameterPtr(), pt.data()};
////
////                projectError.Evaluate(parameters, residual.data(), NULL);
////
////                if (residual.norm() > 1e-6) {
////                    std::cout << "too large project error" << std::endl;
////                }
//            }
//        }
//        observation_per_landmark.push_back(obs);
//    }
//
//    int average_cnt = 0;
//    for(auto i : observation_per_landmark) {
//        average_cnt += i.size();
//    }
//    std::cout <<"average obs for " << observation_per_landmark.size()
//                <<" is " << (double)average_cnt / observation_per_landmark.size()  << std::endl;
//
//    // test project error
//    {
//        ceres::Problem problem;
//
//        std::vector<Pose<double>> pose_vector_param;
//        std::vector<Eigen::Vector3d> landmark_vector_param;
//
//        for (auto pose: T_WC_vec) {
//            Pose<double> T_WC(pose.t_, pose.q_);
//            pose_vector_param.push_back(T_WC);
//            PoseLocalParameter *poseLocalParameter = new PoseLocalParameter;
//
//            problem.AddParameterBlock(pose_vector_param.back().parameterPtr(),7,poseLocalParameter);
//            Eigen::Map<Eigen::Matrix<double,7,1>> map(T_WC.parameterPtr());
////            std::cout << map.transpose() << std::endl;
//            std::cout << std::hex << pose_vector_param.back().parameterPtr() << " " << map.transpose() << std::endl;
//        }
//
//        for (int i = 0; i < landmarks.size(); i++) {
//            landmark_vector_param.push_back(landmarks.at(i));
//            problem.AddParameterBlock(landmark_vector_param.back().data(),3);
//
//            // add constraints
//            Observations obs = observation_per_landmark.at(i);
//            for (auto ob : obs) {
//
//                auto bearing = ob.second;
//                ProjectError* projectError = new ProjectError(Eigen::Vector3d(bearing(0), bearing(1), 1.0));
//                double* parameters[2] = {pose_vector_param.at(ob.first).parameterPtr(),
//                                         landmark_vector_param.at(i).data()};
//
////                Eigen::Vector2d residual;
////                projectError->Evaluate(parameters, residual.data(), NULL);
////
////                if (residual.norm() > 1e-6) {
////                    std::cout << "too large project error" << std::endl;
////                }
//
//                problem.AddResidualBlock(projectError, NULL,
//                                         parameters[0],
//                                         parameters[1]);
//
//                Eigen::Map<Eigen::Matrix<double,7,1>> map(parameters[0]);
//                std::cout << std::hex << parameters[0] << " " <<map.transpose() << std::endl;
//            }
//        }
//
//        ceres::Solver::Options options;
//        options.minimizer_progress_to_stdout = true;
//        options.max_solver_time_in_seconds = 30;
//        options.linear_solver_type = ceres::SPARSE_SCHUR;
//        options.minimizer_progress_to_stdout = true;
//        options.parameter_tolerance = 1e-4;
//        ceres::Solver::Summary summary;
//        ceres::Solve(options, &problem, &summary);
//        std::cout << summary.FullReport() << std::endl;
//
//    }



    Pose<double> T_WC0, T_WC1;

    T_WC1.set(Eigen::Vector3d(1,0,0), unitQuat<double>());

    int num_landmarks = 100;
    std::vector<Eigen::Vector4d> co_visual_obs;
    std::vector<Eigen::Vector3d> Wp_vec;
    // generate observations
    for(int i = 0; i < num_landmarks; i++) {
        double u = uniform_rand(0, image_width);
        double v = uniform_rand(0, image_height);
//        std::cout << u << " " << v << std::endl;
        Eigen::Vector2d bearing((u - cx)/fx, (v - cy)/ fy);

        double z = uniform_rand(0, 10);
        Eigen::Vector3d C0p(bearing(0)*z, bearing(1)*z, z);

        Eigen::Vector3d Wp = T_WC0*C0p;

        Eigen::Vector3d C1p = T_WC1.inverse() * Wp;

        Eigen::Vector2d C1_bearing(C1p(0)/C1p(2), C1p(1)/C1p(2));
        Eigen::Vector2d C1uv(C1_bearing(0)*fx + cx, C1_bearing(1)*fy + cy);

        if (C1uv(0) > 0 && C1uv(0) < image_width && C1uv(1) > 0 && C1uv(1) < image_height) {
            Wp_vec.push_back(Wp);
            Eigen::Vector4d co_obs;
            co_obs << bearing, C1_bearing;
            co_visual_obs.push_back(co_obs);
        }

    }

    std::cout << Wp_vec.size() << " " << co_visual_obs.size() << std::endl;

    Pose<double> T_WC0_param, T_WC1_param, noise;
    noise.setRandom(0.3, 0.5);
    T_WC0_param = T_WC0*noise;
    T_WC1_param = T_WC1*noise;

    std::vector<Pose<double>> T_WC_vec;
    T_WC_vec.push_back(T_WC0_param);
    T_WC_vec.push_back(T_WC1_param);

    std::vector<Eigen::Vector3d> landmark_params;
    for (auto i : Wp_vec) {
        landmark_params.push_back(i);
    }

    ceres::Problem problem;
    PoseLocalParameter *poseLocalParameter = new PoseLocalParameter;
//
    problem.AddParameterBlock(T_WC_vec.at(0).parameterPtr(),7,poseLocalParameter);
    problem.AddParameterBlock(T_WC_vec.at(1).parameterPtr(),7,poseLocalParameter);

    for (int i  = 0; i < Wp_vec.size(); i ++) {
        problem.AddParameterBlock(Wp_vec.at(i).data(),3);
        auto bearing = co_visual_obs.at(i);
        ProjectError* projectError0 = new ProjectError(Eigen::Vector3d(bearing(0), bearing(1), 1.0));
        ProjectError* projectError1 = new ProjectError(Eigen::Vector3d(bearing(2), bearing(3), 1.0));

        problem.AddResidualBlock(projectError0, NULL,
                                 T_WC_vec.at(0).parameterPtr(),
                                 Wp_vec.at(i).data());

        problem.AddResidualBlock(projectError1, NULL,
                                 T_WC_vec.at(1).parameterPtr(),
                                 Wp_vec.at(i).data());


    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 30;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.parameter_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
//


    return 0;
}