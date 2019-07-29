
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
    double fx = 200;
    double fy = 200;
    double cx = image_width/2;
    double cy = image_height/2;


    int num_pose = 100 ;
    int num_landmark = 100;

    std::vector<Eigen::Vector3d> landmarks;
    for (auto i = 0; i < num_landmark; i++) {
        Eigen::Vector3d pt(uniform_rand(-10, 10), uniform_rand(-10, 10),uniform_rand(0, 50) );
        landmarks.push_back(pt);
    }


    double delta = 2*M_PI/100;
    std::vector<Pose<double>> poses;
    for (int i = 0; i < num_pose; i++) {
        Eigen::Vector3d t_WC(sin(delta* i), cos(delta*i), cos(delta*i) );
        Quaternion q_WC;
        q_WC << 0.2*sin(delta* i), 0.2*cos(delta* i), 0.1 * sin(delta* i), 1.0;
        q_WC = quatNorm(q_WC);
        poses.push_back(Pose<double>(t_WC, q_WC));

    }

    // simulate obs
    typedef std::vector<std::pair<int, Eigen::Vector2d>> Observations;
    std::vector<Observations> observation_per_landmark;
    for (int i = 0; i < num_landmark; i++) {
        Eigen::Vector3d Wp = landmarks.at(i);
        Observations obs;
        for (int j = 0;j < num_pose; j++) {
            Pose<double> T_WC = poses.at(j);
            Eigen::Vector3d Cp = T_WC.inverse()*Wp;
            Eigen::Vector2d bearing(Cp(0)/Cp(2), Cp(1)/Cp(2));
            Eigen::Vector2d C1uv(bearing(0)*fx + cx, bearing(1)*fy + cy);

            if (C1uv(0) > 0 && C1uv(0) < image_width && C1uv(1) > 0 && C1uv(1) < image_height) {
                obs.push_back(std::make_pair(j, bearing));
            }

        }
        observation_per_landmark.push_back(obs);
    }

    int average_cnt = 0;
    int min = 10000;
    int max = 0;
    for(auto i : observation_per_landmark) {
        average_cnt += i.size();
        if (i.size() < min) {
            min = i.size();
        }
        if(i.size() > max) {
            max = i.size();
        }
    }

    std::cout <<"average obs for " << observation_per_landmark.size()
                <<" is " << (double)average_cnt / observation_per_landmark.size()
                << " with min nad max : " << min << " " <<  max  << std::endl;



    std::vector<Eigen::Vector3d> landmarks_param;
    for (auto i : landmarks) {
        landmarks_param.push_back(i);
    }
    std::vector<Pose<double>> poses_param;
    for (auto i : poses) {
        // set noise
        Pose<double> noise;
        noise.setRandom(0.2, 0.2);
        poses_param.push_back(i * noise);
    }

    ceres::Problem problem;


    for (int i  = 0; i < num_pose; i++) {
        PoseLocalParameter *poseLocalParameter = new PoseLocalParameter;
        problem.AddParameterBlock(poses_param.at(i).parameterPtr(),7,poseLocalParameter);
    }

    for (int i = 0; i < landmarks.size(); i++) {
        // add constraints
        Observations obs = observation_per_landmark.at(i);
        if (obs.size() <  2) continue;
        problem.AddParameterBlock(landmarks_param.back().data(),3);

        std::cout << "add residuals realted to "<< i << "th landmark: " << obs.size() << std::endl;
        for (auto ob : obs) {
            auto bearing = ob.second;
            ProjectError* projectError = new ProjectError(Eigen::Vector3d(bearing(0), bearing(1), 1.0));


            problem.AddResidualBlock(projectError, NULL,
                                     poses_param.at(ob.first).parameterPtr(),
                                     landmarks_param.at(i).data());

//            Eigen::Map<Eigen::Matrix<double,7,1>> map(parameters[0]);
//            std::cout << std::hex << parameters[0] << " " <<map.transpose() << std::endl;
        }
    }

    std::cout << "start to solve ... " << std::endl;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 30000;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.parameter_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;




    return 0;
}