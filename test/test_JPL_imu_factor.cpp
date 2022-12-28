
#include "extern/JPL_imu_error.h"
#include "extern/vinsmono_imu_error.h"
#include "PoseSpline/maplab/maplab_imu_factor.h"
#include "internal/pose_local_parameterization.h"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/maplab/pose_param_jpl.h"


#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseSpline.hpp"
#include "PoseSpline/Time.hpp"
#include "csv.h"

#include <ceres/gradient_checker.h>

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

int main(){
    JPL::ImuParam imuParam;
    maplab::ImuParam maplab_imu_param;
    hamilton::ImuParam imuParam1;
    std::string pose_file =
            "/home/pang/datasets/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
    std::string imu_meas_file =
            "/home/pang/datasets/euroc/MH_01_easy/mav0/imu0/data.csv";

    TestSample testSample;
    testSample.readStates(pose_file);
    testSample.readImu(imu_meas_file);

    int start  = testSample.states_vec_.size()* 5/10;
    int end = testSample.states_vec_.size()* 6/10;

    PoseSpline poseSpline(0.1);

    std::vector<std::pair<double,Pose<double>>> samples, queryMeas;
    for(uint i = start; i <end; i++){
        StampedPose stampedPose = testSample.states_vec_.at(i);
        Eigen::Quaterniond QuatHamilton(stampedPose.q_);
        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
        Quaternion QuatJPL = rotMatToQuat(R);

        Pose<double> pose(stampedPose.t_, QuatJPL);
        queryMeas.push_back(std::pair<double,Pose<double>>(Time(stampedPose.timestamp_).toSec(), pose ) );

        if(i % 5  == 0){
            samples.push_back(std::pair<double,Pose<double>>(Time(stampedPose.timestamp_).toSec(), pose ) );
        }

    }

    poseSpline.initialPoseSpline(samples);


    std::vector<StampedPose> simulated_states;
    std::vector<StampedImu> simulated_imus;
    const Eigen::Vector3d G(0.0, 0.0, 9.81);

    for(auto pair : queryMeas){
        if(poseSpline.isTsEvaluable(pair.first)){
            Pose<double> query_pose = poseSpline.evalPoseSpline(pair.first);
            Eigen::Vector3d query_velocity = poseSpline.evalLinearVelocity(pair.first);
            Eigen::Vector3d query_omega = poseSpline.evalOmega(pair.first);
            Eigen::Vector3d query_accel = poseSpline.evalLinearAccelerator(pair.first, G);

            StampedPose stampedPose;
            stampedPose.timestamp_ = Time(pair.first).toNSec();
            stampedPose.t_ = query_pose.translation();
            QuaternionTemplate<double> q = query_pose.rotation();
            stampedPose.q_ = Eigen::Quaterniond(q[3], q[0], q[1], q[2]);
            stampedPose.v_ = query_velocity;

            simulated_states.push_back(stampedPose);

            StampedImu stampedImu;
            stampedImu.timestamp_ = Time(pair.first).toNSec();
            stampedImu.gyro_ = query_omega;
            stampedImu.accel_ = query_accel;

            simulated_imus.push_back(stampedImu);

        }
    }

    std::cout << "test on : " << simulated_imus.size() << std::endl;
    int imu_integrated_cnt = 500;
    for (int i = imu_integrated_cnt ; i < simulated_imus.size(); i+=imu_integrated_cnt) {
        QuaternionTemplate<double> JPL_q_WI0, JPL_q_WI1;
        Eigen::Vector4d maplab_q_I0W, maplab_q_I1W;
        Eigen::Quaterniond hamilton_q_WI0, hamilton_q_WI1;
        Eigen::Vector3d t_WI0, t_WI1;
        Eigen::Vector3d v0, v1;
        Eigen::Vector3d ba0, ba1;
        Eigen::Vector3d bg0, bg1;
        JPL_q_WI0 = simulated_states.at(i - imu_integrated_cnt).q_.coeffs();
        maplab_q_I0W = common::quaternionInverseJPL(JPL_q_WI0);
        hamilton_q_WI0 = Eigen::Quaterniond(quatToRotMat(JPL_q_WI0));
        t_WI0 = simulated_states.at(i - imu_integrated_cnt).t_;
        v0 = simulated_states.at(i - imu_integrated_cnt).v_;
        ba0 << 0,0,0; bg0 << 0,0,0;

        JPL_q_WI1 = simulated_states.at(i).q_.coeffs();
        maplab_q_I1W = common::quaternionInverseJPL(JPL_q_WI1);
        hamilton_q_WI1 = Eigen::Quaterniond(quatToRotMat(JPL_q_WI1));

        t_WI1 = simulated_states.at(i).t_;
        v1 = simulated_states.at(i).v_;
        ba1 << 0,0,0; bg1 << 0,0,0;

        uint64_t t0 = simulated_imus.at(i - imu_integrated_cnt).timestamp_;

        Eigen::Vector3d gyro = simulated_imus.at(i - imu_integrated_cnt).gyro_;
        Eigen::Vector3d accel = simulated_imus.at(i - imu_integrated_cnt).accel_;

        std::shared_ptr<JPL::IntegrationBase> JPL_intergrateImu = std::make_shared<JPL::IntegrationBase>(
                accel, gyro, ba0, bg0, imuParam);
                
        std::shared_ptr<maplab::IntegrationBase> maplab_intergrateImu = std::make_shared<maplab::IntegrationBase>(
                accel, gyro, ba0, bg0, maplab_imu_param);

        std::shared_ptr<hamilton::IntegrationBase> hamilton_intergrateImu = std::make_shared<hamilton::IntegrationBase>(
                accel, gyro, ba0, bg0, imuParam1);

        for (int j = i - imu_integrated_cnt + 1; j < i; j++) {
            uint64_t tk = simulated_imus.at(j).timestamp_;
            Eigen::Vector3d gyro = simulated_imus.at(j).gyro_;
            Eigen::Vector3d accel = simulated_imus.at(j).accel_;

            auto dt = (double)(tk - t0)/ 1e9;
            t0 = tk;
            JPL_intergrateImu->push_back(dt, accel,gyro);
            maplab_intergrateImu->push_back(dt, accel, gyro);
            hamilton_intergrateImu->push_back(dt, accel,gyro);
        }

        QuaternionTemplate<double> JPL_delta_q = JPL_intergrateImu->delta_q;
        Eigen::Vector4d maplab_delta_q = maplab_intergrateImu->delta_q_;
        QuaternionTemplate<double> JPL_q_I1I0 = quatMult(quatInv(JPL_q_WI1),JPL_q_WI0);
        std::cout << std::endl;
        std::cout << "JPL_delta_q         : " << JPL_delta_q.transpose() << std::endl;
        std::cout << "maplab_delta_q      : " << maplab_delta_q.transpose() << std::endl;
        std::cout << "JPL_q_I1I0          : " << JPL_q_I1I0.transpose() << std::endl;

        Eigen::Quaterniond hmailton_delta_q = hamilton_intergrateImu->delta_q;
        Eigen::Quaterniond hmailton_q_I0I1 = hamilton_q_WI0.inverse()*hamilton_q_WI1;
        std::cout << "hmailton_delta_q: " << hmailton_delta_q.coeffs().transpose() << std::endl;
        std::cout << "hmailton_q_I0I1 : " << hmailton_q_I0I1.coeffs().transpose() << std::endl;


        Eigen::Vector3d JPL_delta_p = JPL_intergrateImu->delta_p;
        Eigen::Vector3d JPL_delta_v = JPL_intergrateImu->delta_v;
        std::cout << "JPL_delta_p     : " << JPL_delta_p.transpose() << std::endl;
        std::cout << "JPL_delta_v     : " << JPL_delta_v.transpose() << std::endl;

        Eigen::Vector3d maplab_delta_p = maplab_intergrateImu->delta_p_;
        Eigen::Vector3d maplab_delta_v = maplab_intergrateImu->delta_v_;
        std::cout << "maplab_delta_p     : " << maplab_delta_p.transpose() << std::endl;
        std::cout << "maplab_delta_v     : " << maplab_delta_v.transpose() << std::endl;


        Eigen::Vector3d hamilton_delta_p = hamilton_intergrateImu->delta_p;
        Eigen::Vector3d hamilton_delta_v = hamilton_intergrateImu->delta_v;
        std::cout << "hamilton_delta_p: " << hamilton_delta_p.transpose() << std::endl;
        std::cout << "hamilton_delta_v: " << hamilton_delta_v.transpose() << std::endl;

        Eigen::Matrix<double,15,15> JPL_jacobian, JPL_covariance;
        JPL_jacobian = JPL_intergrateImu->jacobian;
        JPL_covariance = JPL_intergrateImu->covariance;

        Eigen::Matrix<double,15,15> maplab_jacobian, maplab_covariance;
        maplab_jacobian = maplab_intergrateImu->jacobian_;
        maplab_covariance = maplab_intergrateImu->covariance_;

        Eigen::Matrix<double,15,15> hamilton_jacobian, hamilton_covariance;
        hamilton_jacobian = hamilton_intergrateImu->jacobian;
        hamilton_covariance = hamilton_intergrateImu->covariance;

        CHECK_EQ((JPL_jacobian - maplab_jacobian).squaredNorm() < 1e-6, true) << "jacobian error is large";
        CHECK_EQ((JPL_covariance - maplab_covariance).squaredNorm() < 1e-6, true) << "covariance error is large";
        CHECK_EQ((JPL_jacobian - hamilton_jacobian).squaredNorm() < 1e-6, true) << "jacobian error is large";
        CHECK_EQ((JPL_covariance - hamilton_covariance).squaredNorm() < 1e-6, true) << "covariance error is large";


        Eigen::Matrix<double,15,1> JPL_residuals, hamilton_residuals, maplab_residuals;
        JPL_residuals = JPL_intergrateImu->evaluate(t_WI0, JPL_q_WI0, v0,ba0,bg0, t_WI1, JPL_q_WI1, v1,ba1,bg1);
        hamilton_residuals = hamilton_intergrateImu->evaluate(t_WI0, hamilton_q_WI0, v0,ba0,bg0, t_WI1, hamilton_q_WI1, v1,ba1,bg1);
        maplab_residuals = maplab_intergrateImu->evaluate(t_WI0, maplab_q_I0W, v0, ba0, bg0, t_WI1, maplab_q_I1W, v1,ba1,bg1);

        std::cout << "JPL_residuals     :" << JPL_residuals.transpose() << std::endl;
        std::cout << "hamilton_residuals:" << hamilton_residuals.transpose() << std::endl;
        std::cout << "maplab_residuals  :" << maplab_residuals.transpose() << std::endl;

        CHECK_EQ((JPL_residuals - hamilton_residuals).squaredNorm() < 1e-3, true) << "residual error is large";
        CHECK_EQ((JPL_residuals - maplab_residuals).squaredNorm() < 1e-3, true) << "residual error is large";

        Eigen::Matrix<double,7,1> JPL_T0, JPL_T1;
        JPL_T0 << t_WI0, JPL_q_WI0;
        JPL_T1 << t_WI1, JPL_q_WI1;

        Eigen::Matrix<double,7,1> maplab_T0, maplab_T1;
        maplab_T0 << maplab_q_I0W, t_WI0;
        maplab_T1 << maplab_q_I1W, t_WI1;

        Eigen::Matrix<double,7,1> hamilton_T0, hamilton_T1;
        hamilton_T0 << t_WI0, hamilton_q_WI0.coeffs();
        hamilton_T1 << t_WI1, hamilton_q_WI1.coeffs();
        Eigen::Matrix<double,9,1> sb0, sb1;
//        ba0 << 0.1,-0.1,-0.2;
        bg0 << 0.005, -0.0051, -0.00512;

        ba1 << 0.1,0.1,0.2;
        bg1 << 0.1,0.1,0.2;

        sb0 << v0, ba0, bg0;
        sb1 << v1, ba1, bg1;

        double* JPL_parameters[4] = {JPL_T0.data(), sb0.data(), JPL_T1.data(), sb1.data()};
        double* hamilton_parameters[4] = {hamilton_T0.data(), sb0.data(), hamilton_T1.data(), sb1.data()};
        double* maplab_parameters[8] = {maplab_T0.data(), bg0.data(), v0.data(), ba0.data(), 
                                        maplab_T1.data(), bg1.data(), v1.data(), ba1.data()};


        JPL::IMUFactor JPL_Imufactor(JPL_intergrateImu.get());
        maplab::IMUFactor maplab_Imufactor(maplab_intergrateImu);
        hamilton::IMUFactor hamilton_Imufactor(hamilton_intergrateImu.get());

        JPL_Imufactor.Evaluate(JPL_parameters, JPL_residuals.data(), NULL);
        maplab_Imufactor.Evaluate(maplab_parameters, maplab_residuals.data(), NULL);
        hamilton_Imufactor.Evaluate(hamilton_parameters, hamilton_residuals.data(), NULL);

//        std::cout << "JPL_intergrateImu     -> correct_q: " << JPL_intergrateImu->corrected_delta_q.transpose() << std::endl;
//        std::cout << "hamilton_intergrateImu-> correct_q: " << hamilton_intergrateImu->corrected_delta_q.coeffs().transpose() << std::endl;

        std::cout << "factor hamilton_residuals:" << hamilton_residuals.transpose() << std::endl;
        std::cout << "factor JPL_residuals     :" << JPL_residuals.transpose() << std::endl;
        std::cout << "factor maplab_residuals  :" << maplab_residuals.transpose() << std::endl;

        //todo(Pang): the orientation part is not accurent enough
        // CHECK_EQ((JPL_residuals - hamilton_residuals).squaredNorm() < 1e-3, true) << "residual error is large";
        CHECK_EQ((JPL_residuals - maplab_residuals).squaredNorm() < 1e-3, true) << "residual error is large";


//         // jacnobian
//         ceres::LocalParameterization* localParameterization =
//                 new PoseLocalParameter;

//         ceres::NumericDiffOptions numeric_diff_options;
//         numeric_diff_options.ridders_relative_initial_step_size = 1e-3;

//         std::vector<const ceres::LocalParameterization*> local_parameterizations;
//         local_parameterizations.push_back(localParameterization);
//         local_parameterizations.push_back(NULL);
//         local_parameterizations.push_back(localParameterization);
//         local_parameterizations.push_back(NULL);

//         ceres::GradientChecker gradient_checker(
//                 &JPL_Imufactor, &local_parameterizations, numeric_diff_options);
//         ceres::GradientChecker::ProbeResults results;

//         gradient_checker.Probe(JPL_parameters, 1e-9, &results);
//         for (int i = 0; i < 4; i++) {
//             std::cout << "jacobian       " << i << " :  \n" << results.local_jacobians.at(i) << std::endl;
//             std::cout << "num jacobian:  " << i << " :  \n" << results.local_numeric_jacobians.at(i) << std::endl;
// //
//             CHECK_EQ((results.local_jacobians.at(i) - results.local_numeric_jacobians.at(i)).squaredNorm() < 1e-6, true) 
//                 << "jcaobian error is large: " 
//                 << (results.local_jacobians.at(i) - results.local_numeric_jacobians.at(i)).squaredNorm();


//         }


          // jacnobian
        ceres::LocalParameterization* localParameterization =
                new ceres_error_terms::JplPoseParameterization;

        ceres::NumericDiffOptions numeric_diff_options;
        numeric_diff_options.ridders_relative_initial_step_size = 1e-3;

        std::vector<const ceres::LocalParameterization*> local_parameterizations;
        local_parameterizations.push_back(localParameterization);
        local_parameterizations.push_back(NULL); // bg
        local_parameterizations.push_back(NULL); // v
        local_parameterizations.push_back(NULL); // ba
        local_parameterizations.push_back(localParameterization);
        local_parameterizations.push_back(NULL); // bg
        local_parameterizations.push_back(NULL); // v
        local_parameterizations.push_back(NULL); // ba

        ceres::GradientChecker gradient_checker(
                &maplab_Imufactor, &local_parameterizations, numeric_diff_options);
        ceres::GradientChecker::ProbeResults results;

        gradient_checker.Probe(maplab_parameters, 1e-9, &results);


        for (int i = 0; i < 7; i++) {
            std::cout << "jacobian       " << i << " :  \n" << results.local_jacobians.at(i) << std::endl;
            std::cout << "num jacobian:  " << i << " :  \n" << results.local_numeric_jacobians.at(i) << std::endl;
//
            CHECK_EQ((results.local_jacobians.at(i) - results.local_numeric_jacobians.at(i)).squaredNorm() < 1e-6, true) 
                << "jcaobian error is large: " 
                << (results.local_jacobians.at(i) - results.local_numeric_jacobians.at(i)).squaredNorm();


        }
       

     



    }


    return 0;
}