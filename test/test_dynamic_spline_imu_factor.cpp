
#include "extern/dynamic_spline_imu_error.h"
#include "extern/vinsmono_imu_error.h"
#include "internal/pose_local_parameterization.h"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/VectorSpaceSpline.hpp"
#include "PoseSpline/NumbDifferentiator.hpp"

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
    Eigen::Vector3d ba_;
    Eigen::Vector3d bg_;
};
class TestSample {
public:
    void readStates(const std::string& states_file) {
        io::CSVReader<11> in(states_file);
        in.read_header(io::ignore_extra_column, "#timestamp",
                       "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
                       "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
                       "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]");
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
    hamilton::ImuParam imuParam1;
    std::string pose_file =
            "/home/pang/disk/dataset/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";

    TestSample testSample;
    testSample.readStates(pose_file);

    int start  = testSample.states_vec_.size()* 5/10;
    int end = testSample.states_vec_.size()* 6/10;

    double spline_dt = 1.0;
    PoseSpline poseSpline(spline_dt);

    VectorSpaceSpline<3> baSpline(spline_dt);
    VectorSpaceSpline<3> bgSpline(spline_dt);

    std::vector<std::pair<double, Pose<double>>> samples, queryMeas;
    std::vector<std::pair<double, Eigen::Matrix<double,3,1>>> baSamples;
    for(uint i = start; i <end; i++){
        StampedPose stampedPose = testSample.states_vec_.at(i);
        Eigen::Quaterniond QuatHamilton(stampedPose.q_);
        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
        Quaternion QuatJPL = rotMatToQuat(R);

        Pose<double> pose(stampedPose.t_, QuatJPL);
        queryMeas.push_back(std::pair<double,Pose<double>>(Time(stampedPose.timestamp_).toSec(), pose ) );

        if(i % 5  == 0){
            samples.push_back(std::pair<double,Pose<double>>(Time(stampedPose.timestamp_).toSec(), pose ) );
            baSamples.push_back(
                    std::pair<double, Eigen::Matrix<double,3,1>>(Time(stampedPose.timestamp_).toSec(),
                            Eigen::Matrix<double,3,1>::Zero()) );
        }

    }

    poseSpline.initialPoseSpline(samples);
    baSpline.initialSpline(baSamples);
    bgSpline.initialSpline(baSamples);


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



            Eigen::Vector3d  query_ba = baSpline.evaluateSpline(pair.first);
            Eigen::Vector3d  query_bg = bgSpline.evaluateSpline(pair.first);
            stampedImu.ba_ = query_ba;
            stampedImu.bg_ = query_bg;


            simulated_imus.push_back(stampedImu);

        }
    }

    std::cout << "test on : " << simulated_imus.size() << std::endl;
    int imu_integrated_cnt = 20;
    for (int i = imu_integrated_cnt ; i < simulated_imus.size(); i+=imu_integrated_cnt) {
        double tt0 = Time(simulated_imus.at(i - imu_integrated_cnt).timestamp_).toSec();
        double tt1 = Time(simulated_imus.at(i).timestamp_).toSec();

        std::pair<real_t,int> time_pair0 = poseSpline.computeUAndTIndex(tt0);
        std::pair<real_t,int> time_pair1 = poseSpline.computeUAndTIndex(tt1);

        std::cout << "time_pair0 idx: " << time_pair0.second << " " <<  time_pair1.second << std::endl;
        std::cout << "time_pair0   u: " << time_pair0.first << " " <<  time_pair1.first << std::endl;


        QuaternionTemplate<double> JPL_q_WI0, JPL_q_WI1;
        Eigen::Vector3d t_WI0, t_WI1;
        Eigen::Vector3d v0, v1;
        Eigen::Vector3d ba0, ba1;
        Eigen::Vector3d bg0, bg1;
        JPL_q_WI0 = simulated_states.at(i - imu_integrated_cnt).q_.coeffs();
        auto temp =  poseSpline.evalPoseSpline(tt0).translation();
        std::cout << temp.transpose() << std::endl;
        t_WI0 = simulated_states.at(i - imu_integrated_cnt).t_;
        v0 = simulated_states.at(i - imu_integrated_cnt).v_;
        ba0 << 0,0,0; bg0 << 0,0,0;


        JPL_q_WI1 = simulated_states.at(i).q_.coeffs();
        t_WI1 = simulated_states.at(i).t_;
        v1 = simulated_states.at(i).v_;
        ba1 << 0,0,0; bg1 << 0,0,0;

        uint64_t t0 = simulated_imus.at(i - imu_integrated_cnt).timestamp_;

        Eigen::Vector3d gyro = simulated_imus.at(i - imu_integrated_cnt).gyro_;
        Eigen::Vector3d accel = simulated_imus.at(i - imu_integrated_cnt).accel_;

        std::shared_ptr<JPL::IntegrationBase> JPL_intergrateImu = std::make_shared<JPL::IntegrationBase>(
                accel, gyro, ba0, bg0, imuParam);
        std::shared_ptr<JPL::IntegrationBase> spline_intergrateImu = std::make_shared<JPL::IntegrationBase>(
                accel, gyro, ba0, bg0, imuParam);


        for (int j = i - imu_integrated_cnt + 1; j < i; j++) {
            uint64_t tk = simulated_imus.at(j).timestamp_;
            Eigen::Vector3d gyro = simulated_imus.at(j).gyro_;
            Eigen::Vector3d accel = simulated_imus.at(j).accel_;

            auto dt = (double)(tk - t0)/ 1e9;
            t0 = tk;
            JPL_intergrateImu->push_back(dt, accel, gyro);
            spline_intergrateImu->push_back(dt, accel, gyro);
        }

        JPL::IMUFactor JPL_imuFactor(JPL_intergrateImu.get());


        Eigen::Matrix<double,7,1> JPL_T0, JPL_T1;
        JPL_T0 << t_WI0, JPL_q_WI0;
        JPL_T1 << t_WI1, JPL_q_WI1;

        Eigen::Matrix<double,9,1> sb0, sb1;
        sb0 << v0, ba0, bg0;
        sb1 << v1, ba1, bg1;

//

        double* JPL_parameters[4] = {JPL_T0.data(), sb0.data(), JPL_T1.data(), sb1.data()};
        Eigen::VectorXd JPLresidual(15);
        JPL_imuFactor.Evaluate(JPL_parameters, JPLresidual.data(), NULL);




        int bidx = time_pair0.second - poseSpline.spline_order() + 1;

        if (time_pair0.second == time_pair1.second) {
            JPL::DynamicSplineIMUFactor<4> splineImuFactor(spline_intergrateImu, spline_dt, time_pair0.first, time_pair1.first);
            Eigen::Matrix<double,7,1> cp_T0(poseSpline.getControlPoint(bidx));
            Eigen::Matrix<double,7,1> cp_T1(poseSpline.getControlPoint(bidx+1));
            Eigen::Matrix<double,7,1> cp_T2(poseSpline.getControlPoint(bidx+2));
            Eigen::Matrix<double,7,1> cp_T3(poseSpline.getControlPoint(bidx+3));

//        std::cout << "t0: " << cp_T0.transpose() << std::endl;
//        std::cout << "t1: " << cp_T1.transpose() << std::endl;
//        std::cout << "t2: " << cp_T2.transpose() << std::endl;
//        std::cout << "t3: " << cp_T3.transpose() << std::endl;


            Eigen::Matrix<double,6,1> cp_bias0, cp_bias1, cp_bias2, cp_bias3;
            cp_bias0 << Eigen::Vector3d(baSpline.getControlPoint(bidx)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx));

            cp_bias1 << Eigen::Vector3d(baSpline.getControlPoint(bidx + 1)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx + 1 ));

            cp_bias2 << Eigen::Vector3d(baSpline.getControlPoint(bidx+2)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx+2));

            cp_bias3 << Eigen::Vector3d(baSpline.getControlPoint(bidx + 3)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx + 3));

            double* spline_parameters[8] = {cp_T0.data(), cp_T1.data(), cp_T2.data(), cp_T3.data(),
                                            cp_bias0.data(), cp_bias1.data(), cp_bias2.data(), cp_bias3.data()};
            Eigen::VectorXd spline_residual(15);
            splineImuFactor.evaluate(spline_parameters, spline_residual.data(), NULL);
            std::cout << "JPL residual   : " << JPLresidual.transpose() << std::endl;
            std::cout << "spline_residual: " << spline_residual.transpose() << std::endl;

            CHECK_EQ((spline_residual - JPLresidual).norm() < 1e-6, true) << "residuals are not consist";


            // jacnobian
            ceres::LocalParameterization* localParameterization =
                    new PoseLocalParameter;

            ceres::NumericDiffOptions numeric_diff_options;
            numeric_diff_options.ridders_relative_initial_step_size = 1e-3;

            std::vector<const ceres::LocalParameterization*> local_parameterizations;
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);


            ceres::DynamicAutoDiffCostFunction<JPL::DynamicSplineIMUFactor<4>, 4>* cost_function =
                    new ceres::DynamicAutoDiffCostFunction<JPL::DynamicSplineIMUFactor<4>, 4>(
                        new  JPL::DynamicSplineIMUFactor<4>(spline_intergrateImu, spline_dt, time_pair0.first, time_pair1.first));
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->SetNumResiduals(15);

            ceres::GradientChecker gradient_checker(
                    cost_function, &local_parameterizations, numeric_diff_options);
            ceres::GradientChecker::ProbeResults results;


            gradient_checker.Probe(spline_parameters, 1e-9, &results);
//        std::cout << "jacobian0:  \n" << results.local_jacobians.at(0) << std::endl;
//        std::cout << "num jacobian0:  \n" << results.local_numeric_jacobians.at(0) << std::endl;

//            std::cout << "jacobian0 error " << (results.local_jacobians.at(0) - results.local_numeric_jacobians.at(0)).norm() << std::endl;
            CHECK_EQ((results.local_jacobians.at(0) - results.local_numeric_jacobians.at(0)).norm() < 1e-6, true) << "jcaobian error is large";

//        std::cout << "jacobian1:  \n" << results.local_jacobians.at(1) << std::endl;
//        std::cout << "num jacobian1:  \n" << results.local_numeric_jacobians.at(1) << std::endl;

            CHECK_EQ((results.local_jacobians.at(1) - results.local_numeric_jacobians.at(1)).norm() < 1e-6, true) << "jcaobian error is large";
//
//            std::cout << "jacobian2:  \n" << results.local_jacobians.at(2) << std::endl;
//            std::cout << "num jacobian2:  \n" << results.local_numeric_jacobians.at(2) << std::endl;

            CHECK_EQ((results.local_jacobians.at(2) - results.local_numeric_jacobians.at(2)).norm() < 1e-6, true) << "jcaobian error is large";

//        std::cout << "jacobian3:  \n" << results.local_jacobians.at(3) << std::endl;
//        std::cout << "num jacobian3:  \n" << results.local_numeric_jacobians.at(3) << std::endl;

            CHECK_EQ((results.local_jacobians.at(3) - results.local_numeric_jacobians.at(3)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(4) - results.local_numeric_jacobians.at(4)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(5) - results.local_numeric_jacobians.at(5)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(6) - results.local_numeric_jacobians.at(6)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(7) - results.local_numeric_jacobians.at(7)).norm() < 1e-6, true) << "jcaobian error is large";


        } else {
            std::cout << "------------- 5 ----------------" << std::endl;
//            std::cout << "-- Pi: " << t_WI0.transpose() << std::endl;
//            std::cout << "-- Qi: " << JPL_q_WI0.transpose() << std::endl;
//            std::cout << "-- Vi: " << v0.transpose() << std::endl;
//            std::cout << "-- Bai: " << ba0.transpose() << std::endl;
//            std::cout << "-- Bgi: " << bg0.transpose() << std::endl;
//
//            std::cout << "-- Pj: " << t_WI1.transpose() << std::endl;
//            std::cout << "-- Qj: " << JPL_q_WI1.transpose() << std::endl;
//            std::cout << "-- Vj: " << v1.transpose() << std::endl;
//            std::cout << "-- Baj: " << ba1.transpose() << std::endl;
//            std::cout << "-- Bgj: " << bg1.transpose() << std::endl;

            JPL::DynamicSplineIMUFactor<5> splineImuCrossFactor(spline_intergrateImu, spline_dt, time_pair0.first, time_pair1.first);

            Eigen::Matrix<double,7,1> cp_T0(poseSpline.getControlPoint(bidx));
            Eigen::Matrix<double,7,1> cp_T1(poseSpline.getControlPoint(bidx+1));
            Eigen::Matrix<double,7,1> cp_T2(poseSpline.getControlPoint(bidx+2));
            Eigen::Matrix<double,7,1> cp_T3(poseSpline.getControlPoint(bidx+3));
            Eigen::Matrix<double,7,1> cp_T4(poseSpline.getControlPoint(bidx+4));

//        std::cout << "t0: " << cp_T0.transpose() << std::endl;
//        std::cout << "t1: " << cp_T1.transpose() << std::endl;
//        std::cout << "t2: " << cp_T2.transpose() << std::endl;
//        std::cout << "t3: " << cp_T3.transpose() << std::endl;


            Eigen::Matrix<double,6,1> cp_bias0, cp_bias1, cp_bias2, cp_bias3, cp_bias4;
            cp_bias0 << Eigen::Vector3d(baSpline.getControlPoint(bidx)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx));

            cp_bias1 << Eigen::Vector3d(baSpline.getControlPoint(bidx + 1)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx + 1 ));

            cp_bias2 << Eigen::Vector3d(baSpline.getControlPoint(bidx+2)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx+2));

            cp_bias3 << Eigen::Vector3d(baSpline.getControlPoint(bidx + 3)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx + 3));

            cp_bias4 << Eigen::Vector3d(baSpline.getControlPoint(bidx + 4)),
                    Eigen::Vector3d(bgSpline.getControlPoint(bidx + 4));

            double* spline_parameters[10] = {cp_T0.data(), cp_T1.data(), cp_T2.data(), cp_T3.data(), cp_T4.data(),
                                            cp_bias0.data(), cp_bias1.data(), cp_bias2.data(), cp_bias3.data(), cp_bias4.data()};
            Eigen::VectorXd spline_residual(15);
            splineImuCrossFactor.evaluate(spline_parameters, spline_residual.data(), NULL);
            std::cout << "JPL residual   : " << JPLresidual.transpose() << std::endl;
            std::cout << "spline_residual: " << spline_residual.transpose() << std::endl;

            CHECK_EQ((spline_residual - JPLresidual).norm() < 1e-6, true) << "residuals are not consist";


//
            // jacnobian
            ceres::LocalParameterization* localParameterization =
                    new PoseLocalParameter;

            ceres::NumericDiffOptions numeric_diff_options;
            numeric_diff_options.ridders_relative_initial_step_size = 1e-3;

            std::vector<const ceres::LocalParameterization*> local_parameterizations;
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(localParameterization);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);
            local_parameterizations.push_back(NULL);


            ceres::DynamicAutoDiffCostFunction<JPL::DynamicSplineIMUFactor<5>, 4>* cost_function =
                    new ceres::DynamicAutoDiffCostFunction<JPL::DynamicSplineIMUFactor<5>, 4>(
                            new  JPL::DynamicSplineIMUFactor<5>(spline_intergrateImu, spline_dt, time_pair0.first, time_pair1.first));
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(7);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->AddParameterBlock(6);
            cost_function->SetNumResiduals(15);

            ceres::GradientChecker gradient_checker(
                    cost_function, &local_parameterizations, numeric_diff_options);
            ceres::GradientChecker::ProbeResults results;


            gradient_checker.Probe(spline_parameters, 1e-9, &results);
//        std::cout << "jacobian0:  \n" << results.local_jacobians.at(0) << std::endl;
//        std::cout << "num jacobian0:  \n" << results.local_numeric_jacobians.at(0) << std::endl;

//            std::cout << "jacobian0 error " << (results.local_jacobians.at(0) - results.local_numeric_jacobians.at(0)).norm() << std::endl;
            CHECK_EQ((results.local_jacobians.at(0) - results.local_numeric_jacobians.at(0)).norm() < 1e-6, true) << "jcaobian error is large";

//            std::cout << "jacobian1:  \n" << results.local_jacobians.at(1) << std::endl;
//            std::cout << "num jacobian1:  \n" << results.local_numeric_jacobians.at(1) << std::endl;

            CHECK_EQ((results.local_jacobians.at(1) - results.local_numeric_jacobians.at(1)).norm() < 1e-6, true) << "jcaobian error is large";
//
//            std::cout << "jacobian2:  \n" << results.local_jacobians.at(2) << std::endl;
//            std::cout << "num jacobian2:  \n" << results.local_numeric_jacobians.at(2) << std::endl;

            CHECK_EQ((results.local_jacobians.at(2) - results.local_numeric_jacobians.at(2)).norm() < 1e-6, true) << "jcaobian error is large";

//        std::cout << "jacobian3:  \n" << results.local_jacobians.at(3) << std::endl;
//        std::cout << "num jacobian3:  \n" << results.local_numeric_jacobians.at(3) << std::endl;

            CHECK_EQ((results.local_jacobians.at(3) - results.local_numeric_jacobians.at(3)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(4) - results.local_numeric_jacobians.at(4)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(5) - results.local_numeric_jacobians.at(5)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(6) - results.local_numeric_jacobians.at(6)).norm() < 1e-6, true) << "jcaobian error is large";

            CHECK_EQ((results.local_jacobians.at(7) - results.local_numeric_jacobians.at(7)).norm() < 1e-6, true) << "jcaobian error is large";
            CHECK_EQ((results.local_jacobians.at(8) - results.local_numeric_jacobians.at(8)).norm() < 1e-6, true) << "jcaobian error is large";
            CHECK_EQ((results.local_jacobians.at(9) - results.local_numeric_jacobians.at(9)).norm() < 1e-6, true) << "jcaobian error is large";

        }






    }
    return 0;
}