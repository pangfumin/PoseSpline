
//#include "vins_estimator/utility/NumbDifferentiator.hpp"
#include "extern/spline_imu_error.h"
#include "internal/pose_local_parameterization.h"
#include "PoseSpline/NumbDifferentiator.hpp"

#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseSpline.hpp"
#include "PoseSpline/Time.hpp"
#include "csv.h"

void T2double(Eigen::Isometry3d& T,double* ptr){

    Eigen::Vector3d trans = T.matrix().topRightCorner(3,1);
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Quaterniond q(R);

    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);
    ptr[3] = q.x();
    ptr[4] = q.y();
    ptr[5] = q.z();
    ptr[6] = q.w();
}

void applyNoise(const Eigen::Isometry3d& Tin,Eigen::Isometry3d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.9*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.26*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;

    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3) =   Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}




class PosegraphErrorTermsEigen {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    void SetUp() {
        rot0_.coeffs() << 0, 0, 0, 1;
        rot1_.coeffs() << 0, 0, 0, 1;
        pos0_ << 0, 0, 0;
        pos1_ << 1.5, 0, 0;

        accel_bias0_ << 0, 0, 0;
        accel_bias1_ << 0, 0, 0;
        gyro_bias0_ << 0, 0, 0;
        gyro_bias1_ << 0, 0, 0;

        velocity0_ << 1, 0, 0;
        velocity1_ << 2, 0, 0;


        gravity_magnitude_ = 9.8;

        imu_timestamps_ns_.push_back(0);
        imu_timestamps_ns_.push_back(0.5);
        imu_timestamps_ns_.push_back(1.0);
        Eigen::Matrix<double,6,1> imu;
        imu << 1, 0, gravity_magnitude_, 0, 0, 0;
        imu_data_.push_back(imu);
        imu_data_.push_back(imu);
        imu_data_.push_back(imu);



    }

    void addResidual();
    void solve();
    void checkGradient();

    ceres::Problem problem_;
    ceres::Solver::Summary summary_;

    std::vector<double> imu_timestamps_ns_;
    std::vector<Eigen::Matrix<double, 6, 1>> imu_data_;

    Eigen::Quaterniond rot0_;
    Eigen::Quaterniond rot1_;
    Eigen::Vector3d pos0_;
    Eigen::Vector3d pos1_;

    Eigen::Vector3d accel_bias0_;
    Eigen::Vector3d accel_bias1_;
    Eigen::Vector3d gyro_bias0_;
    Eigen::Vector3d gyro_bias1_;

    Eigen::Vector3d velocity0_;
    Eigen::Vector3d velocity1_;

    Eigen::Matrix<double, 6, 1> imu_bias0_;
    Eigen::Matrix<double, 6, 1> imu_bias1_;

    double gravity_magnitude_;
};


void PosegraphErrorTermsEigen::addResidual() {
//    rot0_.normalize();
//    rot1_.normalize();
//    imu_bias0_ << gyro_bias0_, accel_bias0_;
//    imu_bias1_ << gyro_bias1_, accel_bias1_;
//
//    ceres::CostFunction* inertial_term_cost =
//            new ceres_error_terms::InertialErrorTermEigen(
//                    imu_data_, imu_timestamps_ns_, 1, 1, 1, 1, gravity_magnitude_);
//
//    problem_.AddResidualBlock(
//            inertial_term_cost, NULL, rot0_.coeffs().data(), pos0_.data(),
//            velocity0_.data(), imu_bias0_.data(), rot1_.coeffs().data(), pos1_.data(),
//            velocity1_.data(), imu_bias1_.data());
//
//    ceres::LocalParameterization* quaternion_parameterization =
//            new ceres_error_terms::EigenQuaternionParameterization;
//    problem_.SetParameterization(
//            rot0_.coeffs().data(), quaternion_parameterization);
//    problem_.SetParameterization(
//            rot1_.coeffs().data(), quaternion_parameterization);
}

void PosegraphErrorTermsEigen::checkGradient() {
//    rot0_.normalize();
//    rot1_.normalize();
//    imu_bias0_ << gyro_bias0_, accel_bias0_;
//    imu_bias1_ << gyro_bias1_, accel_bias1_;
//
//    ceres::CostFunction* inertial_term_cost =
//            new ceres_error_terms::InertialErrorTermEigen(
//                    imu_data_, imu_timestamps_ns_, 1, 1, 1, 1, gravity_magnitude_);
//
//    std::vector<double*> parameter_blocks;
//    parameter_blocks.push_back(rot0_.coeffs().data());
//    parameter_blocks.push_back(pos0_.data());
//    parameter_blocks.push_back(velocity0_.data());
//    parameter_blocks.push_back(imu_bias0_.data());
//    parameter_blocks.push_back(rot1_.coeffs().data());
//    parameter_blocks.push_back(pos1_.data());
//    parameter_blocks.push_back(velocity1_.data());
//    parameter_blocks.push_back(imu_bias1_.data());
//
//    ceres::LocalParameterization* orientation_parameterization =
//            new ceres_error_terms::EigenQuaternionParameterization;
//
//    ceres::NumericDiffOptions numeric_diff_options;
//    numeric_diff_options.ridders_relative_initial_step_size = 1e-3;
//
//    std::vector<const ceres::LocalParameterization*> local_parameterizations;
//    local_parameterizations.push_back(orientation_parameterization);
//    local_parameterizations.push_back(NULL);
//    local_parameterizations.push_back(NULL);
//    local_parameterizations.push_back(NULL);
//    local_parameterizations.push_back(orientation_parameterization);
//    local_parameterizations.push_back(NULL);
//    local_parameterizations.push_back(NULL);
//    local_parameterizations.push_back(NULL);
//
////    ceres::GradientChecker gradient_checker(
////            inertial_term_cost, &local_parameterizations, numeric_diff_options);
////    ceres::GradientChecker::ProbeResults results;
////
////    if (!gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results)) {
////        std::cout << "An error has occurred:\n" << results.error_log;
////    }
}

void PosegraphErrorTermsEigen::solve() {
//    ceres::Solver::Options options;
//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = false;
//    options.max_num_iterations = 500;
//    options.gradient_tolerance = 1e-50;
//    options.function_tolerance = 1e-50;
//    options.parameter_tolerance = 1e-50;
//    options.num_threads = 8;
//    options.num_linear_solver_threads = 8;
//
//    ceres::Solve(options, &problem_, &summary_);
//
//    LOG(INFO) << summary_.message;
//    LOG(INFO) << summary_.BriefReport();
}


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

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    PosegraphErrorTermsEigen posegraphErrorTermsEigen;
    posegraphErrorTermsEigen.SetUp();

    ImuParam imuParam;
    std::shared_ptr<IntegrationBase> integrationBase
            = std::make_shared<IntegrationBase>( IntegrationBase(
                    posegraphErrorTermsEigen.imu_data_.front().head(3),
                    posegraphErrorTermsEigen.imu_data_.front().tail(3),
                    ba, bg,
                    imuParam));

    integrationBase->push_back(0.5, posegraphErrorTermsEigen.imu_data_.at(1).head(3),
                               posegraphErrorTermsEigen.imu_data_.at(1).tail(3));

    integrationBase->push_back(0.5, posegraphErrorTermsEigen.imu_data_.at(2).head(3),
                               posegraphErrorTermsEigen.imu_data_.at(2).tail(3));

    Eigen::Vector3d delta_p = integrationBase->delta_p;
    std::cout << "delta_p: " << delta_p.transpose() << std::endl;


    IMUFactor imuFactor(integrationBase.get());

    Eigen::Matrix<double,7,1> T0, T1;
    T0 << posegraphErrorTermsEigen.pos0_, posegraphErrorTermsEigen.rot0_.coeffs();
    T1 << posegraphErrorTermsEigen.pos1_, posegraphErrorTermsEigen.rot1_.coeffs();

    Eigen::Matrix<double,9,1> sb0, sb1;
    sb0 << posegraphErrorTermsEigen.velocity0_, posegraphErrorTermsEigen.accel_bias0_, posegraphErrorTermsEigen.gyro_bias0_;
    sb1 << posegraphErrorTermsEigen.velocity1_, posegraphErrorTermsEigen.accel_bias1_, posegraphErrorTermsEigen.gyro_bias1_;


    Eigen::Matrix<double,15,1> residuals;
    const double* parameters[4] =  {T0.data(), sb0.data(), T1.data(), sb1.data()};
    imuFactor.Evaluate(parameters, residuals.data(),NULL);
    std::cout << residuals.transpose() << std::endl;

    Eigen::Matrix<double,7,1> noised_T0, noised_T1;
        noised_T0 = T0;
        noised_T1 = T1;
        noised_T0.head<3>() += Eigen::Vector3d(-0.2, 0.01, 0.1);
        noised_T1.head<3>() -= Eigen::Vector3d(-0.2, 0.01, 0.1);
        double* noised_parameters[4] =  {noised_T0.data(), sb0.data(), noised_T1.data(), sb1.data()};

        Eigen::Matrix<double,15,7,Eigen::RowMajor> jacobian0;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian1;
        Eigen::Matrix<double,15,7,Eigen::RowMajor> jacobian2;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian3;
        double* jacobians[4] = {jacobian0.data(),jacobian1.data(), jacobian2.data(),jacobian3.data()};
        Eigen::Matrix<double,15,6,Eigen::RowMajor> jacobian0_min;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian1_min;
        Eigen::Matrix<double,15,6,Eigen::RowMajor> jacobian2_min;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian3_min;

        double* jacobians_min[4] = {jacobian0_min.data(),jacobian1_min.data(),jacobian2_min.data(),jacobian3_min.data()};

        Eigen::Matrix<double,15,6,Eigen::RowMajor> numJ0_minimal;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> numJ1_minimal;
        Eigen::Matrix<double,15,6,Eigen::RowMajor> numJ2_minimal;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> numJ3_minimal;

        imuFactor.EvaluateWithMinimalJacobians(noised_parameters,residuals.data(),jacobians,jacobians_min);

        NumbDifferentiator<IMUFactor,4>*  numDiffer =
                new NumbDifferentiator<IMUFactor,4>(&imuFactor);

        numDiffer->df_r_xi<15,7,6,hamilton::PoseLocalParameterization>(noised_parameters,0,numJ0_minimal.data());

        std::cout<<"J0_minimal: "<<std::endl<<jacobian0_min<<std::endl;
        std::cout<<"numJ0_minimal: "<<std::endl<<numJ0_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian0_min - numJ0_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal. Error:"
                                                                              <<std::endl<<(jacobian0_min - numJ0_minimal);

        numDiffer->df_r_xi<15,9>(noised_parameters,1,numJ1_minimal.data());

        std::cout<<"J1_minimal: "<<std::endl<<jacobian1_min<<std::endl;
        std::cout<<"numJ1_minimal: "<<std::endl<<numJ1_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian1_min - numJ1_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal."
                                                                                <<std::endl<<jacobian1_min
                                                                                << "\n numJ1_minimal: "
                                                                                <<std::endl<< numJ1_minimal;


        numDiffer->df_r_xi<15,7,6,hamilton::PoseLocalParameterization>(noised_parameters,2,numJ2_minimal.data());

        std::cout<<"J2_minimal: "<<std::endl<<jacobian0_min<<std::endl;
        std::cout<<"numJ2_minimal: "<<std::endl<<numJ0_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian2_min - numJ2_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal. Error:"
                                                                              <<std::endl<<(jacobian2_min - numJ2_minimal);

        numDiffer->df_r_xi<15,9>(noised_parameters,3,numJ3_minimal.data());

        std::cout<<"J3_minimal: "<<std::endl<<jacobian3_min<<std::endl;
        std::cout<<"numJ3_minimal: "<<std::endl<<numJ3_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian3_min - numJ3_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal."
                    <<std::endl<<(jacobian3_min - numJ3_minimal);



    std::string pose_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
    std::string imu_meas_file =
            "/home/pang/data/dataset/euroc/MH_01_easy/mav0/imu0/data.csv";

    TestSample testSample;
    testSample.readStates(pose_file);
    testSample.readImu(imu_meas_file);

    int start  = testSample.states_vec_.size()* 5/10;
    int end = testSample.states_vec_.size()* 6/10;

    PoseSpline poseSpline(0.1);

    std::vector<std::pair<double,Pose<double>>> samples, queryMeas;


    //
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
            Eigen::Matrix3d R = query_pose.Transformation().topLeftCorner(3,3);
            stampedPose.q_ = Eigen::Quaterniond(R);
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
    int imu_integrated_cnt = 5;
    for (int i = imu_integrated_cnt ; i < simulated_imus.size(); i+=imu_integrated_cnt) {
        Eigen::Quaterniond q_WI0, q_WI1;
        Eigen::Vector3d t_WI0, t_WI1;
        Eigen::Vector3d v0, v1;
        Eigen::Vector3d ba0, ba1;
        Eigen::Vector3d bg0, bg1;
        q_WI0 = simulated_states.at(i - imu_integrated_cnt).q_;
        t_WI0 = simulated_states.at(i - imu_integrated_cnt).t_;
        v0 = simulated_states.at(i - imu_integrated_cnt).v_;
        ba0 << 0,0,0; bg0 << 0,0,0;

        q_WI1 = simulated_states.at(i).q_;
        t_WI1 = simulated_states.at(i).t_;
        v1 = simulated_states.at(i).v_;
        ba1 << 0,0,0; bg1 << 0,0,0;

        uint64_t t0 = simulated_imus.at(i - imu_integrated_cnt).timestamp_;

        Eigen::Vector3d gyro = simulated_imus.at(i - imu_integrated_cnt).gyro_;
        Eigen::Vector3d accel = simulated_imus.at(i - imu_integrated_cnt).accel_;

        std::shared_ptr<IntegrationBase> intergrateImu = std::make_shared<IntegrationBase>(
                accel, gyro, ba0, bg0, imuParam);

        for (int j = i - imu_integrated_cnt + 1; j < i; j++) {
            uint64_t tk = simulated_imus.at(j).timestamp_;
            Eigen::Vector3d gyro = simulated_imus.at(j).gyro_;
            Eigen::Vector3d accel = simulated_imus.at(j).accel_;

            auto dt = (double)(tk - t0)/ 1e9;
            t0 = tk;
            intergrateImu->push_back(dt, accel,gyro);
        }

        IMUFactor imuFactor(intergrateImu.get());

        Eigen::Matrix<double,7,1> T0, T1;
        T0 << t_WI0, q_WI0.coeffs();
        T1 << t_WI1, q_WI1.coeffs();
//        std::cout << "T0: " << T0.transpose() << std::endl;
//        std::cout << "T1: " << T1.transpose() << std::endl;

        Eigen::Matrix<double,9,1> sb0, sb1;
        sb0 << v0, ba0, bg0;
        sb1 << v1, ba1, bg1;

//        std::cout << "sb0: " << sb0.transpose() << std::endl;
//        std::cout << "sb1: " << sb1.transpose() << std::endl;



        Eigen::Matrix<double,15,1> residuals;
        const double* parameters[4] =  {T0.data(), sb0.data(), T1.data(), sb1.data()};
        imuFactor.Evaluate(parameters, residuals.data(),NULL);

        CHECK_EQ((residuals).squaredNorm() < 1e-2, true) << "residuals is large"
                                                                              <<std::endl<<residuals.transpose();

        Eigen::Matrix<double,7,1> noised_T0, noised_T1;
        noised_T0 = T0;
        noised_T1 = T1;
        noised_T0.head<3>() += Eigen::Vector3d(-0.2, 0.01, 0.1);
        noised_T1.head<3>() -= Eigen::Vector3d(-0.2, 0.01, 0.1);
        double* noised_parameters[4] =  {noised_T0.data(), sb0.data(), noised_T1.data(), sb1.data()};

        Eigen::Matrix<double,15,7,Eigen::RowMajor> jacobian0;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian1;
        Eigen::Matrix<double,15,7,Eigen::RowMajor> jacobian2;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian3;
        double* jacobians[4] = {jacobian0.data(),jacobian1.data(), jacobian2.data(),jacobian3.data()};
        Eigen::Matrix<double,15,6,Eigen::RowMajor> jacobian0_min;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian1_min;
        Eigen::Matrix<double,15,6,Eigen::RowMajor> jacobian2_min;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> jacobian3_min;

        double* jacobians_min[4] = {jacobian0_min.data(),jacobian1_min.data(),jacobian2_min.data(),jacobian3_min.data()};

        Eigen::Matrix<double,15,6,Eigen::RowMajor> numJ0_minimal;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> numJ1_minimal;
        Eigen::Matrix<double,15,6,Eigen::RowMajor> numJ2_minimal;
        Eigen::Matrix<double,15,9,Eigen::RowMajor> numJ3_minimal;

        imuFactor.EvaluateWithMinimalJacobians(noised_parameters,residuals.data(),jacobians,jacobians_min);

        NumbDifferentiator<IMUFactor,4>*  numDiffer =
                new NumbDifferentiator<IMUFactor,4>(&imuFactor);

        numDiffer->df_r_xi<15,7,6,hamilton::PoseLocalParameterization>(noised_parameters,0,numJ0_minimal.data());

//        std::cout<<"J0_minimal: "<<std::endl<<jacobian0_min<<std::endl;
//        std::cout<<"numJ0_minimal: "<<std::endl<<numJ0_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian0_min - numJ0_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal. Error:"
                                                                              <<std::endl<<(jacobian0_min - numJ0_minimal);
//
        numDiffer->df_r_xi<15,9>(noised_parameters,1,numJ1_minimal.data());
//
////        std::cout<<"J1_minimal: "<<std::endl<<jacobian1_min<<std::endl;
////        std::cout<<"numJ1_minimal: "<<std::endl<<numJ1_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian1_min - numJ1_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal."
                                                                                <<std::endl<<jacobian1_min
                                                                                << "\n numJ1_minimal: "
                                                                                <<std::endl<< numJ1_minimal;


        numDiffer->df_r_xi<15,7,6,hamilton::PoseLocalParameterization>(noised_parameters,2,numJ2_minimal.data());

//        std::cout<<"J2_minimal: "<<std::endl<<jacobian0_min<<std::endl;
//        std::cout<<"numJ2_minimal: "<<std::endl<<numJ0_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian2_min - numJ2_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal. Error:"
                                                                              <<std::endl<<(jacobian2_min - numJ2_minimal);

        numDiffer->df_r_xi<15,9>(noised_parameters,3,numJ3_minimal.data());

//        std::cout<<"J3_minimal: "<<std::endl<<jacobian3_min<<std::endl;
//        std::cout<<"numJ3_minimal: "<<std::endl<<numJ3_minimal<<std::endl<<std::endl;
        CHECK_EQ((jacobian3_min - numJ3_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal."
                    <<std::endl<<(jacobian3_min - numJ3_minimal);


    }


    return 0;
}