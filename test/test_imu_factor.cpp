
//#include "vins_estimator/utility/NumbDifferentiator.hpp"
#include "extern/spline_imu_error.h"
#include "internal/pose_local_parameterization.h"
#include "PoseSpline/NumbDifferentiator.hpp"

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

//    Eigen::Vector3d delta_p = integrationBase->delta_p;
//    std::cout << "delta_p: " << delta_p.transpose() << std::endl;


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

//
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
    CHECK_EQ((jacobian1_min - numJ1_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal.";


    numDiffer->df_r_xi<15,7,6,hamilton::PoseLocalParameterization>(noised_parameters,2,numJ2_minimal.data());

    std::cout<<"J2_minimal: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"numJ2_minimal: "<<std::endl<<numJ0_minimal<<std::endl<<std::endl;
    CHECK_EQ((jacobian2_min - numJ2_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal. Error:"
                                                                          <<std::endl<<(jacobian2_min - numJ2_minimal);

    numDiffer->df_r_xi<15,9>(noised_parameters,3,numJ3_minimal.data());

    std::cout<<"J3_minimal: "<<std::endl<<jacobian3_min<<std::endl;
    std::cout<<"numJ3_minimal: "<<std::endl<<numJ3_minimal<<std::endl<<std::endl;
    CHECK_EQ((jacobian3_min - numJ3_minimal).squaredNorm() < 0.001, true) << "Analytic and numDiff NOT equal.";







    return 0;
}