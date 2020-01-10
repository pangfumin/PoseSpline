#include "extern/spline_projection_error.h"
#include "extern/spline_projection_error1.h"
#include "extern/spline_projection_error2.h"
#include "extern/spline_projection_error3.h"
#include "extern/spline_projection_error4.h"
#include "extern/spline_projection_error_simple.h"

#include <iostream>
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

TEST(Ceres, SplineProjectCeres) {
    Pose<double> T0, T1, T2, T3;
    T0.setRandom();
    T1.setRandom();
    T2.setRandom();
    T3.setRandom();

//    std::cout << T0.coeffs().transpose() << std::endl;
//    std::cout << T1.coeffs().transpose() << std::endl;
//    std::cout << T2.coeffs().transpose() << std::endl;
//    std::cout << T3.coeffs().transpose() << std::endl;

    double u0 = 0.5;
    double u1 = 0.75;
    double dt = 0.03;
    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0 + dt, T0, T1, T2, T3);
    Pose<double> T_WI1 = PSUtility::EvaluatePS(u1 + dt, T0, T1, T2, T3);

    Pose<double> T_IC;
    T_IC.setRandom();

    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
    C0p(2) = std::fabs(C0p(2));

    Pose<double> T_WC0 = T_WI0 * T_IC;
    Pose<double> T_WC1 = T_WI1 * T_IC;



    Eigen::Vector3d Wp = T_WC0 * C0p;
    Eigen::Vector3d C1p = T_WC1.inverse() * Wp;

    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
    Eigen::Vector3d uv1(C1p(0)/ C1p(2), C1p(1)/C1p(2), 1);

    double rho = 1.0 / C0p(2);

    /*
    * Zero Test
    * Passed!
    */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    Eigen::Isometry3d ext_T_IC;
    ext_T_IC.matrix() = T_IC.Transformation();
    SplineProjectFunctor splineProjectFunctor(u0, uv0, u1, uv1, ext_T_IC);
    SplineProjectError* splineProjectError = new SplineProjectError(splineProjectFunctor);

    double* param_rho;
    param_rho = &rho;
    double param_dt = dt;


    double* paramters[6] = {T0.parameterPtr(), T1.parameterPtr(),
                            T2.parameterPtr(), T3.parameterPtr(),
                            param_rho,
                            &param_dt};


    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
    Eigen::Matrix<double,2,1> jacobian4_min;
    Eigen::Matrix<double,2,1> jacobian5_min;
    double* jacobians_min[6] = {jacobian0_min.data(), jacobian1_min.data(),
                                jacobian2_min.data(), jacobian3_min.data(),
                                jacobian4_min.data(), jacobian5_min.data()};


    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
    Eigen::Matrix<double,2,1> jacobian4;
    Eigen::Matrix<double,2,1> jacobian5;
    double* jacobians[6] = {jacobian0.data(), jacobian1.data(),
                            jacobian2.data(), jacobian3.data(),
                            jacobian4.data(), jacobian5.data()};

    splineProjectError->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";

    /*
    * Jacobian Check: compare the analytical jacobian to num-diff jacobian
    */

    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;
    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised;
    Pose<double> noise;
    noise.setRandom(0.3, 0.03);
    T0_noised = T0*noise;
    noise.setRandom(0.3, 0.03);
    T1_noised = T1*noise;
    noise.setRandom(0.3, 0.03);
    T2_noised = T2*noise;
    noise.setRandom(0.3, 0.03);
    T3_noised = T3*noise;

    double rho_noised = rho + 0.1;
    double param_dt_noised = param_dt;


    double* paramters_noised[6] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
                                   T2_noised.parameterPtr(), T3_noised.parameterPtr(),
                                   &rho_noised, &param_dt_noised};


    splineProjectError->EvaluateWithMinimalJacobians(paramters_noised, residual.data(),
                                                       jacobians, jacobians_min);
    std::cout<<"residual: "<< residual.transpose()<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<SplineProjectError,6> numbDifferentiator(splineProjectError);
    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());
//
    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
             std::endl<<jacobian0_min<<std::endl;
    GTEST_ASSERT_EQ((numJacobian_min0 - jacobian0_min).norm()< 1e6, true);

    // check jacobian_minimal1
    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<jacobian1_min<<std::endl;
    GTEST_ASSERT_EQ((numJacobian_min1 - jacobian1_min).norm()< 1e6, true);

// check jacobian_minimal2
    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<jacobian2_min<<std::endl;
GTEST_ASSERT_EQ((numJacobian_min2 - jacobian2_min).norm()< 1e6, true);


// check jacobian_minimal3
    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<jacobian3_min<<std::endl;
GTEST_ASSERT_EQ((numJacobian_min3 - jacobian3_min).norm()< 1e6, true);


// check jacobian_minimal4
    Eigen::Matrix<double,2,1> numJacobian_min4;
    numbDifferentiator.df_r_xi<2,1>(paramters_noised,4,numJacobian_min4.data());

    std::cout<<"numJacobian_min4: "<<std::endl<<numJacobian_min4<<std::endl;
    std::cout<<"AnaliJacobian_minimal4: "<<
             std::endl<<jacobian4_min<<std::endl;
GTEST_ASSERT_EQ((numJacobian_min4 - jacobian4_min).norm()< 1e6, true);

// check jacobian_minimal5
    Eigen::Matrix<double,2,1> numJacobian_min5;
    numbDifferentiator.df_r_xi<2,1>(paramters_noised,5,numJacobian_min5.data());

    std::cout<<"numJacobian_min5: "<<std::endl<<numJacobian_min5<<std::endl;
    std::cout<<"AnaliJacobian_minimal5: "<<
             std::endl<<jacobian5_min<<std::endl;
    GTEST_ASSERT_EQ((numJacobian_min5 - jacobian5_min).norm()< 1e6, true);

    /*
    * Optimization test
    */

    ceres::Problem problem;
    ceres::LossFunction* loss_function =  new ceres::HuberLoss(1.0);

    PoseLocalParameter*  local_parameterization = new PoseLocalParameter;
    double est_dt = 0;
    Pose<double> pose0 = T0;
    Pose<double> pose1 = T1;
    Pose<double> pose2 = T2;
    Pose<double> pose3 = T3;

    problem.AddParameterBlock(pose0.parameterPtr(), 7);
    problem.SetParameterization(pose0.parameterPtr(),local_parameterization);
    problem.SetParameterBlockConstant(pose0.parameterPtr());

    problem.AddParameterBlock(pose1.parameterPtr(), 7);
    problem.SetParameterization(pose1.parameterPtr(),local_parameterization);
    problem.SetParameterBlockConstant(pose1.parameterPtr());

    problem.AddParameterBlock(pose2.parameterPtr(), 7);
    problem.SetParameterization(pose2.parameterPtr(),local_parameterization);
    problem.SetParameterBlockConstant(pose2.parameterPtr());

    problem.AddParameterBlock(pose3.parameterPtr(), 7);
    problem.SetParameterization(pose3.parameterPtr(),local_parameterization);
    problem.SetParameterBlockConstant(pose3.parameterPtr());

    problem.AddParameterBlock(&est_dt, 1);




    SplineProjectFunctor functor(u0, uv0, u1, uv1, ext_T_IC);
    SplineProjectError* costfunction = new SplineProjectError(functor);

    problem.AddResidualBlock(costfunction, loss_function,
            pose0.parameterPtr(),
            pose1.parameterPtr(),
             pose2.parameterPtr(),
             pose3.parameterPtr(),
             param_rho,
             &est_dt);




    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout<<"--------------"<<std::endl;
    std::cout << summary.FullReport() << std::endl;

    std::cout << "est_dt: " << est_dt << std::endl;
    std::cout << "true_dt: " << dt << std::endl;


}

