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

//    std::cout << "T_WI0: " << T_WI0.parameters().transpose() << std::endl;
//    std::cout << "T_WI1: " << T_WI1.parameters().transpose() << std::endl;

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
//
//TEST(Ceres, SplineProjectError1) {
//    Pose<double> T0, T1, T2, T3, T4;
//    T0.setRandom();
//    T1.setRandom();
//    T2.setRandom();
//    T3.setRandom();
//    T4.setRandom();
//
//    std::cout << T0.coeffs().transpose() << std::endl;
//    std::cout << T1.coeffs().transpose() << std::endl;
//    std::cout << T2.coeffs().transpose() << std::endl;
//    std::cout << T3.coeffs().transpose() << std::endl;
//    std::cout << T4.coeffs().transpose() << std::endl;
//
//    double u0 = 0.5;
//    double u1 = 0.75;
//    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0, T0, T1, T2, T3);
//    Pose<double> T_WI1 = PSUtility::EvaluatePS(u1, T1, T2, T3, T4);
//
//    Pose<double> T_IC;
//    T_IC.setRandom();
//
//    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
//    C0p(2) = std::fabs(C0p(2));
//
//    Pose<double> T_WC0 = T_WI0 * T_IC;
//    Pose<double> T_WC1 = T_WI1 * T_IC;
//
//
//
//    Eigen::Vector3d Wp = T_WC0 * C0p;
//    Eigen::Vector3d C1p = T_WC1.inverse() * Wp;
//
//    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
//    Eigen::Vector3d uv1(C1p(0)/ C1p(2), C1p(1)/C1p(2), 1);
//
//    double rho = 1.0 / C0p(2);
//
//    /*
//    * Zero Test
//    * Passed!
//    */
//
//    std::cout<<"------------ Zero Test -----------------"<<std::endl;
//
//    Eigen::Isometry3d ext_T_IC;
//    ext_T_IC.matrix() = T_IC.Transformation();
//    SplineProjectFunctor1 splineProjectFunctor1(u0, uv0, u1, uv1, ext_T_IC);
//    SplineProjectError1* splineProjectError1 = new SplineProjectError1(splineProjectFunctor1);
//
//    double* param_rho;
//    param_rho = &rho;
//
//    double* paramters[6] = {T0.parameterPtr(), T1.parameterPtr(),
//                            T2.parameterPtr(), T3.parameterPtr(),
//                            T4.parameterPtr(), param_rho};
//
//    Eigen::Matrix<double, 2,1> residual;
//
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian4_min;
//    Eigen::Matrix<double,2,1> jacobian5_min;
//    double* jacobians_min[6] = {jacobian0_min.data(), jacobian1_min.data(),
//                                jacobian2_min.data(), jacobian3_min.data(),
//                                jacobian4_min.data(), jacobian5_min.data()};
//
//
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian4;
//    Eigen::Matrix<double,2,1> jacobian5;
//    double* jacobians[6] = {jacobian0.data(), jacobian1.data(),
//                            jacobian2.data(), jacobian3.data(),
//                            jacobian4.data(), jacobian5.data()};
//
//    splineProjectError1->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,NULL);
//
//    std::cout<<"residual: "<<residual.transpose()<<std::endl;
//    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
//    /*
//    * Jacobian Check: compare the analytical jacobian to num-diff jacobian
//    */
//
//    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;
//    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised, T4_noised;
//    Pose<double> noise;
//    noise.setRandom(0.3, 0.03);
//    T0_noised = T0*noise;
//    noise.setRandom(0.3, 0.03);
//    T1_noised = T1*noise;
//    noise.setRandom(0.3, 0.03);
//    T2_noised = T2*noise;
//    noise.setRandom(0.3, 0.03);
//    T3_noised = T3*noise;
//    noise.setRandom(0.3, 0.03);
//    T4_noised = T4*noise;
//
//    double rho_noised = rho + 0.1;
//
//
//    double* paramters_noised[6] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
//                                   T2_noised.parameterPtr(), T3_noised.parameterPtr(),
//                                   T4_noised.parameterPtr(),
//                                   &rho_noised};
//
//
//    splineProjectError1->EvaluateWithMinimalJacobians(paramters_noised, residual.data(),
//                                                      jacobians, jacobians_min);
//    std::cout<<"residual: "<< residual.transpose()<<std::endl;
//
//
//    // check jacobian_minimal0
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min0;
//    NumbDifferentiator<SplineProjectError1,6> numbDifferentiator(splineProjectError1);
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());
//
//    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
//    std::cout<<"AnaliJacobian_minimal0: "<<
//             std::endl<<jacobian0_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min0 - jacobian0_min).norm()< 1e6, true);
//
//    // check jacobian_minimal1
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min1;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());
//
//    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
//    std::cout<<"AnaliJacobian_minimal1: "<<
//             std::endl<<jacobian1_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min1 - jacobian1_min).norm()< 1e6, true);
//
//// check jacobian_minimal2
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min2;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());
//
//    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
//    std::cout<<"AnaliJacobian_minimal2: "<<
//             std::endl<<jacobian2_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min2 - jacobian2_min).norm()< 1e6, true);
//
//
//    // check jacobian_minimal3
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min3;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());
//
//    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
//    std::cout<<"AnaliJacobian_minimal3: "<<
//             std::endl<<jacobian3_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min3 - jacobian3_min).norm()< 1e6, true);
//
//
//    // check jacobian_minimal4
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min4;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,4,numJacobian_min4.data());
//
//    std::cout<<"numJacobian_min4: "<<std::endl<<numJacobian_min4<<std::endl;
//    std::cout<<"AnaliJacobian_minimal4: "<<
//             std::endl<<jacobian4_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min4 - jacobian4_min).norm()< 1e6, true);
//
//
//    // check jacobian_minimal5
//    Eigen::Matrix<double,2,1> numJacobian_min5;
//    numbDifferentiator.df_r_xi<2,1>(paramters_noised,5,numJacobian_min5.data());
//
//    std::cout<<"numJacobian_min5: "<<std::endl<<numJacobian_min5<<std::endl;
//    std::cout<<"AnaliJacobian_minimal5: "<<
//             std::endl<<jacobian5_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min5 - jacobian5_min).norm()< 1e6, true);
//}
//
//TEST(Ceres, SplineProjectError2) {
//    Pose<double> T0, T1, T2, T3, T4, T5;
//    T0.setRandom();
//    T1.setRandom();
//    T2.setRandom();
//    T3.setRandom();
//    T4.setRandom();
//    T5.setRandom();
//
//    std::cout << T0.coeffs().transpose() << std::endl;
//    std::cout << T1.coeffs().transpose() << std::endl;
//    std::cout << T2.coeffs().transpose() << std::endl;
//    std::cout << T3.coeffs().transpose() << std::endl;
//    std::cout << T4.coeffs().transpose() << std::endl;
//
//    double u0 = 0.5;
//    double u1 = 0.75;
//    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0, T0, T1, T2, T3);
//    Pose<double> T_WI1 = PSUtility::EvaluatePS(u1, T2, T3, T4, T5);
//
//    Pose<double> T_IC;
//    T_IC.setRandom();
//
//    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
//    C0p(2) = std::fabs(C0p(2));
//
//    Pose<double> T_WC0 = T_WI0 * T_IC;
//    Pose<double> T_WC1 = T_WI1 * T_IC;
//
//
//
//    Eigen::Vector3d Wp = T_WC0 * C0p;
//    Eigen::Vector3d C1p = T_WC1.inverse() * Wp;
//
//    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
//    Eigen::Vector3d uv1(C1p(0)/ C1p(2), C1p(1)/C1p(2), 1);
//
//    double rho = 1.0 / C0p(2);
//
//    /*
//    * Zero Test
//    * Passed!
//    */
//
//    std::cout<<"------------ Zero Test -----------------"<<std::endl;
//
//    Eigen::Isometry3d ext_T_IC;
//    ext_T_IC.matrix() = T_IC.Transformation();
//    SplineProjectFunctor2 splineProjectFunctor2(u0, uv0, u1, uv1, ext_T_IC);
//    SplineProjectError2* splineProjectError2 = new SplineProjectError2(splineProjectFunctor2);
//
//    double* param_rho;
//    param_rho = &rho;
//
//    double* paramters[7] = {T0.parameterPtr(), T1.parameterPtr(),
//                            T2.parameterPtr(), T3.parameterPtr(),
//                            T4.parameterPtr(), T5.parameterPtr(),
//                            param_rho};
//
//    Eigen::Matrix<double, 2,1> residual;
//
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian4_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian5_min;
//    Eigen::Matrix<double,2,1> jacobian6_min;
//    double* jacobians_min[7] = {jacobian0_min.data(), jacobian1_min.data(),
//                                jacobian2_min.data(), jacobian3_min.data(),
//                                jacobian4_min.data(), jacobian5_min.data(),
//                                jacobian6_min.data()};
//
//
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian4;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian5;
//    Eigen::Matrix<double,2,1> jacobian6;
//    double* jacobians[7] = {jacobian0.data(), jacobian1.data(),
//                            jacobian2.data(), jacobian3.data(),
//                            jacobian4.data(), jacobian5.data(),
//                            jacobian6.data()};
//
//    splineProjectError2->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,NULL);
//
//    std::cout<<"residual: "<<residual.transpose()<<std::endl;
//    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
//    /*
//    * Jacobian Check: compare the analytical jacobian to num-diff jacobian
//    */
//
//    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;
//    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised, T4_noised, T5_noised;
//    Pose<double> noise;
//    noise.setRandom(0.3, 0.03);
//    T0_noised = T0*noise;
//    noise.setRandom(0.3, 0.03);
//    T1_noised = T1*noise;
//    noise.setRandom(0.3, 0.03);
//    T2_noised = T2*noise;
//    noise.setRandom(0.3, 0.03);
//    T3_noised = T3*noise;
//    noise.setRandom(0.3, 0.03);
//    T4_noised = T4*noise;
//    noise.setRandom(0.3, 0.03);
//    T5_noised = T5*noise;
//
//    double rho_noised = rho + 0.1;
//
//
//    double* paramters_noised[7] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
//                                   T2_noised.parameterPtr(), T3_noised.parameterPtr(),
//                                   T4_noised.parameterPtr(), T5_noised.parameterPtr(),
//                                   &rho_noised};
//
//
//    splineProjectError2->EvaluateWithMinimalJacobians(paramters_noised, residual.data(),
//                                                      jacobians, jacobians_min);
//    std::cout<<"residual: "<< residual.transpose()<<std::endl;
//
//    NumbDifferentiator<SplineProjectError2, 7> numbDifferentiator(splineProjectError2);
//    for (int i = 0; i < 6; i++) {
//        std::cout << "Check " << i << " th Jacobian" << std::endl;
//        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> numJacobian_min;
//        numbDifferentiator.df_r_xi<2, 7, 6, PoseLocalParameter>(paramters_noised, i, numJacobian_min.data());
//
//        std::cout << "numJacobian_min: " << std::endl << numJacobian_min << std::endl;
//        std::cout << "AnaliJacobian_minimal: " <<
//                  std::endl << Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>>(jacobians_min[i]) << std::endl;
//        GTEST_ASSERT_EQ((numJacobian_min -
//        Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>>(jacobians_min[i])).norm()< 1e6, true);
//    }
//
//    // check jacobian_minimal6
//    Eigen::Matrix<double,2,1> numJacobian_min6;
//    numbDifferentiator.df_r_xi<2,1>(paramters_noised,6,numJacobian_min6.data());
//    std::cout<<"numJacobian_min6: "<<std::endl<<numJacobian_min6<<std::endl;
//    std::cout<<"AnaliJacobian_minimal6: "<<
//             std::endl<<jacobian6_min<<std::endl;
//
//    GTEST_ASSERT_EQ((numJacobian_min6 - jacobian6_min).norm()< 1e6, true);
//
//}
//
//TEST(Ceres, SplineProjectError3) {
//    Pose<double> T0, T1, T2, T3, T4, T5;
//    Pose<double> T6;
//    T0.setRandom();
//    T1.setRandom();
//    T2.setRandom();
//    T3.setRandom();
//    T4.setRandom();
//    T5.setRandom();
//    T6.setRandom();
//
//    std::cout << T0.coeffs().transpose() << std::endl;
//    std::cout << T1.coeffs().transpose() << std::endl;
//    std::cout << T2.coeffs().transpose() << std::endl;
//    std::cout << T3.coeffs().transpose() << std::endl;
//    std::cout << T4.coeffs().transpose() << std::endl;
//
//    double u0 = 0.5;
//    double u1 = 0.75;
//    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0, T0, T1, T2, T3);
//    Pose<double> T_WI1 = PSUtility::EvaluatePS(u1, T3, T4, T5, T6);
//
//    Pose<double> T_IC;
//    T_IC.setRandom();
//
//    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
//    C0p(2) = std::fabs(C0p(2));
//
//    Pose<double> T_WC0 = T_WI0 * T_IC;
//    Pose<double> T_WC1 = T_WI1 * T_IC;
//
//
//
//    Eigen::Vector3d Wp = T_WC0 * C0p;
//    Eigen::Vector3d C1p = T_WC1.inverse() * Wp;
//
//    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
//    Eigen::Vector3d uv1(C1p(0)/ C1p(2), C1p(1)/C1p(2), 1);
//
//    double rho = 1.0 / C0p(2);
//
//    /*
//    * Zero Test
//    * Passed!
//    */
//
//    std::cout<<"------------ Zero Test -----------------"<<std::endl;
//
//    Eigen::Isometry3d ext_T_IC;
//    ext_T_IC.matrix() = T_IC.Transformation();
//    SplineProjectFunctor3 splineProjectFunctor3(u0, uv0, u1, uv1, ext_T_IC);
//    SplineProjectError3* splineProjectError3 = new SplineProjectError3(splineProjectFunctor3);
//
//    double* param_rho;
//    param_rho = &rho;
//
//    double* paramters[8] = {T0.parameterPtr(), T1.parameterPtr(),
//                            T2.parameterPtr(), T3.parameterPtr(),
//                            T4.parameterPtr(), T5.parameterPtr(),
//                            T6.parameterPtr(),
//                            param_rho};
//
//    Eigen::Matrix<double, 2,1> residual;
//
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian4_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian5_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian6_min;
//    Eigen::Matrix<double,2,1> jacobian7_min;
//    double* jacobians_min[8] = {jacobian0_min.data(), jacobian1_min.data(),
//                                jacobian2_min.data(), jacobian3_min.data(),
//                                jacobian4_min.data(), jacobian5_min.data(),
//                                jacobian6_min.data(), jacobian7_min.data()};
//
//
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian4;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian5;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian6;
//    Eigen::Matrix<double,2,1> jacobian7;
//    double* jacobians[8] = {jacobian0.data(), jacobian1.data(),
//                            jacobian2.data(), jacobian3.data(),
//                            jacobian4.data(), jacobian5.data(),
//                            jacobian6.data(), jacobian7.data()};
//
//    splineProjectError3->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,NULL);
//
//    std::cout<<"residual: "<<residual.transpose()<<std::endl;
//    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
//    /*
//    * Jacobian Check: compare the analytical jacobian to num-diff jacobian
//    */
//
//    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;
//    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised, T4_noised, T5_noised;
//    Pose<double> T6_noised;
//    Pose<double> noise;
//    noise.setRandom(0.3, 0.03);
//    T0_noised = T0*noise;
//    noise.setRandom(0.3, 0.03);
//    T1_noised = T1*noise;
//    noise.setRandom(0.3, 0.03);
//    T2_noised = T2*noise;
//    noise.setRandom(0.3, 0.03);
//    T3_noised = T3*noise;
//    noise.setRandom(0.3, 0.03);
//    T4_noised = T4*noise;
//    noise.setRandom(0.3, 0.03);
//    T5_noised = T5*noise;
//    noise.setRandom(0.3, 0.03);
//    T6_noised = T6*noise;
//
//    double rho_noised = rho + 0.1;
//
//
//    double* paramters_noised[8] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
//                                   T2_noised.parameterPtr(), T3_noised.parameterPtr(),
//                                   T4_noised.parameterPtr(), T5_noised.parameterPtr(),
//                                   T6_noised.parameterPtr(),
//                                   &rho_noised};
//
//
//    splineProjectError3->EvaluateWithMinimalJacobians(paramters_noised, residual.data(),
//                                                      jacobians, jacobians_min);
//    std::cout<<"residual: "<< residual.transpose()<<std::endl;
//
//    NumbDifferentiator<SplineProjectError3, 8> numbDifferentiator(splineProjectError3);
//    for (int i = 0; i < 7; i++) {
//        std::cout << "Check " << i << " th Jacobian" << std::endl;
//        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> numJacobian_min;
//        numbDifferentiator.df_r_xi<2, 7, 6, PoseLocalParameter>(paramters_noised, i, numJacobian_min.data());
//
//        std::cout << "numJacobian_min: " << std::endl << numJacobian_min << std::endl;
//        std::cout << "AnaliJacobian_minimal: " <<
//                  std::endl << Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>>(jacobians_min[i]) << std::endl;
//        GTEST_ASSERT_EQ((numJacobian_min -
//                         Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>>(jacobians_min[i])).norm()< 1e6, true);
//    }
//
//    // check jacobian_minimal7
//    Eigen::Matrix<double,2,1> numJacobian_min7;
//    numbDifferentiator.df_r_xi<2,1>(paramters_noised,7,numJacobian_min7.data());
//    std::cout<<"numJacobian_min7: "<<std::endl<<numJacobian_min7<<std::endl;
//    std::cout<<"AnaliJacobian_minimal7: "<<
//             std::endl<<jacobian7_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min7 -
//            jacobian7_min).norm()< 1e6, true);
//
//
//}
//
//TEST(Ceres, SplineProjectError4) {
//    Pose<double> T0, T1, T2, T3, T4, T5;
//    Pose<double> T6, T7;
//    T0.setRandom();
//    T1.setRandom();
//    T2.setRandom();
//    T3.setRandom();
//    T4.setRandom();
//    T5.setRandom();
//    T6.setRandom();
//    T7.setRandom();
//
//    std::cout << T0.coeffs().transpose() << std::endl;
//    std::cout << T1.coeffs().transpose() << std::endl;
//    std::cout << T2.coeffs().transpose() << std::endl;
//    std::cout << T3.coeffs().transpose() << std::endl;
//    std::cout << T4.coeffs().transpose() << std::endl;
//
//    double u0 = 0.5;
//    double u1 = 0.75;
//    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0, T0, T1, T2, T3);
//    Pose<double> T_WI1 = PSUtility::EvaluatePS(u1, T4, T5, T6, T7);
//
//    Pose<double> T_IC;
//    T_IC.setRandom();
//
//    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
//    C0p(2) = std::fabs(C0p(2));
//
//    Pose<double> T_WC0 = T_WI0 * T_IC;
//    Pose<double> T_WC1 = T_WI1 * T_IC;
//
//
//
//    Eigen::Vector3d Wp = T_WC0 * C0p;
//    Eigen::Vector3d C1p = T_WC1.inverse() * Wp;
//
//    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
//    Eigen::Vector3d uv1(C1p(0)/ C1p(2), C1p(1)/C1p(2), 1);
//
//    double rho = 1.0 / C0p(2);
//
//    /*
//    * Zero Test
//    * Passed!
//    */
//
//    std::cout<<"------------ Zero Test -----------------"<<std::endl;
//
//    Eigen::Isometry3d ext_T_IC;
//    ext_T_IC.matrix() = T_IC.Transformation();
//    SplineProjectFunctor4 splineProjectFunctor4(u0, uv0, u1, uv1, ext_T_IC);
//    SplineProjectError4* splineProjectError4 = new SplineProjectError4(splineProjectFunctor4);
//
//    double* param_rho;
//    param_rho = &rho;
//
//    double* paramters[9] = {T0.parameterPtr(), T1.parameterPtr(),
//                            T2.parameterPtr(), T3.parameterPtr(),
//                            T4.parameterPtr(), T5.parameterPtr(),
//                            T6.parameterPtr(), T7.parameterPtr(),
//                            param_rho};
//
//    Eigen::Matrix<double, 2,1> residual;
//
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian4_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian5_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian6_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian7_min;
//    Eigen::Matrix<double,2,1> jacobian8_min;
//    double* jacobians_min[9] = {jacobian0_min.data(), jacobian1_min.data(),
//                                jacobian2_min.data(), jacobian3_min.data(),
//                                jacobian4_min.data(), jacobian5_min.data(),
//                                jacobian6_min.data(), jacobian7_min.data(),
//                                jacobian8_min.data()};
//
//
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian4;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian5;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian6;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian7;
//    Eigen::Matrix<double,2,1> jacobian8;
//    double* jacobians[9] = {jacobian0.data(), jacobian1.data(),
//                            jacobian2.data(), jacobian3.data(),
//                            jacobian4.data(), jacobian5.data(),
//                            jacobian6.data(), jacobian7.data(),
//                            jacobian8.data()};
//
//    splineProjectError4->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,NULL);
//
//    std::cout<<"residual: "<<residual.transpose()<<std::endl;
//    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
//    /*
//    * Jacobian Check: compare the analytical jacobian to num-diff jacobian
//    */
//
//    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;
//    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised, T4_noised, T5_noised;
//    Pose<double> T6_noised, T7_noised;
//    Pose<double> noise;
//    noise.setRandom(0.3, 0.03);
//    T0_noised = T0*noise;
//    noise.setRandom(0.3, 0.03);
//    T1_noised = T1*noise;
//    noise.setRandom(0.3, 0.03);
//    T2_noised = T2*noise;
//    noise.setRandom(0.3, 0.03);
//    T3_noised = T3*noise;
//    noise.setRandom(0.3, 0.03);
//    T4_noised = T4*noise;
//    noise.setRandom(0.3, 0.03);
//    T5_noised = T5*noise;
//    noise.setRandom(0.3, 0.03);
//    T6_noised = T6*noise;
//    noise.setRandom(0.3, 0.03);
//    T7_noised = T7*noise;
//
//    double rho_noised = rho + 0.1;
//
//
//    double* paramters_noised[9] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
//                                   T2_noised.parameterPtr(), T3_noised.parameterPtr(),
//                                   T4_noised.parameterPtr(), T5_noised.parameterPtr(),
//                                   T6_noised.parameterPtr(), T7_noised.parameterPtr(),
//                                   &rho_noised};
//
//
//    splineProjectError4->EvaluateWithMinimalJacobians(paramters_noised, residual.data(),
//                                                      jacobians, jacobians_min);
//    std::cout<<"residual: "<< residual.transpose()<<std::endl;
//
//    NumbDifferentiator<SplineProjectError4, 9> numbDifferentiator(splineProjectError4);
//    for (int i = 0; i < 8; i++) {
//        std::cout << "Check " << i << " th Jacobian" << std::endl;
//        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> numJacobian_min;
//        numbDifferentiator.df_r_xi<2, 7, 6, PoseLocalParameter>(paramters_noised, i, numJacobian_min.data());
//
//        std::cout << "numJacobian_min: " << std::endl << numJacobian_min << std::endl;
//        std::cout << "AnaliJacobian_minimal: " <<
//                  std::endl << Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>>(jacobians_min[i]) << std::endl;
//        GTEST_ASSERT_EQ((numJacobian_min -
//                         Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>>(jacobians_min[i])).norm()< 1e6, true);
//    }
//
//    // check jacobian_minimal8
//    Eigen::Matrix<double,2,1> numJacobian_min8;
//    numbDifferentiator.df_r_xi<2,1>(paramters_noised,8,numJacobian_min8.data());
//    std::cout<<"numJacobian_min8: "<<std::endl<<numJacobian_min8<<std::endl;
//    std::cout<<"AnaliJacobian_minimal8: "<<
//             std::endl<<jacobian8_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min8 -
//            jacobian8_min).norm()< 1e6, true);
//
//
//}
//
//
//TEST(Ceres, SplineProjectErrorSimple) {
//    Pose<double> T0, T1, T2, T3;
//    T0.setRandom();
//    T1.setRandom();
//    T2.setRandom();
//    T3.setRandom();
//
//    std::cout << T0.coeffs().transpose() << std::endl;
//    std::cout << T1.coeffs().transpose() << std::endl;
//    std::cout << T2.coeffs().transpose() << std::endl;
//    std::cout << T3.coeffs().transpose() << std::endl;
//
//    double u0 = 0.5;
//    double u1 = 0.75;
//    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0, T0, T1, T2, T3);
//
//    Pose<double> T_IC;
//    T_IC.setRandom();
//
//    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
//    C0p(2) = std::fabs(C0p(2));
//
//    Pose<double> T_WC0 = T_WI0 * T_IC;
//
//    Eigen::Vector3d Wp = T_WC0 * C0p;
//
//
//    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
//
//
//    /*
//    * Zero Test
//    * Passed!
//    */
//
//    std::cout<<"------------ Zero Test -----------------"<<std::endl;
//
//    Eigen::Isometry3d ext_T_IC;
//    ext_T_IC.matrix() = T_IC.Transformation();
//    SplineProjectSimpleError* splineProjectError = new SplineProjectSimpleError(u0, uv0, ext_T_IC);
//
//    Eigen::Vector3d param_Wp;
//    param_Wp = Wp;
//
//    double* paramters[5] = {T0.parameterPtr(), T1.parameterPtr(),
//                            T2.parameterPtr(), T3.parameterPtr(), param_Wp.data()};
//
//    Eigen::Matrix<double, 2,1> residual;
//
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
//    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian4_min;
//    double* jacobians_min[5] = {jacobian0_min.data(), jacobian1_min.data(),
//                                jacobian2_min.data(), jacobian3_min.data(),
//                                jacobian4_min.data()};
//
//
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
//    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
//    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian4;
//    double* jacobians[5] = {jacobian0.data(), jacobian1.data(),
//                            jacobian2.data(), jacobian3.data(),
//                            jacobian4.data()};
//
//    splineProjectError->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);
//
//    std::cout<<"residual: "<<residual.transpose()<<std::endl;
//    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
//    /*
//    * Jacobian Check: compare the analytical jacobian to num-diff jacobian
//    */
//
//    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;
//    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised;
//    Pose<double> noise;
//    noise.setRandom(0.3, 0.03);
//    T0_noised = T0*noise;
//    noise.setRandom(0.3, 0.03);
//    T1_noised = T1*noise;
//    noise.setRandom(0.3, 0.03);
//    T2_noised = T2*noise;
//    noise.setRandom(0.3, 0.03);
//    T3_noised = T3*noise;
//
//    param_Wp = Wp;
//
//
//    double* paramters_noised[5] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
//                                   T2_noised.parameterPtr(), T3_noised.parameterPtr(),
//                                   param_Wp.data()};
//
//
//    splineProjectError->EvaluateWithMinimalJacobians(paramters_noised, residual.data(),
//                                                     jacobians, jacobians_min);
//    std::cout<<"residual: "<< residual.transpose()<<std::endl;
//
//
//    // check jacobian_minimal0
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min0;
//    NumbDifferentiator<SplineProjectSimpleError,5> numbDifferentiator(splineProjectError);
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());
//
//    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
//    std::cout<<"AnaliJacobian_minimal0: "<<
//             std::endl<<jacobian0_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min0 - jacobian0_min).norm()< 1e6, true);
//
//    // check jacobian_minimal1
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min1;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());
//
//    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
//    std::cout<<"AnaliJacobian_minimal1: "<<
//             std::endl<<jacobian1_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min1 - jacobian1_min).norm()< 1e6, true);
//
//// check jacobian_minimal2
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min2;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());
//
//    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
//    std::cout<<"AnaliJacobian_minimal2: "<<
//             std::endl<<jacobian2_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min2 - jacobian2_min).norm()< 1e6, true);
//
//
//// check jacobian_minimal3
//    Eigen::Matrix<double,2,6,Eigen::RowMajor> numJacobian_min3;
//    numbDifferentiator.df_r_xi<2,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());
//
//    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
//    std::cout<<"AnaliJacobian_minimal3: "<<
//             std::endl<<jacobian3_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min3 - jacobian3_min).norm()< 1e6, true);
//
//
//// check jacobian_minimal4
//    Eigen::Matrix<double,2,3,Eigen::RowMajor> numJacobian_min4;
//    numbDifferentiator.df_r_xi<2,3>(paramters_noised,4,numJacobian_min4.data());
//
//    std::cout<<"numJacobian_min4: "<<std::endl<<numJacobian_min4<<std::endl;
//    std::cout<<"AnaliJacobian_minimal4: "<<
//             std::endl<<jacobian4_min<<std::endl;
//    GTEST_ASSERT_EQ((numJacobian_min4 - jacobian4_min).norm()< 1e6, true);
//
//}