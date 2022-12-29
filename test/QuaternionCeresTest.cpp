#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionError.hpp"
#include <gtest/gtest.h>
#include <ceres/gradient_checker.h>
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/QuaternionSplineSampleError.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/QuaternionOmegaSampleError.hpp"




bool test(Quaternion Q_target, Quaternion Q_param){
    double* parametrs[1] = {Q_param.data()};
    ceres::LocalParameterization *local_parameterization = new QuaternionLocalParameter();
    QuaternionErrorCostFunction* quaternionError  = new QuaternionErrorCostFunction(Q_target);

    // check jacobians
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numMinimalJacobian0;
    numMinimalJacobian0.setIdentity();
    NumbDifferentiator<QuaternionErrorCostFunction,1> numDiffer(quaternionError);
    numDiffer.df_r_xi<3,4,3,QuaternionLocalParameter>(parametrs,0,numMinimalJacobian0.data());
    quaternionError->VerifyJacobianNumDiff(parametrs);

    // solve
    ceres::Problem problem;
    ceres::LossFunction* loss_function =  new ceres::HuberLoss(1.0);

    problem.AddParameterBlock(Q_param.data(), 4);
    problem.SetParameterization(Q_param.data(),local_parameterization);

    problem.AddResidualBlock(quaternionError, loss_function, Q_param.data());
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    double const mse = quatMult(Q_target,quatInv(Q_param)).squaredNorm();

//    std::cout<<"Q_target: "<<Q_target.transpose()<<std::endl;
//    std::cout<<"Q_param : "<<Q_param.transpose()<<std::endl;

/*
    // use ceres gradient checker
    std::vector<const ceres::LocalParameterization*>* localParameters;
    localParameters->push_back(local_parameterization);

    ceres::NumericDiffOptions option;
    ceres::GradientChecker graChecker(quaternionError,localParameters,option);

    ceres::GradientChecker::ProbeResults* res;
    graChecker.Probe(parametrs,1e-6,res);

*/


    return std::abs(mse - 1) < 0.0001;

}


TEST(Ceres, QuaternionFitting){
   std::vector<Quaternion> quaternionVec;
    Quaternion Q(1,0,3,5);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(1,8,3,5);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(1,8,3,50);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(1,8,3,-50);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);
    Q = Quaternion(1,-108,3,50);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(1000,8,3,50);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(-1000,8,3,50);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(-1000,8,-3,-50);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(-1000,8,-3,0);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    Q = Quaternion(0,0,0,1);
    Q = Q/Q.norm();
    quaternionVec.push_back(Q);

    for(unsigned int i = 0 ; i < quaternionVec.size(); i++){
        bool pass  = test(quaternionVec.at(i), quaternionVec.at(0));
        GTEST_ASSERT_EQ(pass,true);

    }
}

TEST(Ceres, QuaternionSplineCeres) {
    double u = 0.6;
    Quaternion Q_meas(0.0,1.0,0.0,1.9);
    Q_meas = quatNorm(Q_meas);

    double u1 = 0.4;
    Quaternion Q_meas1(0.1,1.0,0.0,1.9);
    Q_meas1 = quatNorm(Q_meas1);

    double u2 = 0.1;
    Quaternion Q_meas2(0.1,1.3,0.0,1.9);
    Q_meas2 = quatNorm(Q_meas2);

    double u3 = 0.9;
    Quaternion Q_meas3(0.1,1.3,-0.7,1.9);
    Q_meas3 = quatNorm(Q_meas3);

    double u4 = 0.3;
    Quaternion Q_meas4(0.1,1.3,-0.732,1.9);
    Q_meas4 = quatNorm(Q_meas4);

    double u5 = 0.5;
    Quaternion Q_meas5(0.12,1.3,-0.7,1.9);
    Q_meas5 = quatNorm(Q_meas5);


    Quaternion Cp0,Cp1,Cp2,Cp3;
    Cp0 = Cp1 = Cp2 = Cp3 = unitQuat<double>();
    Cp0 = Quaternion(-0.0233343  ,0.538966 ,  0.805091,   0.246575); Cp0 = quatNorm(Cp0);
    Cp1 = Quaternion(0.142278 ,  0.44318 ,-0.513372 ,  0.72097);  Cp1 = quatNorm(Cp1);
    Cp2 = Quaternion(-0.112329,  0.379688,   0.34445,  0.851219);  Cp2 = quatNorm(Cp2);
    Cp3 = Quaternion(-0.164781, -0.303314,  0.876392, -0.335836);  Cp3 = quatNorm(Cp3);

    Quaternion Cp0_init,Cp1_init,Cp2_init,Cp3_init;
    Cp0_init = Cp0;
    Cp1_init = Cp1;
    Cp2_init = Cp2;
    Cp3_init = Cp3;

    double* paramters[4] = {Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data()};
    Eigen::Vector3d Residual;


    /*
     *  Test jacobians
     */
    Eigen::Matrix<double,3,3,Eigen::RowMajor> AnaliJacobian_minimal0,AnaliJacobian_minimal1,AnaliJacobian_minimal2,AnaliJacobian_minimal3;
    double* AnaliJacobians_minimal[4] = {AnaliJacobian_minimal0.data(),
                                         AnaliJacobian_minimal1.data(),
                                         AnaliJacobian_minimal2.data(),
                                         AnaliJacobian_minimal3.data()};
    Eigen::Matrix<double,3,4,Eigen::RowMajor> AnaliJacobian0,AnaliJacobian1,AnaliJacobian2,AnaliJacobian3;
    double* AnaliJacobians[4] = {AnaliJacobian0.data(),
                                 AnaliJacobian1.data(),
                                 AnaliJacobian2.data(),
                                 AnaliJacobian3.data()};


    //QuaternionLocalParameter localParameter = new QuaternionLocalParameter;
    QuaternionSplineSampleError * quatSplineError = new QuaternionSplineSampleError(u, Q_meas);
    quatSplineError->EvaluateWithMinimalJacobians(paramters,Residual.data(),AnaliJacobians,AnaliJacobians_minimal);

    std::cout<<"AnaliJacobian_minimal0: "<<std::endl<<AnaliJacobian_minimal0<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<std::endl<<AnaliJacobian_minimal1<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<std::endl<<AnaliJacobian_minimal2<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<std::endl<<AnaliJacobian_minimal3<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<QuaternionSplineSampleError,4> numbDifferentiator(quatSplineError);
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0*numJacobian_min0: "<<
             std::endl<<AnaliJacobian_minimal0*numJacobian_min0.inverse()<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min0 - AnaliJacobian_minimal0).norm(), 5e-6);


    // check jacobian_minimal1
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1*numJacobian_min1: "<<
             std::endl<<AnaliJacobian_minimal1*numJacobian_min1.inverse()<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min1 - AnaliJacobian_minimal1).norm(), 5e-6);


    // check jacobian_minimal2
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2*numJacobian_min2: "<<
             std::endl<<AnaliJacobian_minimal2*numJacobian_min2.inverse()<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min2 - AnaliJacobian_minimal2).norm(), 5e-6);


    // check jacobian_minimal3
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3*numJacobian_min3: "<<
             std::endl<<AnaliJacobian_minimal3*numJacobian_min3.inverse()<<std::endl;
    GTEST_ASSERT_LT((numJacobian_min3 - AnaliJacobian_minimal3).norm(), 5e-6);


    /*
    * Test opt
    */
    // solve
    ceres::Problem problem;
    ceres::LossFunction* loss_function =  new ceres::HuberLoss(1.0);

    QuaternionLocalParameter*  local_parameterization = new QuaternionLocalParameter;

    problem.AddParameterBlock(Cp0.data(), 4);
    problem.SetParameterization(Cp0.data(),local_parameterization);
    problem.AddParameterBlock(Cp1.data(), 4);
    problem.SetParameterization(Cp1.data(),local_parameterization);
    problem.AddParameterBlock(Cp2.data(), 4);
    problem.SetParameterization(Cp2.data(),local_parameterization);
    problem.AddParameterBlock(Cp3.data(), 4);
    problem.SetParameterization(Cp3.data(),local_parameterization);

    QuaternionSplineSampleError* quatSplineError1 = new QuaternionSplineSampleError(u1,Q_meas1);
    QuaternionSplineSampleError* quatSplineError2 = new QuaternionSplineSampleError(u2,Q_meas2);
    QuaternionSplineSampleError* quatSplineError3 = new QuaternionSplineSampleError(u3,Q_meas3);
    QuaternionSplineSampleError* quatSplineError4 = new QuaternionSplineSampleError(u4,Q_meas4);
    QuaternionSplineSampleError* quatSplineError5 = new QuaternionSplineSampleError(u5,Q_meas5);





    problem.AddResidualBlock(quatSplineError, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(quatSplineError1, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(quatSplineError2, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(quatSplineError3, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(quatSplineError4, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(quatSplineError5, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());



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




    std::cout<<"CP0_init: "<<Cp0_init.transpose()<<std::endl<<"After opt: "<<Cp0.transpose()<<std::endl;
    std::cout<<"CP1_init: "<<Cp1_init.transpose()<<std::endl<<"After opt: "<<Cp1.transpose()<<std::endl;
    std::cout<<"CP2_init: "<<Cp2_init.transpose()<<std::endl<<"After opt: "<<Cp2.transpose()<<std::endl;
    std::cout<<"CP3_init: "<<Cp3_init.transpose()<<std::endl<<"After opt: "<<Cp3.transpose()<<std::endl;


    Quaternion Qhat0 = QSUtility::EvaluateQS(u,Cp0,Cp1,Cp2,Cp3);
    Quaternion Qhat1 = QSUtility::EvaluateQS(u1,Cp0,Cp1,Cp2,Cp3);
    Quaternion Qhat2 = QSUtility::EvaluateQS(u2,Cp0,Cp1,Cp2,Cp3);
    Quaternion Qhat3 = QSUtility::EvaluateQS(u3,Cp0,Cp1,Cp2,Cp3);

    std::cout<<"Qmeas0: "<<Q_meas.transpose()<<std::endl<<"Qhat0: "<<Qhat0.transpose()<<std::endl;
    std::cout<<"Qmeas1: "<<Q_meas1.transpose()<<std::endl<<"Qhat1: "<<Qhat1.transpose()<<std::endl;
    std::cout<<"Qmeas2: "<<Q_meas2.transpose()<<std::endl<<"Qhat2: "<<Qhat2.transpose()<<std::endl;
    std::cout<<"Qmeas3: "<<Q_meas3.transpose()<<std::endl<<"Qhat3: "<<Qhat3.transpose()<<std::endl;


}
//
//TEST(Ceres, QuaternianOmegaCeres) {
//    double u = 0.6;
//    Quaternion Q_meas(0.0,1.0,0.0,1.9);
//    Q_meas = quatNorm(Q_meas);
//
//    double u1 = 0.4;
//    Quaternion Q_meas1(0.1,1.0,0.0,1.9);
//    Q_meas1 = quatNorm(Q_meas1);
//
//    double u2 = 0.1;
//    Quaternion Q_meas2(0.1,1.3,0.0,1.9);
//    Q_meas2 = quatNorm(Q_meas2);
//
//    double u3 = 0.9;
//    Quaternion Q_meas3(0.1,1.3,-0.7,1.9);
//    Q_meas3 = quatNorm(Q_meas3);
//
//    double u4 = 0.3;
//    Quaternion Q_meas4(0.1,1.3,-0.732,1.9);
//    Q_meas4 = quatNorm(Q_meas4);
//
//    double u5 = 0.5;
//    Quaternion Q_meas5(0.12,1.3,-0.7,1.9);
//    Q_meas5 = quatNorm(Q_meas5);
//
//
//    Quaternion Cp0,Cp1,Cp2,Cp3;
//    Cp0 = Cp1 = Cp2 = Cp3 = unitQuat<double>();
//    Cp0 = Quaternion(-0.0233343  ,0.538966 ,  0.805091,   0.246575); Cp0 = quatNorm(Cp0);
//    Cp1 = Quaternion(0.142278 ,  0.44318 ,-0.513372 ,  0.72097);  Cp1 = quatNorm(Cp1);
//    Cp2 = Quaternion(-0.112329,  0.379688,   0.34445,  0.851219);  Cp2 = quatNorm(Cp2);
//    Cp3 = Quaternion(-0.164781, -0.303314,  0.876392, -0.335836);  Cp3 = quatNorm(Cp3);
//
//    Quaternion Cp0_init,Cp1_init,Cp2_init,Cp3_init;
//    Cp0_init = Cp0;
//    Cp1_init = Cp1;
//    Cp2_init = Cp2;
//    Cp3_init = Cp3;
//
//
//
//    Quaternion Q_ba = QSUtility::EvaluateQS(0.5,Cp0,Cp1,Cp2,Cp3);
//    Quaternion dot_Q_ba = QSUtility::Evaluate_dot_QS(1, 0.5,Cp0,Cp1,Cp2,Cp3);
//    Eigen::Vector3d omega = QSUtility::w_in_body_frame(Q_ba, dot_Q_ba);
//    std::cout<<"Omega: "<< omega.transpose() << std::endl;
//
//
//    double* paramters[4] = {Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data()};
//    Eigen::Vector3d Residual;
//
//    QuaternionOmegaSampleFunctor* quaternionOmegaSampleFunctor
//            = new QuaternionOmegaSampleFunctor(0.5, 1.0, omega, 1.0);
//
//    QuaternionOmegaSampleAutoError* quaternionOmegaSampleAutoError
//            = new QuaternionOmegaSampleAutoError(quaternionOmegaSampleFunctor);
//
//
//
//    Eigen::Matrix<double,3,3,Eigen::RowMajor> AnaliJacobian_minimal0,AnaliJacobian_minimal1,AnaliJacobian_minimal2,AnaliJacobian_minimal3;
//    double* AnaliJacobians_minimal[4] = {AnaliJacobian_minimal0.data(),
//                                         AnaliJacobian_minimal1.data(),
//                                         AnaliJacobian_minimal2.data(),
//                                         AnaliJacobian_minimal3.data()};
//    Eigen::Matrix<double,3,4,Eigen::RowMajor> AnaliJacobian0,AnaliJacobian1,AnaliJacobian2,AnaliJacobian3;
//    double* AnaliJacobians[4] = {AnaliJacobian0.data(),
//                                 AnaliJacobian1.data(),
//                                 AnaliJacobian2.data(),
//                                 AnaliJacobian3.data()};
//
//
//    /*
//     *  Test Zero
//     */
//
//    quaternionOmegaSampleAutoError->EvaluateWithMinimalJacobians(paramters,Residual.data(),
//                                                                 AnaliJacobians,AnaliJacobians_minimal);
//
//    std::cout<<"Residual: "<<Residual.transpose() << std::endl;
////
//
//
//
//
//    /*
//     *  Test jacobians
//     */
//    Quaternion noise0, noise1, noise2, noise3;
//    Quaternion CP_noise0, CP_noise1, CP_noise2, CP_noise3;
//    noise0 = Quaternion(0.1,0,0,1);
//    noise1 = Quaternion(0.1,0.1,0,1);
//    noise2 = Quaternion(0.1,0,0.1,1);
//    noise3 = Quaternion(0,0,0.1,1);
//    CP_noise0 = quatMult(Cp0,noise0);CP_noise0 = quatNorm(CP_noise0);
//    CP_noise1 = quatMult(Cp1,noise1);CP_noise1 = quatNorm(CP_noise1);
//    CP_noise2 = quatMult(Cp2,noise2);CP_noise2 = quatNorm(CP_noise2);
//    CP_noise3 = quatMult(Cp3,noise3);CP_noise3 = quatNorm(CP_noise3);
//
//
//    double* paramters_noised[4] = {CP_noise0.data(),CP_noise1.data(),CP_noise2.data(),CP_noise3.data()};
//
//    quaternionOmegaSampleAutoError->EvaluateWithMinimalJacobians(paramters_noised,Residual.data(),
//                                                                 AnaliJacobians,AnaliJacobians_minimal);
//
//
////
//    // check jacobian_minimal0
//    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min0;
//    NumbDifferentiator<QuaternionOmegaSampleAutoError,4> numbDifferentiator(quaternionOmegaSampleAutoError);
//    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,0,numJacobian_min0.data());
//
//    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
//    std::cout<<"AnaliJacobian_minimal0: "<<
//             std::endl<<AnaliJacobian_minimal0<<std::endl;
//
//    GTEST_ASSERT_LT((numJacobian_min0 - AnaliJacobian_minimal0).norm(), 5e-6);
//
////
//    // check jacobian_minimal1
//    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min1;
//    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,1,numJacobian_min1.data());
//
//    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
//    std::cout<<"AnaliJacobian_minimal1: "<<
//             std::endl<<AnaliJacobian_minimal1<<std::endl;
//    GTEST_ASSERT_LT((numJacobian_min1 - AnaliJacobian_minimal1).norm(), 5e-6);
//
//
//
////
//    // check jacobian_minimal2
//    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min2;
//    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,2,numJacobian_min2.data());
//
//    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
//    std::cout<<"AnaliJacobian_minimal2: "<<
//             std::endl<<AnaliJacobian_minimal2<<std::endl;
//    GTEST_ASSERT_LT((numJacobian_min2 - AnaliJacobian_minimal2).norm(), 5e-6);
//
//
//    // check jacobian_minimal3
//    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min3;
//    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,3,numJacobian_min3.data());
//
//    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
//    std::cout<<"AnaliJacobian_minimal3: "<<
//             std::endl<<AnaliJacobian_minimal3<<std::endl;
//    GTEST_ASSERT_LT((numJacobian_min3 - AnaliJacobian_minimal3).norm(), 5e-6);
//
//
//}
//


