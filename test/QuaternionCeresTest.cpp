#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionError.hpp"
#include <gtest/gtest.h>
#include <ceres/gradient_checker.h>


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


TEST(ceres, quaternion){
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
