#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"
#include "PoseSpline/PoseSplineSampleError.hpp"
#include "PoseSpline/NumbDifferentiator.hpp"

int main() {
    Pose<double> pose0;
    pose0.setRandom();
    std::cout<<pose0.parameters().transpose()<<std::endl;

    Pose<double> pose1;
    pose1.setRandom();
    std::cout<<pose1.parameters().transpose()<<std::endl;

    Pose<double> pose2;
    pose2.setRandom();
    std::cout<<pose2.parameters().transpose()<<std::endl;

    Pose<double> pose3;
    pose3.setRandom();
    std::cout<<pose3.parameters().transpose()<<std::endl;

    double u = 0.5;
    double u1 = 0.15;
    double u2 = 0.25;
    double u3 = 0.45;
    double u4 = 0.75;
    double u5 = 0.85;

    Pose<double> P_meas = PSUtility::EvaluatePS(u, pose0, pose1, pose2, pose3);
    Pose<double> P_meas1 = PSUtility::EvaluatePS(u1, pose0, pose1, pose2, pose3);
    Pose<double> P_meas2 = PSUtility::EvaluatePS(u2, pose0, pose1, pose2, pose3);
    Pose<double> P_meas3 = PSUtility::EvaluatePS(u3, pose0, pose1, pose2, pose3);
    Pose<double> P_meas4 = PSUtility::EvaluatePS(u4, pose0, pose1, pose2, pose3);
    Pose<double> P_meas5 = PSUtility::EvaluatePS(u5, pose0, pose1, pose2, pose3);

    std::cout<<"P_meas: "<<P_meas.parameters().transpose()<<std::endl;



    /**
     *  Zero Test
     */
    double* paramters[4] = {pose0.parameterPtr(), pose1.parameterPtr(),
                            pose2.parameterPtr(), pose3.parameterPtr()};
    Eigen::Matrix<double, 6, 1> Residual;


    Eigen::Matrix<double,6,6,Eigen::RowMajor> AnaliJacobian_minimal0,AnaliJacobian_minimal1,AnaliJacobian_minimal2,AnaliJacobian_minimal3;
    double* AnaliJacobians_minimal[4] = {AnaliJacobian_minimal0.data(),
                                         AnaliJacobian_minimal1.data(),
                                         AnaliJacobian_minimal2.data(),
                                         AnaliJacobian_minimal3.data()};
    Eigen::Matrix<double,6,7,Eigen::RowMajor> AnaliJacobian0,AnaliJacobian1,AnaliJacobian2,AnaliJacobian3;
    double* AnaliJacobians[4] = {AnaliJacobian0.data(),
                                 AnaliJacobian1.data(),
                                 AnaliJacobian2.data(),
                                 AnaliJacobian3.data()};

    PoseSplineSampleError* poseSplineSampleError = new PoseSplineSampleError(u, P_meas);
    poseSplineSampleError->EvaluateWithMinimalJacobians(paramters, Residual.data(), AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;

    /**
    *  Test jacobians
    */

    Pose<double> noise;
    noise.setRandom();
    Pose<double> pose0_noised = pose0*noise;

    noise.setRandom();
    Pose<double> pose1_noised = pose1*noise;

    noise.setRandom();
    Pose<double> pose2_noised = pose2*noise;

    noise.setRandom();
    Pose<double> pose3_noised = pose3*noise;

    double* paramters_noised[4] = {pose0_noised.parameterPtr(), pose1_noised.parameterPtr(),
                            pose2_noised.parameterPtr(), pose3_noised.parameterPtr()};


    poseSplineSampleError->EvaluateWithMinimalJacobians(paramters_noised, Residual.data(),
                                                        AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,6,6,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<PoseSplineSampleError,4> numbDifferentiator(poseSplineSampleError);
    numbDifferentiator.df_r_xi<6,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
             std::endl<<AnaliJacobian_minimal0<<std::endl;

    // check jacobian_minimal1
    Eigen::Matrix<double,6,6,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<6,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl;

    // check jacobian_minimal2
    Eigen::Matrix<double,6,6,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<6,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl;

    // check jacobian_minimal1
    Eigen::Matrix<double,6,6,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<6,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl;

    /*
     * Optimization test
     */

    pose0 = pose0_noised;
    pose1 = pose1_noised;
    pose2 = pose2_noised;
    pose3 = pose3_noised;

    ceres::Problem problem;
    ceres::LossFunction* loss_function =  new ceres::HuberLoss(1.0);

    PoseLocalParameter*  local_parameterization = new PoseLocalParameter;

    problem.AddParameterBlock(pose0.parameterPtr(), 7);
    problem.SetParameterization(pose0.parameterPtr(),local_parameterization);
    problem.AddParameterBlock(pose1.parameterPtr(), 7);
    problem.SetParameterization(pose1.parameterPtr(),local_parameterization);
    problem.AddParameterBlock(pose2.parameterPtr(), 7);
    problem.SetParameterization(pose2.parameterPtr(),local_parameterization);
    problem.AddParameterBlock(pose3.parameterPtr(), 7);
    problem.SetParameterization(pose3.parameterPtr(),local_parameterization);

    PoseSplineSampleError* quatSplineError = new PoseSplineSampleError(u1,P_meas);
    PoseSplineSampleError* quatSplineError1 = new PoseSplineSampleError(u2,P_meas1);
    PoseSplineSampleError* quatSplineError2 = new PoseSplineSampleError(u3,P_meas2);
    PoseSplineSampleError* quatSplineError3 = new PoseSplineSampleError(u4,P_meas3);
    PoseSplineSampleError* quatSplineError4 = new PoseSplineSampleError(u5,P_meas4);
    PoseSplineSampleError* quatSplineError5 = new PoseSplineSampleError(u5,P_meas5);


    problem.AddResidualBlock(quatSplineError, loss_function, pose0.parameterPtr(),pose1.parameterPtr(),
                             pose2.parameterPtr(),pose3.parameterPtr());
    problem.AddResidualBlock(quatSplineError1, loss_function, pose0.parameterPtr(),pose1.parameterPtr(),
                             pose2.parameterPtr(),pose3.parameterPtr());
    problem.AddResidualBlock(quatSplineError2, loss_function, pose0.parameterPtr(),pose1.parameterPtr(),
                             pose2.parameterPtr(),pose3.parameterPtr());
    problem.AddResidualBlock(quatSplineError3, loss_function, pose0.parameterPtr(),pose1.parameterPtr(),
                             pose2.parameterPtr(),pose3.parameterPtr());
    problem.AddResidualBlock(quatSplineError4, loss_function, pose0.parameterPtr(),pose1.parameterPtr(),
                             pose2.parameterPtr(),pose3.parameterPtr());
    problem.AddResidualBlock(quatSplineError5, loss_function, pose0.parameterPtr(),pose1.parameterPtr(),
                             pose2.parameterPtr(),pose3.parameterPtr());



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




//    std::cout<<"CP0_init: "<<Cp0_init.transpose()<<std::endl<<"After opt: "<<Cp0.transpose()<<std::endl;
//    std::cout<<"CP1_init: "<<Cp1_init.transpose()<<std::endl<<"After opt: "<<Cp1.transpose()<<std::endl;
//    std::cout<<"CP2_init: "<<Cp2_init.transpose()<<std::endl<<"After opt: "<<Cp2.transpose()<<std::endl;
//    std::cout<<"CP3_init: "<<Cp3_init.transpose()<<std::endl<<"After opt: "<<Cp3.transpose()<<std::endl;
//

    Pose<double> Qhat0 = PSUtility::EvaluatePS(u,pose0,pose1,pose2,pose3);
    Pose<double> Qhat1 = PSUtility::EvaluatePS(u1,pose0,pose1,pose2,pose3);
    Pose<double> Qhat2 = PSUtility::EvaluatePS(u2,pose0,pose1,pose2,pose3);
    Pose<double> Qhat3 = PSUtility::EvaluatePS(u3,pose0,pose1,pose2,pose3);

    std::cout<<"Qmeas0: "<<P_meas.r().transpose()<<" "<< P_meas.q().transpose()<<std::endl
             <<"Qhat0: "<<Qhat0.r().transpose()<<" "<<Qhat0.q().transpose()<<std::endl;

    std::cout<<"Qmeas1: "<<P_meas1.r().transpose()<<" "<< P_meas1.q().transpose()<<std::endl
             <<"Qhat1: "<<Qhat1.r().transpose()<<" "<<Qhat1.q().transpose()<<std::endl;

    std::cout<<"Qmeas2: "<<P_meas2.r().transpose()<<" "<< P_meas2.q().transpose()<<std::endl
             <<"Qhat2: "<<Qhat2.r().transpose()<<" "<<Qhat2.q().transpose()<<std::endl;

    std::cout<<"Qmeas3: "<<P_meas3.r().transpose()<<" "<< P_meas3.q().transpose()<<std::endl
             <<"Qhat3: "<<Qhat3.r().transpose()<<" "<<Qhat3.q().transpose()<<std::endl;





    return 0;
}