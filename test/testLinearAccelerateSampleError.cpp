#include <utility/timer.h>
#include "pose-spline/Pose.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
#include "pose-spline/PoseSplineUtility.hpp"
#include "pose-spline/LinearAccelerateSampleError.hpp"
#include "pose-spline/NumbDifferentiator.hpp"

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

//    Pose<double> P_meas = PSUtility::EvaluatePS(u, pose0, pose1, pose2, pose3);
//    Pose<double> P_meas1 = PSUtility::EvaluatePS(u1, pose0, pose1, pose2, pose3);
//    Pose<double> P_meas2 = PSUtility::EvaluatePS(u2, pose0, pose1, pose2, pose3);
//    Pose<double> P_meas3 = PSUtility::EvaluatePS(u3, pose0, pose1, pose2, pose3);
//    Pose<double> P_meas4 = PSUtility::EvaluatePS(u4, pose0, pose1, pose2, pose3);
//    Pose<double> P_meas5 = PSUtility::EvaluatePS(u5, pose0, pose1, pose2, pose3);
//
//    std::cout<<"P_meas: "<<P_meas.parameters().transpose()<<std::endl;

    double u_meas = u5;
    Eigen::Vector3d a_meas = PSUtility::EvaluateLinearAccelerate(u_meas, 1.0, pose0,pose1,pose2,pose3);



    /**
     *  Zero Test
     */


    double* paramters[4] = {pose0.parameterPtr(), pose1.parameterPtr(),
                            pose2.parameterPtr(), pose3.parameterPtr()};
    Eigen::Matrix<double, 3, 1> Residual;


    Eigen::Matrix<double,3,6,Eigen::RowMajor> AnaliJacobian_minimal0,AnaliJacobian_minimal1,AnaliJacobian_minimal2,AnaliJacobian_minimal3;
    double* AnaliJacobians_minimal[4] = {AnaliJacobian_minimal0.data(),
                                         AnaliJacobian_minimal1.data(),
                                         AnaliJacobian_minimal2.data(),
                                         AnaliJacobian_minimal3.data()};
    Eigen::Matrix<double,3,7,Eigen::RowMajor> AnaliJacobian0,AnaliJacobian1,AnaliJacobian2,AnaliJacobian3;
    double* AnaliJacobians[4] = {AnaliJacobian0.data(),
                                 AnaliJacobian1.data(),
                                 AnaliJacobian2.data(),
                                 AnaliJacobian3.data()};

    LinearAccelerateSampleError* linearAccelerateSampleError = new LinearAccelerateSampleError(u_meas, 1.0, a_meas);
    linearAccelerateSampleError->EvaluateWithMinimalJacobians(paramters, Residual.data(), AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;
//
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


    TimeStatistics::Timer anali_timer;
    linearAccelerateSampleError->EvaluateWithMinimalJacobians(paramters_noised, Residual.data(),
                                                        AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"anali_timer: "<<anali_timer.stopAndGetSeconds()<<std::endl;
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<LinearAccelerateSampleError,4> numbDifferentiator(linearAccelerateSampleError);
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
             std::endl<<AnaliJacobian_minimal0<<std::endl;

    // check jacobian_minimal1
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl;

    // check jacobian_minimal2
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl;

    // check jacobian_minimal1
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl;


    /*
     *  Test Auto Jacobian
     */

    LinearAccelerateSampleFunctor* linearAccelerateSampleFunctor
            = new LinearAccelerateSampleFunctor(u_meas, 1.0, a_meas, 1.0);
    LinearAccelerateSampleError* linearAccelerateSampleError1
                                        = new LinearAccelerateSampleError(linearAccelerateSampleFunctor);
    TimeStatistics::Timer auto_timer;
    linearAccelerateSampleError1->AutoEvaluateWithMinimalJacobians(paramters_noised, Residual.data(),
                                                              AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"auto_timer: "<<auto_timer.stopAndGetSeconds()<<std::endl;


    std::cout<<"auto_minimal0: "<<
             std::endl<<AnaliJacobian_minimal0<<std::endl;
    std::cout<<"auto_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl;
    std::cout<<"auto_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl;
    std::cout<<"auto_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl;

    return 0;
}