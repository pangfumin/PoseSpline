#include "pose-spline/Pose.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
#include "pose-spline/PoseSplineUtility.hpp"
#include "pose-spline/PoseSplineSampleError.hpp"
#include "pose-spline/NumbDifferentiator.hpp"

int main() {


    Pose pose0;
    pose0.setRandom();
    std::cout<<pose0.parameters().transpose()<<std::endl;

    Pose pose1;
    pose1.setRandom();
    std::cout<<pose1.parameters().transpose()<<std::endl;

    Pose pose2;
    pose2.setRandom();
    std::cout<<pose2.parameters().transpose()<<std::endl;

    Pose pose3;
    pose3.setRandom();
    std::cout<<pose3.parameters().transpose()<<std::endl;

    double u = 0.5;

    Pose P_meas = PSUtility::EvaluateQS(u, pose0, pose1, pose2, pose3);

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

    Pose noise;
    noise.setRandom();
    Pose pose0_noised = pose0*noise;

    noise.setRandom();
    Pose pose1_noised = pose1*noise;

    noise.setRandom();
    Pose pose2_noised = pose2*noise;

    noise.setRandom();
    Pose pose3_noised = pose3*noise;

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

    return 0;
}