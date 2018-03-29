#include "extern/RotateVectorError.hpp"
#include "pose-spline/QuaternionSplineUtility.hpp"
#include "pose-spline/NumbDifferentiator.hpp"

int main() {
    Quaternion q0 = randomQuat<double>();
    std::cout<<"q0: "<< q0.transpose()<<std::endl;

    Quaternion q1 = randomQuat<double>();
    std::cout<<"q1: "<< q1.transpose()<<std::endl;

    Quaternion q2 = randomQuat<double>();
    std::cout<<"q2: "<< q2.transpose()<<std::endl;

    Quaternion q3 = randomQuat<double>();
    std::cout<<"q3: "<< q3.transpose()<<std::endl;

    double u = 0.5;
    Quaternion q_meas = QSUtility::EvaluateQS(u, q0,q1,q2,q3);
    std::cout<<"q_meas: "<<q_meas.transpose()<<std::endl;

    Eigen::Vector3d originalVector(1,0,0);
    Eigen::Vector3d rotatedVector = quatToRotMat(q_meas)*originalVector;
    std::cout<<"rotatedVector: "<<rotatedVector.transpose()<<std::endl;

    /**
    *  Zero Test
    */


    double* paramters[4] = {q0.data(), q1.data(),
                            q2.data(), q3.data()};
    Eigen::Matrix<double, 3, 1> Residual;


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

    RoatateVectorError* roatateVectorError = new RoatateVectorError(u,originalVector, rotatedVector);
    roatateVectorError->EvaluateWithMinimalJacobians(paramters, Residual.data(), AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;

    /**
   *  Test jacobians
   */

    Quaternion noise = randomQuat<double>();
    Quaternion q0_noised = quatMult(q0,noise);

    noise = randomQuat<double>();
    Quaternion q1_noised = quatMult(q1,noise);


    noise = randomQuat<double>();
    Quaternion q2_noised = quatMult(q2,noise);


    noise = randomQuat<double>();
    Quaternion q3_noised = quatMult(q3,noise);


    double* paramters_noised[4] = {q0_noised.data(), q1_noised.data(),
                                   q2_noised.data(), q3.data()};


    roatateVectorError->EvaluateWithMinimalJacobians(paramters_noised, Residual.data(),
                                                        AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<RoatateVectorError,4> numbDifferentiator(roatateVectorError);
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
             std::endl<<AnaliJacobian_minimal0<<std::endl;

    // check jacobian_minimal1
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl;

    // check jacobian_minimal2
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl;

    // check jacobian_minimal3
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl;

    return 0;
}