#include "extern/RotateVectorError.hpp"
#include "extern/TransformVectorError.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include <gtest/gtest.h>
TEST(ceres, SplineRotationError) {
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
                                   q2_noised.data(), q3_noised.data()};


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
    GTEST_ASSERT_LT((numJacobian_min0 - AnaliJacobian_minimal0).norm(), 1e6);


// check jacobian_minimal1
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl;
    GTEST_ASSERT_LT((numJacobian_min1 - AnaliJacobian_minimal1).norm(), 1e6);


// check jacobian_minimal2
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min2 - AnaliJacobian_minimal2).norm(), 1e6);


// check jacobian_minimal3
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min3 - AnaliJacobian_minimal3).norm(), 1e6);
}

TEST(ceres, SplineTransformError) {
    Pose<double> T0, T1, T2, T3;
    T0.setRandom();
    T1.setRandom();
    T2.setRandom();
    T3.setRandom();

    std::cout << T0.coeffs().transpose() << std::endl;
    std::cout << T1.coeffs().transpose() << std::endl;
    std::cout << T2.coeffs().transpose() << std::endl;
    std::cout << T3.coeffs().transpose() << std::endl;

    double u = 0.5;
    Pose<double> T_meas = PSUtility::EvaluatePS(u, T0, T1, T2, T3);
    std::cout << "T_meas: " << T_meas.coeffs().transpose() << std::endl;

    Eigen::Vector3d originalVector(1,0,0);
    Eigen::Matrix3d R_meas = T_meas.C();
    Eigen::Vector3d t_meas = T_meas.r();
    Eigen::Vector3d rotatedVector = R_meas * originalVector;
    Eigen::Vector3d transformedVector = R_meas * originalVector + t_meas;
    std::cout<<"rotatedVector: "<<rotatedVector.transpose()<<std::endl;
    std::cout<<"transformedVector: "<<transformedVector.transpose()<<std::endl;

    /**
    *  Zero Test
    */


    double* paramters[4] = {T0.parameterPtr(), T1.parameterPtr(),
                            T2.parameterPtr(), T3.parameterPtr()};
    Eigen::Matrix<double, 3, 1> Residual;


    Eigen::Matrix<double,3,6,Eigen::RowMajor> AnaliJacobian_minimal0,
                                                AnaliJacobian_minimal1,
                                                AnaliJacobian_minimal2,
                                                AnaliJacobian_minimal3;
    double* AnaliJacobians_minimal[4] = {AnaliJacobian_minimal0.data(),
                                         AnaliJacobian_minimal1.data(),
                                         AnaliJacobian_minimal2.data(),
                                         AnaliJacobian_minimal3.data()};
    Eigen::Matrix<double,3,7,Eigen::RowMajor> AnaliJacobian0,
                                            AnaliJacobian1,
                                            AnaliJacobian2,
                                            AnaliJacobian3;
    double* AnaliJacobians[4] = {AnaliJacobian0.data(),
                                 AnaliJacobian1.data(),
                                 AnaliJacobian2.data(),
                                 AnaliJacobian3.data()};


    TransformVectorError* transformVectorError = new TransformVectorError(u,originalVector, transformedVector);
    transformVectorError->EvaluateWithMinimalJacobians(paramters, Residual.data(), AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;
    GTEST_ASSERT_LT(Residual.norm() , 1e-8);

    Quaternion q0 = T0.q();
    Quaternion q1 = T1.q();
    Quaternion q2 = T2.q();
    Quaternion q3 = T3.q();
    double* quat_paramters[4] = {q0.data(), q1.data(),
                            q2.data(), q3.data()};
    Eigen::Matrix<double, 3, 1> quat_Residual;


    Eigen::Matrix<double,3,3,Eigen::RowMajor> quat_AnaliJacobian_minimal0,
                                                quat_AnaliJacobian_minimal1,
                                                quat_AnaliJacobian_minimal2,
                                                quat_AnaliJacobian_minimal3;
    double* quat_AnaliJacobians_minimal[4] = {quat_AnaliJacobian_minimal0.data(),
                        quat_AnaliJacobian_minimal1.data(),
                        quat_AnaliJacobian_minimal2.data(),
                        quat_AnaliJacobian_minimal3.data()};
    Eigen::Matrix<double,3,4,Eigen::RowMajor> quat_AnaliJacobian0,
                                        quat_AnaliJacobian1,
                                        quat_AnaliJacobian2,
                                        quat_AnaliJacobian3;
    double* quat_AnaliJacobians[4] = {quat_AnaliJacobian0.data(),
                        quat_AnaliJacobian1.data(),
                        quat_AnaliJacobian2.data(),
                        quat_AnaliJacobian3.data()};

    RoatateVectorError* roatateVectorError = new RoatateVectorError(u,originalVector, rotatedVector);
    roatateVectorError->EvaluateWithMinimalJacobians(quat_paramters, quat_Residual.data(), quat_AnaliJacobians, quat_AnaliJacobians_minimal);
    std::cout<<"residual: "<< quat_Residual.transpose()<<std::endl;




    /**
   *  Test jacobians
   */

    Pose<double> T0_noised, T1_noised, T2_noised, T3_noised;
    Pose<double> noise;
    noise.setRandom(0.3, 0.3);
    T0_noised = T0*noise;
    noise.setRandom(0.3, 0.3);
    T1_noised = T1*noise;
    noise.setRandom(0.3, 0.3);
    T2_noised = T2*noise;
    noise.setRandom(0.3, 0.3);
    T3_noised = T3*noise;


    double* paramters_noised[4] = {T0_noised.parameterPtr(), T1_noised.parameterPtr(),
                                   T2_noised.parameterPtr(), T3_noised.parameterPtr()};


    transformVectorError->EvaluateWithMinimalJacobians(paramters_noised, Residual.data(),
                                                     AnaliJacobians, AnaliJacobians_minimal);
    std::cout<<"residual: "<< Residual.transpose()<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<TransformVectorError,4> numbDifferentiator(transformVectorError);
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
             std::endl<<AnaliJacobian_minimal0<<std::endl;
    GTEST_ASSERT_LT((numJacobian_min0 - AnaliJacobian_minimal0).norm(), 1e6);

//
// check jacobian_minimal1
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl;
    GTEST_ASSERT_LT((numJacobian_min1 - AnaliJacobian_minimal1).norm(), 1e6);

// check jacobian_minimal2
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min2 - AnaliJacobian_minimal2).norm(), 1e6);

//
// check jacobian_minimal3
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl;

    GTEST_ASSERT_LT((numJacobian_min3 - AnaliJacobian_minimal3).norm(), 1e6);


    Quaternion q0_noised = T0_noised.q();
    Quaternion q1_noised = T1_noised.q();
    Quaternion q2_noised = T2_noised.q();
    Quaternion q3_noised = T3_noised.q();



    double* quat_paramters_noised[4] = {q0_noised.data(), q1_noised.data(),
                                   q2_noised.data(), q3_noised.data()};


    roatateVectorError->EvaluateWithMinimalJacobians(quat_paramters_noised, quat_Residual.data(),
                                                     quat_AnaliJacobians, quat_AnaliJacobians_minimal);
    std::cout<<"residual: "<< quat_Residual.transpose()<<std::endl;


    // check jacobian_minimal0
    Eigen::Matrix<double,3,3,Eigen::RowMajor> quat_numJacobian_min0;
    NumbDifferentiator<RoatateVectorError,4> quat_numbDifferentiator(roatateVectorError);
    quat_numbDifferentiator.df_r_xi<3,4,3,QuaternionLocalParameter>(quat_paramters_noised,0,quat_numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<quat_numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
             std::endl<<quat_AnaliJacobian_minimal0<<std::endl;
    GTEST_ASSERT_LT((quat_numJacobian_min0 - quat_AnaliJacobian_minimal0).norm(), 1e6);

}