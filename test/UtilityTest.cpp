
#include <iostream>
#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include "PoseSpline/QuaternionSpline.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"


TEST(Geometry, quaternion){
// test exp and log
    Eigen::Vector3d delta_phi(1.1,1.43,-0.9);

    Quaternion Q_exp = quatExp(delta_phi);
    Eigen::Vector3d phi_log = quatLog(Q_exp);
    GTEST_ASSERT_EQ((delta_phi - phi_log).squaredNorm() < 0.0001,true);

//    std::cout<<"S(phi)*L(q): "<<std::endl<<quatS(delta_phi)*quatL(Q_exp)<<std::endl;

    RotMat rot  = axisAngleToRotMat(delta_phi);
    Quaternion quatFromRot = rotMatToQuat(rot);


    GTEST_ASSERT_EQ((quatFromRot - Q_exp).squaredNorm() < 0.0001,true);

    RotMat rot_X = rotX(3.1415/2);
    RotMat rotAA = axisAngleToRotMat(Eigen::Vector3d(3.1415/2,0,0));

    Eigen::Matrix4d Q_R= quatRightComp(Q_exp);
    Eigen::Matrix4d  invQ_R= quatRightComp(quatInv(Q_exp));

//    std::cout<<Q_R*invQ_R<<std::endl;
    GTEST_ASSERT_LT((Q_R*invQ_R  -
                     Eigen::Matrix<double,4,4>::Identity()).squaredNorm() , 1e-8);


    Quaternion Cp0,Cp1,Cp2,Cp3;
    Cp0 = Cp1 = Cp2 = Cp3 = unitQuat<double>();
    Cp0 = Quaternion(-0.0233343  ,0.538966 ,  0.805091,   0.246575); Cp0 = quatNorm(Cp0);
    Cp1 = Quaternion(0.142278 ,  0.44318 ,-0.513372 ,  0.72097);  Cp1 = quatNorm(Cp1);
    Cp2 = Quaternion(-0.112329,  0.379688,   0.34445,  0.851219);  Cp2 = quatNorm(Cp2);
    Cp3 = Quaternion(-0.164781, -0.303314,  0.876392, -0.335836);  Cp3 = quatNorm(Cp3);

    /*
     * test dr_dt
     */
    double dt = 0.1;
    double u = 0.19;

    double b1 = QSUtility::beta1(u);
    Eigen::Vector3d phi1 = QSUtility::Phi(Cp1,Cp2);
    double db1 = QSUtility::dot_beta1(dt,u);
    Quaternion  r = QSUtility::r(b1,phi1);
    Quaternion  dot_r = QSUtility::dr_dt(db1,b1,Cp1,Cp2);
    dot_r = quatNorm(dot_r);

    double eps = 1e-5;
    double u_p_eps = u + eps/dt;
    double u_m_eps = u - eps/dt;

    // plus
    double b1_p = QSUtility::beta1(u_p_eps);
    double db1_p = QSUtility::dot_beta1(dt,u_p_eps);
    Quaternion  r_p = QSUtility::r(b1_p,phi1);

    double b1_m = QSUtility::beta1(u_m_eps);
    double db1_m = QSUtility::dot_beta1(dt,u_m_eps);
    Quaternion  r_m = QSUtility::r(b1_m,phi1);

    Quaternion numdiff_r = (r_p - r_m)/(2.0*eps);
    numdiff_r = quatNorm(numdiff_r);

    GTEST_ASSERT_LT((dot_r - numdiff_r).norm(), 1e-5);


    /*
     * test d2r_dt2
     */

    double ddb1 = QSUtility::dot_dot_beta1(dt,u);
    Quaternion  dot_dot_r = QSUtility::d2r_dt2(ddb1,db1,b1,Cp2,Cp3);

    Quaternion  dot_r_p = QSUtility::dr_dt(db1_p,b1_p,Cp2,Cp3);
    Quaternion  dot_r_m = QSUtility::dr_dt(db1_m,b1_m,Cp2,Cp3);

    Quaternion numDiff_drdt = (dot_r_p - dot_r_m)/(2.0*eps);

    GTEST_ASSERT_LT((dot_dot_r - numDiff_drdt).norm(), 1e-5);

}

TEST(Geometry, Hamilton_VS_JPL) {
    for (int i  = 0; i < 1000; i++){
        Eigen::Vector3d tmp = Eigen::Vector3d::Random();
        Eigen::AngleAxisd aa(tmp.norm(), tmp /tmp.norm());
        Eigen::Matrix3d rot = aa.toRotationMatrix();
        Eigen::Quaterniond hamiltonQuat(rot);
        Quaternion JPLQuat = rotMatToQuat(rot);
//    std::cout << hamiltonQuat.coeffs().transpose() << std::endl;
//    std::cout << JPLQuat.transpose() << std::endl;
        GTEST_ASSERT_LT((hamiltonQuat.inverse().coeffs() - JPLQuat).norm(), 1e6);
    }

}
