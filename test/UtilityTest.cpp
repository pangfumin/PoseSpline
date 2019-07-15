
#include <iostream>
#include <ceres/ceres.h>

#include "PoseSpline/QuaternionSpline.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"


int main(){



// test exp and log
    Eigen::Vector3d delta_phi(1.1,1.43,-0.9);

    Quaternion Q_exp = quatExp(delta_phi);
    Eigen::Vector3d phi_log = quatLog(Q_exp);
    CHECK_EQ((delta_phi - phi_log).squaredNorm() < 0.0001,true);

    std::cout<<"S(phi)*L(q): "<<std::endl<<quatS(delta_phi)*quatL(Q_exp)<<std::endl;

    RotMat rot  = axisAngleToRotMat(delta_phi);
    Quaternion quatFromRot = rotMatToQuat(rot);


    CHECK_EQ((quatFromRot - Q_exp).squaredNorm() < 0.0001,true);

    RotMat rot_X = rotX(3.1415/2);
    RotMat rotAA = axisAngleToRotMat(Eigen::Vector3d(3.1415/2,0,0));



    Eigen::Matrix4d Q_R= quatRightComp(Q_exp);
    Eigen::Matrix4d  invQ_R= quatRightComp(quatInv(Q_exp));

    std::cout<<Q_R*invQ_R<<std::endl;


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
    std::cout<<phi1.transpose()<<std::endl;
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


    std::cout<<"Num diff dr_dt and analytics dr_dt: "<<std::endl;
    std::cout<< dot_r.transpose()<<std::endl;
    std::cout<< numdiff_r.transpose()<<std::endl;

    /*
     * test d2r_dt2
     */

    double ddb1 = QSUtility::dot_dot_beta1(dt,u);
    Quaternion  dot_dot_r = QSUtility::d2r_dt2(ddb1,db1,b1,Cp2,Cp3);

    Quaternion  dot_r_p = QSUtility::dr_dt(db1_p,b1_p,Cp2,Cp3);
    Quaternion  dot_r_m = QSUtility::dr_dt(db1_m,b1_m,Cp2,Cp3);

    Quaternion numDiff_drdt = (dot_r_p - dot_r_m)/(2.0*eps);

    std::cout<<"Num diff d2r_dt2 and analytics d2r_dt2: "<<std::endl;
    std::cout<< dot_dot_r.transpose()<<std::endl;
    std::cout<< numDiff_drdt.transpose()<<std::endl;


    return 0;
}
