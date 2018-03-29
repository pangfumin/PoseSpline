#include "pose-spline/PoseSplineUtility.hpp"

Pose PSUtility::EvaluateQS(double u, const Pose& P0, const Pose& P1,
                const Pose& P2, const Pose& P3) {


    double b1 = QSUtility::beta1(u);
    double b2 = QSUtility::beta2(u);
    double b3 = QSUtility::beta3(u);
    Quaternion Q0 = P0.rotation();
    Quaternion Q1 = P1.rotation();
    Quaternion Q2 = P2.rotation();
    Quaternion Q3 = P3.rotation();
    Eigen::Vector3d Phi1 = QSUtility::Phi(Q0,Q1);
    Eigen::Vector3d Phi2 = QSUtility::Phi(Q1,Q2);
    Eigen::Vector3d Phi3 = QSUtility::Phi(Q2,Q3);

    Quaternion r1 = QSUtility::r(b1,Phi1);
    Quaternion r2 = QSUtility::r(b2,Phi2);
    Quaternion r3 = QSUtility::r(b3,Phi3);

    Eigen::Vector3d V0 = P0.translation();
    Eigen::Vector3d V1 = P1.translation();
    Eigen::Vector3d V2 = P2.translation();
    Eigen::Vector3d V3 = P3.translation();

    Eigen::Vector3d V = V0 + b1*(V1 - V0) +  b2*(V2 - V1) + b3*(V3 - V2);

    return Pose( V, quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3);
}

