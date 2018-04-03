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


Pose PoseSplineEvaluation::operator() (double u, const Pose& P0, const Pose& P1,
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

    Q = quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;
    t = V0 + b1*(V1 - V0) +  b2*(V2 - V1) + b3*(V3 - V2);

    Eigen::Matrix<double,4,3> Vee = QSUtility::V<double>();

    Eigen::Vector3d BetaPhi1 = b1*Phi1;
    Eigen::Vector3d BetaPhi2 = b2*Phi2;
    Eigen::Vector3d BetaPhi3 = b3*Phi3;
    Eigen::Matrix3d S1 = quatS(BetaPhi1);
    Eigen::Matrix3d S2 = quatS(BetaPhi2);
    Eigen::Matrix3d S3 = quatS(BetaPhi3);


    Quaternion invQ0Q1 = quatLeftComp(quatInv<double>(Q0))*Q1;
    Quaternion invQ1Q2 = quatLeftComp(quatInv<double>(Q1))*Q2;
    Quaternion invQ2Q3 = quatLeftComp(quatInv<double>(Q2))*Q3;
    Eigen::Matrix3d L1 = quatL(invQ0Q1);
    Eigen::Matrix3d L2 = quatL(invQ1Q2);
    Eigen::Matrix3d L3 = quatL(invQ2Q3);

    Eigen::Matrix3d C0 = quatToRotMat<double>(Q0);
    Eigen::Matrix3d C1 = quatToRotMat<double>(Q1);
    Eigen::Matrix3d C2 = quatToRotMat<double>(Q2);


    Eigen::Matrix<double,4,3> temp0;
    Eigen::Matrix<double,4,3> temp01;
    Eigen::Matrix<double,4,3> temp12;
    Eigen::Matrix<double,4,3> temp23;


    temp0 = quatRightComp<double>(quatLeftComp<double>(r1)*quatLeftComp<double>(r2)*r3)*quatRightComp<double>(Q0)*Vee;
    temp01 = quatLeftComp<double>(Q0)*quatRightComp<double>(quatLeftComp<double>(r2)*r3)*quatRightComp<double>(r1)*Vee*S1*b1*L1*C0.transpose();
    temp12 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r1)*quatRightComp<double>(r3)*quatRightComp<double>(r2)*Vee*S2*b2*L2*C1.transpose();
    temp23 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r1)*quatLeftComp<double>(r2)*quatRightComp<double>(r3)*Vee*S3*b3*L3*C2.transpose();

    JacobianTrans0 = (1 - b1)*Eigen::Matrix3d::Identity();
    JacobianTrans1 = (b1 - b2)*Eigen::Matrix3d::Identity();
    JacobianTrans2 = (b2 - b3)*Eigen::Matrix3d::Identity();
    JacobianTrans3 = b3*Eigen::Matrix3d::Identity();


    JacobianRotate0 = temp0 - temp01;
    JacobianRotate1 = temp01 - temp12;
    JacobianRotate2 = temp12 - temp23;
    JacobianRotate3 = temp23;

    return Pose( t, Q);
}
