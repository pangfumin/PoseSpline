#include "pose-spline/QuaternionSplineUtility.hpp"


Eigen::Vector3d QSUtility::Phi(const Quaternion & Q_k_1, const Quaternion &Q_k){
    Quaternion invQ_k_1 = quatInv(Q_k_1);
    Quaternion tmp  = quatMult(invQ_k_1,Q_k);

    return quatLog(tmp);
}
Quaternion QSUtility::r(double beta_t, Eigen::Vector3d Phi){

    return quatExp<double>(beta_t*Phi);

}

std::pair<Jacobian_Quat,Jacobian_Quat> QSUtility::Jcobian_Phi_Quat(Quaternion &q_k_1, Quaternion &q_k){
    Eigen::Matrix3d L,C,J_k_1,J_k;
    Quaternion invQ = quatInv(q_k_1);
    Quaternion tmp = (quatLeftComp(invQ)*q_k);

    L = quatL<double>(tmp);
    C = quatToRotMat(q_k_1);

    J_k_1 = -L*C;
    J_k = L*C;

    return std::make_pair(J_k_1,J_k);
}

Quaternion QSUtility::EvaluateQS(double u,
                                const Quaternion& Q0,
                                const Quaternion& Q1,
                                const Quaternion& Q2,
                                const Quaternion& Q3){

    Eigen::Vector3d Phi1 = Phi(Q0,Q1);
    Eigen::Vector3d Phi2 = Phi(Q1,Q2);
    Eigen::Vector3d Phi3 = Phi(Q2,Q3);

    Quaternion r1 = r(beta1(u),Phi1);
    Quaternion r2 = r(beta2(u),Phi2);
    Quaternion r3 = r(beta3(u),Phi3);

    return quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;

}


Quaternion QSUtility::Evaluate_dot_QS(double dt,
                          double u,
                          const Quaternion& Q0,
                          const Quaternion& Q1,
                          const Quaternion& Q2,
                          const Quaternion& Q3){

    Eigen::Vector3d Phi1 = Phi(Q0,Q1);
    Eigen::Vector3d Phi2 = Phi(Q1,Q2);
    Eigen::Vector3d Phi3 = Phi(Q2,Q3);


    double b1 = beta1(u);
    double b2 = beta2(u);
    double b3 = beta3(u);

    double dot_b1 = dot_beta1(dt,u);
    double dot_b2 = dot_beta2(dt,u);
    double dot_b3 = dot_beta3(dt,u);


    Quaternion r1 = r(b1,Phi1);
    Quaternion r2 = r(b2,Phi2);
    Quaternion r3 = r(b3,Phi3);

    Quaternion dot_q, part1, part2,part3;
    part1 = quatLeftComp(dr_dt(dot_b1,b1,Q0,Q1))*quatLeftComp(r2)*r3;
    part2 = quatLeftComp(r1)*quatLeftComp(dr_dt(dot_b2,b2,Q1,Q2))*r3;
    part3 = quatLeftComp(r1)*quatLeftComp(r2)*dr_dt(dot_b3,b3,Q2,Q3);

    dot_q = quatLeftComp(Q0)*(part1 + part2 + part3);

    return dot_q;

}



Quaternion QSUtility::Evaluate_dot_dot_QS(double dt,
                                      double u,
                                      const Quaternion& Q0,
                                      const Quaternion& Q1,
                                      const Quaternion& Q2,
                                      const Quaternion& Q3){

    Eigen::Vector3d Phi1 = Phi(Q0,Q1);
    Eigen::Vector3d Phi2 = Phi(Q1,Q2);
    Eigen::Vector3d Phi3 = Phi(Q2,Q3);


    double b1 = beta1(u);
    double b2 = beta2(u);
    double b3 = beta3(u);

    double dot_b1 = dot_beta1(dt,u);
    double dot_b2 = dot_beta2(dt,u);
    double dot_b3 = dot_beta3(dt,u);

    double dot_dot_b1 = dot_dot_beta1(dt,u);
    double dot_dot_b2 = dot_dot_beta2(dt,u);
    double dot_dot_b3 = dot_dot_beta3(dt,u);

    Quaternion  ddr1 = d2r_dt2(dot_dot_b1,dot_b1,b1,Phi1);
    Quaternion  ddr2 = d2r_dt2(dot_dot_b2,dot_b2,b2,Phi2);
    Quaternion  ddr3 = d2r_dt2(dot_dot_b3,dot_b3,b3,Phi3);

    Quaternion dr1 = dr_dt(dot_b1,b1,Phi1);
    Quaternion dr2 = dr_dt(dot_b2,b2,Phi2);
    Quaternion dr3 = dr_dt(dot_b3,b3,Phi3);



    Quaternion r1 = r(b1,Phi1);
    Quaternion r2 = r(b2,Phi2);
    Quaternion r3 = r(b3,Phi3);

    Quaternion dot_dot_q, part11, part12, part13, part21, part22, part23, part31, part32, part33;
    part11 = quatLeftComp(ddr1)*quatLeftComp(r2)*r3;
    part12 = quatLeftComp(dr1)*quatLeftComp(dr2)*r3;
    part13 = quatLeftComp(dr1)*quatLeftComp(r2)*dr3;

    part21 = quatLeftComp(dr1)*quatLeftComp(dr2)*r3;
    part22 = quatLeftComp(r1)*quatLeftComp(ddr2)*r3;
    part23 = quatLeftComp(r1)*quatLeftComp(dr2)*dr3;

    part31 = quatLeftComp(dr1)*quatLeftComp(r2)*dr3;
    part32 = quatLeftComp(r1)*quatLeftComp(dr2)*dr3;
    part33 = quatLeftComp(r1)*quatLeftComp(r2)*ddr3;


    dot_dot_q = quatLeftComp(Q0)*(part11 + part12 + part13
                                    + part21 + part22 + part23
                                    + part31 + part32 + part33);

    return dot_dot_q;

}


/*
 *
 */
Eigen::Matrix<double,4,3> QSUtility::V(){

    Eigen::Matrix<double,4,3> M;
    M<< 0.5,   0,   0,
          0, 0.5,   0,
          0,   0, 0.5,
          0,   0,   0;
    return M;
};


Eigen::Matrix<double,3,4> QSUtility::W(){

    Eigen::Matrix<double,3,4> M;
    M<< 1.0,   0,   0, 0,
            0, 1.0,   0, 0,
            0,   0, 1.0, 0;

    return M;
};

/*
 *  using Indrict Kalman filter for 3D attitude estimation.
 */
Eigen::Vector3d QSUtility::w(Quaternion Q_ba, Quaternion dot_Q_ba){
    return 2.0*(quatLeftComp(dot_Q_ba)*quatInv(Q_ba)).head(3);
}

Eigen::Vector3d QSUtility::alpha(Quaternion Q_ba, Quaternion dot_dot_Q_ba){
    return 2.0*(quatLeftComp(dot_dot_Q_ba)*quatInv(Q_ba)).head(3);
}

/*
 *
 */

//Todo: test
Quaternion QSUtility::Jacobian_dotQinvQ_t(const Quaternion& Q,
                              const Quaternion& dQ,
                              const Quaternion& ddQ){

    Quaternion invQ, A, B0, B1;
    invQ = quatInv(Q);
    A = quatLeftComp(dQ)*invQ;
    B0<< (quatLeftComp(ddQ)*invQ).head(3),0.0;
    B1<< (quatLeftComp(dQ)*invQ).head(3),0.0;
    return quatRightComp(A)*B0 - quatLeftComp(A)*B1;

}
//Todo: test
Eigen::Vector3d QSUtility::Jacobian_omega_t(const Quaternion& Q,
                                            const Quaternion& dQ,
                                            const Quaternion& ddQ,
                                            const Quaternion& extrinsicQ){

    Quaternion invQ, A, B0, B1, B;
    invQ = quatInv(Q);
    A = quatLeftComp(dQ)*invQ;
    B0<< (quatLeftComp(ddQ)*invQ).head(3),0.0;
    B1<< (quatLeftComp(dQ)*invQ).head(3),0.0;

    B = quatRightComp(A)*B0 - quatLeftComp(A)*B1;
    Eigen::Vector4d J = quatLeftComp(quatInv(extrinsicQ))*quatRightComp(extrinsicQ)*B;

    return J.head(3);


}
//Todo: test
Eigen::Matrix3d QSUtility::Jacobian_omega_extrinsicQ(const Quaternion& Q,
                                                     const Quaternion& dQ,
                                                     const Quaternion& extrinsicQ){

    Quaternion invQ, A, invExtQ;
    invQ = quatInv(Q);
    A = quatLeftComp(dQ)*invQ;
    invExtQ = quatInv(extrinsicQ);
    Eigen::Matrix4d J;
    J = quatLeftComp(invExtQ)*quatLeftComp(A)*quatRightComp(extrinsicQ)
            - quatLeftComp(invExtQ)*quatRightComp<double>(quatLeftComp(A)*extrinsicQ);
    return J.topLeftCorner(3,3);

}

Eigen::Matrix<double,4,3> QSUtility::Jac_Exp(Eigen::Vector3d phi){
    return quatRightComp(quatExp(phi))*V()*quatS(phi);
};

