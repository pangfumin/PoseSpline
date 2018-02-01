#ifndef QUATERNIONSPLINEUTILITY_H
#define QUATERNIONSPLINEUTILITY_H


#include "pose-spline/Quaternion.hpp"

typedef  Eigen::Matrix3d Jacobian_Quat;


/*
 * Continuous-Time Estimation of attitude
 * using B-splines on Lie groups
 */


class QSUtility{

public:

    static Eigen::Vector3d Phi(const Quaternion & Q_k_1, const Quaternion &Q_k);
    static Quaternion r(double beta_t, Eigen::Vector3d Phi);
    static std::pair<Jacobian_Quat,Jacobian_Quat>
                    Jcobian_Phi_Quat(Quaternion &q_k_1, Quaternion &q_k);


    /*
     *  bpline basis fuctions
     */
    static double cubicBasisFun0(double u){
        return (1 - u)*(1 - u)*(1 - u)/6.0;
    }

    static double cubicBasisFun1(double u){
        return (3*u*u*u - 6*u*u + 4)/6.0;
    }

    static double cubicBasisFun2(double u){
        return (-3*u*u*u + 3*u*u + 3*u +1)/6.0;
    }

    static double cubicBasisFun3(double u){
        return u*u*u /6.0;
    }

    // An alternative form
    static Eigen::Matrix4d M(){
        return (1.0/6.0)*(Eigen::Matrix4d()<<1,4,1,0,
                                           -3,0,3,0,
                                           3,-6,3,0,
                                           -1,3,-3,1).finished();
    }

    /*
     * bspline basis Cumulative functions
     */
    // todo: need move to base class
    static double beta0(double u){
        return 1.0;
    }
    static double beta1(double u){

        //cubicBasisFun1(u) + cubicBasisFun2(u) + cubicBasisFun3(u)
        return (u*u*u - 3*u*u + 3*u +5)/6.0;
    }
    static double beta2(double u){
        // cubicBasisFun2(u) + cubicBasisFun3(u)
        return (-2*u*u*u + 3*u*u + 3*u +1)/6.0;;
    }
    static double beta3(double u){
        return  cubicBasisFun3(u);
    }


    // An anternative matrix form
    // A Spline-Based Trajectory Representation for Sensor Fusion
    // and Rolling Shutter Cameras
    // But in form: beta = C*[1 u u^2 u^3]^T

    static Eigen::Matrix4d C(){

        Eigen::Matrix4d res;
        res <<  6,5,1,0,
                0,3,3,0,
                0,-3,3,0,
                0,1,-2,1;
        res = (1.0/6.0)*res;


        return res;
    }
    /*
     * First order derivation of basis Cumulative functions
     * Here, we take dt into consideration.
     */

    //TODO: simplify
    static double dot_beta1(const double dt, const double u){

        Eigen::Vector4d uu(0.0, 1, 2*u ,3*u*u);
        return (1/dt)*uu.transpose()*C().col(1);
    }

    static double dot_beta2(const double dt, const double u){

        Eigen::Vector4d uu(0.0, 1, 2*u ,3*u*u);
        return (1/dt)*uu.transpose()*C().col(2);
    }


    static double dot_beta3(const double dt, const double u){

        Eigen::Vector4d uu(0.0, 1, 2*u ,3*u*u);
        return (1/dt)*uu.transpose()*C().col(3);
    }

    /*
     * Second order derivation
     */
    static double dot_dot_beta1(double dt, double u){

        Eigen::Vector4d uu(0.0, 0.0, 2.0 ,6*u);
        return (1/(dt*dt))*uu.transpose()*C().col(1);
    }

    static double dot_dot_beta2(double dt, double u){

        Eigen::Vector4d uu(0.0, 0.0, 2.0 ,6*u);
        return (1/(dt*dt))*uu.transpose()*C().col(2);
    }


    static double dot_dot_beta3(double dt, double u){

        Eigen::Vector4d uu(0.0, 0.0, 2.0 ,6*u);
        return (1/(dt*dt))*uu.transpose()*C().col(3);
    }

/*
 * We follow Pose estimation using linearized rotations and quaternion algebra.
 * The exp and log function is a little different from Hannes Sommer's
 * Continuous-Time Estimation paper.
 * Thus, the dr_dt and d2r_dt2 are not equal to ones in this paper.
 */
    static Quaternion dr_dt(double dot_beta, double beta,
                            const Quaternion & Q_k_1, const Quaternion &Q_k){

        Eigen::Vector4d phi_ext;
        Eigen::Vector3d phi = Phi(Q_k_1,Q_k);

        phi_ext << phi,0.0;

        return 0.5*dot_beta*quatLeftComp(phi_ext)*r(beta,phi);
    }

    static Quaternion dr_dt(double dot_beta, double beta,const Eigen::Vector3d& phi){

        Eigen::Vector4d phi_ext;
        phi_ext << phi,0.0;

        return 0.5*dot_beta*quatLeftComp(phi_ext)*r(beta,phi);
    }


    static Quaternion d2r_dt2(double dot_dot_beta, double dot_beta, double beta,
                              const Quaternion & Q_k_1, const Quaternion &Q_k){

        Eigen::Vector4d phi_ext;
        Eigen::Vector3d phi = Phi(Q_k_1,Q_k);

        phi_ext << phi,0.0;

        return 0.5*quatLeftComp<double>(
                (0.5*dot_beta*dot_beta*phi_ext + dot_dot_beta*unitQuat<double>()))
               *quatLeftComp(phi_ext)*r(beta,phi);
    }

    static Quaternion d2r_dt2(double dot_dot_beta, double dot_beta, double beta,
                              const Eigen::Vector3d& phi){

        Eigen::Vector4d phi_ext;
        phi_ext << phi,0.0;

        return 0.5*quatLeftComp<double>(
                (0.5*dot_beta*dot_beta*phi_ext + dot_dot_beta*unitQuat<double>()))
               *quatLeftComp(phi_ext)*r(beta,phi);
    }

    static Quaternion EvaluateQS(double u,
                                 const Quaternion& Q0,
                                 const Quaternion& Q1,
                                 const Quaternion& Q2,
                                 const Quaternion& Q3);


    static Quaternion Evaluate_dot_QS(double dt,
                                      double u,
                                      const Quaternion& Q0,
                                      const Quaternion& Q1,
                                      const Quaternion& Q2,
                                      const Quaternion& Q3);

    static Quaternion Evaluate_dot_dot_QS(double dt,
                                      double u,
                                      const Quaternion& Q0,
                                      const Quaternion& Q1,
                                      const Quaternion& Q2,
                                      const Quaternion& Q3);

    static Eigen::Matrix<double,4,3> V();
    static Eigen::Matrix<double,3,4> W();
    static Eigen::Vector3d w(Quaternion Q_ba, Quaternion dot_Q_ba);
    static Eigen::Vector3d alpha(Quaternion Q_ba, Quaternion dot_dot_Q_ba);

    static Quaternion Jacobian_dotQinvQ_t(const Quaternion& Q,
                                         const Quaternion& dQ,
                                         const Quaternion& ddQ);

    static Eigen::Vector3d Jacobian_omega_t(const Quaternion& Q,
                                    const Quaternion& dQ,
                                    const Quaternion& ddQ,
                                    const Quaternion& extrinsicQ);

    static Eigen::Matrix3d Jacobian_omega_extrinsicQ(const Quaternion& Q,
                                             const Quaternion& dQ,
                                             const Quaternion& extrinsicQ);
    static Eigen::Matrix<double,4,3> Jac_Exp(Eigen::Vector3d phi);

};

#endif