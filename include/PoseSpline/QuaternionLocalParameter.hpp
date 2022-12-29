#ifndef QUATERNIONLOCALPARAMETER_H
#define QUATERNIONLOCALPARAMETER_H


#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "Quaternion.hpp"
class QuaternionLocalParameter: public  ceres::LocalParameterization{


public:
    virtual ~QuaternionLocalParameter() {
    }

    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
        Eigen::Map<const Quaternion> Q_(x); // Q bar
        Eigen::Map<const Eigen::Vector3d> delta_phi(delta);

        Quaternion dq;
        dq << 0.5*delta_phi, 1.0;

        Eigen::Map<Quaternion> Q_plus(x_plus_delta);
        Q_plus = quatLeftComp(dq)*Q_;

        Q_plus = Q_plus/Q_plus.norm(); // normilize

        return true;
    }
    virtual bool ComputeJacobian(const double *x, double *jacobian) const {
        plusJacobian(x,jacobian);
        return true;
    }
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };


    static bool plusJacobian(const double* x,double* jacobian) {
        Eigen::Map<const Quaternion> Q_(x);
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);

        /*
        Eigen::Matrix<double, 4, 3> m;
        m.setZero();
        m.topLeftCorner(3,3).setIdentity();
        J = quatRightComp<double>(Q_)*0.5*m;
         */

        /// more effient
        J.setZero();
        J(0, 0) = Q_(3);
        J(0, 1) = -Q_(2);
        J(0, 2) = Q_(1);
        J(1, 0) = Q_(2);
        J(1, 1) = Q_(3);
        J(1, 2) = -Q_(0);
        J(2, 0) = -Q_(1);
        J(2, 1) = Q_(0);
        J(2, 2) = Q_(3);
        J(3, 0) = -Q_(0);
        J(3, 1) = -Q_(1);
        J(3, 2) = -Q_(2);
        J *= 0.5;
        return true;

    }
    template <typename  T>
    static bool liftJacobian(const T* x, T* jacobian) {

        Eigen::Map<Eigen::Matrix<T, 3, 4, Eigen::RowMajor> > J_lift(jacobian);

        QuaternionTemplate<T> q_inv(-x[0],-x[1],-x[2],x[3]);
        Eigen::Matrix<T, 3, 4> Jq_pinv;
        Jq_pinv.setZero();
        Jq_pinv.template topLeftCorner<3,3>() = Eigen::Matrix<T,3,3>::Identity() * T(2.0);
        J_lift = Jq_pinv*quatRightComp(q_inv);

        return true;
    }
    // Additional interface
    bool ComputeLiftJacobian(const double* x, double* jacobian) const {
        liftJacobian<double>(x,jacobian);
        return true;
    }

};


#endif