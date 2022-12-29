#ifndef POSELOCALPARAMETER_H
#define POSELOCALPARAMETER_H


#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "PoseSpline/Pose.hpp"


/*
 * PoseLocalParameter
 * Hers we define the pose representaion: first translate , then rotation quaternion
 */
class PoseLocalParameter: public  ceres::LocalParameterization{


public:
    virtual ~PoseLocalParameter() {};
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
        return plus(x, delta, x_plus_delta);
    }
    virtual bool ComputeJacobian(const double *x, double *jacobian) const{
        plusJacobian(x,jacobian);
        return true;
    }
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };

    static bool plus(const double* x, const double* delta, double* x_plus_delta) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1> > delta_(delta);

        // transform to okvis::kinematics framework
        Pose<double> T(Eigen::Vector3d(x[0], x[1], x[2]),
                       Quaternion ( x[3], x[4], x[5], x[6]));

        // call oplus operator in okvis::kinematis
        T.oplus(delta_);

        // copy back
        const Eigen::Vector3d r = T.r();
        x_plus_delta[0] = r[0];
        x_plus_delta[1] = r[1];
        x_plus_delta[2] = r[2];
        const Eigen::Vector4d q = T.q();
        x_plus_delta[3] = q[0];
        x_plus_delta[4] = q[1];
        x_plus_delta[5] = q[2];
        x_plus_delta[6] = q[3];

        //OKVIS_ASSERT_TRUE_DBG(std::runtime_error, T.q().norm()-1.0<1e-15, "damn.");

        return true;
    }


    // Extent interface
    bool ComputeLiftJacobian(const double* x, double* jacobian) const  {
        liftJacobian(x,jacobian);
        return true;

    }
    static bool liftJacobian(const double* x,double* jacobian) {
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);

        J_lift.setIdentity();
        Quaternion q_inv(-x[3],-x[4],-x[5],x[6]);
        Eigen::Matrix<double, 3, 4> Jq_pinv;
        Jq_pinv.setZero();
        Jq_pinv.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() * 2.0;
        J_lift.bottomRightCorner(3,4) = Jq_pinv*quatRightComp(q_inv);

        return true;
    }
    static bool plusJacobian(const double* x,double* jacobian) {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        Eigen::Map<const Quaternion> Q_(x+3);

        Eigen::Matrix<double, 4, 3> m;
        m.setZero();
        m.topLeftCorner(3,3).setIdentity();
        J.bottomRightCorner(4,3) = quatRightComp<double>(Q_)*0.5*m;
        return true;
    }
};


#endif