#include "pose-spline/QuaternionLocalParameter.hpp"
#include "geometry/Quaternion.hpp"

//Instances of LocalParameterization implement the ⊞ operation.
bool QuaternionLocalParameter::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Quaternion> Q_(x); // Q bar
    Eigen::Map<const Eigen::Vector3d> delta_phi(delta);

    Quaternion dq;
    dq<< 0.5*delta_phi, 1.0;

    Eigen::Map<Quaternion> Q_plus(x_plus_delta);
    Q_plus = quatLeftComp(dq)*Q_;

    Q_plus = Q_plus/Q_plus.norm(); // normilize

    return true;
}

//And its derivative with respect to Δx at Δx=0.  // r.f furgale, barfoot and okvis
bool QuaternionLocalParameter::ComputeJacobian(const double *x, double *jacobian) const
{
    plusJacobian(x,jacobian);
    return true;
}

bool QuaternionLocalParameter::plusJacobian(const double* x,double* jacobian) {

    Eigen::Map<const Quaternion> Q_(x);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
    Eigen::Matrix<double, 4, 3> m;
    m.setZero();
    m.topLeftCorner(3,3).setIdentity();
    J = quatRightComp<double>(Q_)*0.5*m;
    return true;

}


/*
 * Note: liftJacobian is [MiniDim x Dim], plusJacobian is [Dim x MiniDim],
 *       liftJacobian x plusJacobian = I_MiniDim
 */

bool QuaternionLocalParameter::ComputeLiftJacobian(const double* x, double* jacobian) const {

    liftJacobian(x,jacobian);
    return true;

}


bool QuaternionLocalParameter::liftJacobian(const double* x, double* jacobian) {

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > J_lift(jacobian);

    Quaternion q_inv(-x[0],-x[1],-x[2],x[3]);
    Eigen::Matrix<double, 3, 4> Jq_pinv;
    Jq_pinv.setZero();
    Jq_pinv.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() * 2.0;
    J_lift = Jq_pinv*quatRightComp(q_inv);

    return true;
}

bool QuaternionLocalParameter::VerifyJacobianNumDiff(const double* x,
                                                      double* jacobian,
                                                      double* jacobianNumDiff) {
    ComputeJacobian(x, jacobian);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor> > Jp(jacobian);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor> > Jpn(
            jacobianNumDiff);
    double dx = 1e-9;
    Eigen::Matrix<double, 4, 1> xp;
    Eigen::Matrix<double, 4, 1> xm;
    for (size_t i = 0; i < 3; ++i) {
        Eigen::Matrix<double, 3, 1> delta;
        delta.setZero();
        delta[i] = dx;
        Plus(x, delta.data(), xp.data());
        delta[i] = -dx;
        Plus(x, delta.data(), xm.data());
        Jpn.col(i) = (xp - xm) / (2 * dx);
    }
    if ((Jp - Jpn).norm() < 1e-6)
        return true;
    else
        return false;
}

