#include "pose-spline/PoseLocalParameter.hpp"
#include "pose-spline/Pose.hpp"

//Instances of LocalParameterization implement the ⊞ operation.
/*
 * Pose define
 * t(x,y,z)
 * q(x,y,z,w)
 */
bool PoseLocalParameter::Plus(const double *x, const double *delta, double *x_plus_delta) const
{

    return plus(x, delta, x_plus_delta);
}

bool PoseLocalParameter::plus(const double* x, const double* delta,
                                     double* x_plus_delta) {

    Eigen::Map<const Eigen::Matrix<double, 6, 1> > delta_(delta);

    // transform to okvis::kinematics framework
    Pose T(
            Eigen::Vector3d(x[0], x[1], x[2]),
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

//And its derivative with respect to Δx at Δx=0.  // r.f furgale, barfoot and okvis
/*
 * 7 x 6
 */
bool PoseLocalParameter::ComputeJacobian(const double *x, double *jacobian) const
{

    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    Eigen::Map<const Quaternion> Q_(x+3);

    Eigen::Matrix<double, 4, 3> m;
    m.setZero();
    m.topLeftCorner(3,3).setIdentity();
    J.bottomRightCorner(4,3) = quatRightComp<double>(Q_)*0.5*m;
    return true;
}

/*
 * Note: liftJacobian is [MiniDim x Dim], plusJacobian is [Dim x MiniDim],
 *       liftJacobian x plusJacobian = I_MiniDim
 */

bool PoseLocalParameter::ComputeLiftJacobian(const double* x, double* jacobian) const {

    liftJacobian(x,jacobian);
    return true;

}


bool PoseLocalParameter::liftJacobian(const double* x, double* jacobian) {

    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);

    J_lift.setIdentity();
    Quaternion q_inv(-x[3],-x[4],-x[5],x[6]);
    Eigen::Matrix<double, 3, 4> Jq_pinv;
    Jq_pinv.setZero();
    Jq_pinv.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() * 2.0;
    J_lift.bottomRightCorner(3,4) = Jq_pinv*quatRightComp(q_inv);

    return true;
}

bool PoseLocalParameter::VerifyJacobianNumDiff(const double* x,
                                                     double* jacobian,
                                                     double* jacobianNumDiff) {

    ComputeJacobian(x, jacobian);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jpn(
            jacobianNumDiff);
    double dx = 1e-9;
    Eigen::Matrix<double, 7, 1> xp;
    Eigen::Matrix<double, 7, 1> xm;
    for (size_t i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
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

        return false;
}

