
/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Dec 2, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file kinematics/Transformation.hpp
 * @brief Header file for the Transformation class.
 * @author Stefan Leutenegger
 */

/**
 * @file Pose.hpp
 * @brief Header file for the Pose class.
 * @author Modified By Pang Fumin
 */

__inline__ double sinc(double x) {
    if (fabs(x) > 1e-6) {
        return sin(x) / x;
    } else {
        static const double c_2 = 1.0 / 6.0;
        static const double c_4 = 1.0 / 120.0;
        static const double c_6 = 1.0 / 5040.0;
        const double x_2 = x * x;
        const double x_4 = x_2 * x_2;
        const double x_6 = x_2 * x_2 * x_2;
        return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
    }
}

__inline__ Quaternion deltaQ(const Eigen::Vector3d& dAlpha)
{
    Quaternion dq;
    double halfnorm = 0.5 * dAlpha.template tail<3>().norm();
    dq.template head<3>() = sinc(halfnorm) * 0.5 * dAlpha.template tail<3>();
    dq[3] = cos(halfnorm);
    return dq;
}

// Right Jacobian, see Forster et al. RSS 2015 eqn. (8)
__inline__ Eigen::Matrix3d rightJacobian(const Eigen::Vector3d & PhiVec) {
    const double Phi = PhiVec.norm();
    Eigen::Matrix3d retMat = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Phi_x = crossMat(PhiVec);
    const Eigen::Matrix3d Phi_x2 = Phi_x*Phi_x;
    if(Phi < 1.0e-4) {
        retMat += -0.5*Phi_x + 1.0/6.0*Phi_x2;
    } else {
        const double Phi2 = Phi*Phi;
        const double Phi3 = Phi2*Phi;
        retMat += -(1.0-cos(Phi))/(Phi2)*Phi_x + (Phi-sin(Phi))/Phi3*Phi_x2;
    }
    return retMat;
}


inline Pose::Pose(const Pose & other)
        : parameters_(other.parameters_),
          r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(other.C_) {

}

inline Pose::Pose(Pose && other)
        : parameters_(std::move(other.parameters_)),
          r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(std::move(other.C_)) {

}

inline Pose::Pose()
        : r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(Eigen::Matrix3d::Identity()) {
    r_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    q_ = unitQuat<double>();
}

inline Pose::Pose(const Eigen::Vector3d & r_AB,
                                      const Quaternion& q_AB)
        : r_(&parameters_[0]),
          q_(&parameters_[3]) {
    r_ = r_AB;
    q_ = q_AB.normalized();
    updateC();
}

inline Pose::Pose(const Eigen::Matrix<double,7,1>& vec): r_(&parameters_[0]),
                                                         q_(&parameters_[3]) {
    r_ = vec.head<3>();
    q_ = vec.tail<4>();
    q_ = quatNorm<double>(q_);
    updateC();

}
inline Pose::Pose(const Eigen::Matrix4d & T_AB)
        : r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(T_AB.topLeftCorner<3, 3>()) {
    r_ = (T_AB.topRightCorner<3, 1>());
    q_ = rotMatToQuat<double>(C_);
    assert(fabs(T_AB(3, 0)) < 1.0e-12);
    assert(fabs(T_AB(3, 1)) < 1.0e-12);
    assert(fabs(T_AB(3, 2)) < 1.0e-12);
    assert(fabs(T_AB(3, 3) - 1.0) < 1.0e-12);
}
inline Pose::~Pose() {

}

template<typename Derived_coeffs>
inline bool Pose::setCoeffs(
        const Eigen::MatrixBase<Derived_coeffs> & coeffs) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_coeffs, 7);
    parameters_ = coeffs;
    updateC();
    return true;
}

// The underlying transformation
inline Eigen::Matrix4d Pose::T() const {
    Eigen::Matrix4d T_ret;
    T_ret.topLeftCorner<3, 3>() = C_;
    T_ret.topRightCorner<3, 1>() = r_;
    T_ret.bottomLeftCorner<1, 3>().setZero();
    T_ret(3, 3) = 1.0;
    return T_ret;
}

// return the rotation matrix
inline const Eigen::Matrix3d & Pose::C() const {
    return C_;
}

// return the translation vector
inline const Eigen::Map<Eigen::Vector3d> & Pose::r() const {
    return r_;
}

inline const Eigen::Map<Quaternion> & Pose::q() const {
    return q_;
}

inline Eigen::Matrix<double, 3, 4> Pose::T3x4() const {
    Eigen::Matrix<double, 3, 4> T3x4_ret;
    T3x4_ret.topLeftCorner<3, 3>() = C_;
    T3x4_ret.topRightCorner<3, 1>() = r_;
    return T3x4_ret;
}
// Return a copy of the transformation inverted.
inline Pose Pose::inverse() const {
    return Pose(-(C_.transpose() * r_), quatInv<double>(q_));
}

// Set this to a random transformation.
inline void Pose::setRandom() {
    setRandom(1.0, M_PI);
}
// Set this to a random transformation with bounded rotation and translation.
inline void Pose::setRandom(double translationMaxMeters,
                                      double rotationMaxRadians) {
    // Create a random unit-length axis.
    Eigen::Vector3d axis = rotationMaxRadians * Eigen::Vector3d::Random();
    // Create a random rotation angle in radians.
    Eigen::Vector3d r = translationMaxMeters * Eigen::Vector3d::Random();
    r_ = r;
    C_ = Eigen::AngleAxisd(axis.norm(), axis.normalized());
    q_ = rotMatToQuat(C_);

}

// Setters
inline void Pose::set(const Eigen::Matrix4d & T_AB) {
    r_ = (T_AB.topRightCorner<3, 1>());
    C_ = (T_AB.topLeftCorner<3, 3>());
    q_ = rotMatToQuat(C_);
}
inline void Pose::set(const Eigen::Vector3d & r_AB,
                                const Quaternion& q_AB) {
    r_ = r_AB;
    q_ = quatNorm<double>(q_AB);

    updateC();
}
// Set this transformation to identity
inline void Pose::setIdentity() {
    q_ = unitQuat<double>();
    r_.setZero();
    C_.setIdentity();
}

inline Pose Pose::Identity() {
    return Pose();
}

// operator*
inline Pose Pose::operator*(
        const Pose & rhs) const {
    return Pose(C_ * rhs.r_ + r_, quatLeftComp<double>(q_) * rhs.q_);
}
inline Eigen::Vector3d Pose::operator*(
        const Eigen::Vector3d & rhs) const {
    return C_ * rhs;
}
inline Eigen::Vector4d Pose::operator*(
        const Eigen::Vector4d & rhs) const {
    const double s = rhs[3];
    Eigen::Vector4d retVec;
    retVec.head<3>() = C_ * rhs.head<3>() + r_ * s;
    retVec[3] = s;
    return retVec;
}

inline Pose& Pose::operator=(const Pose & rhs) {
    parameters_ = rhs.parameters_;
    C_ = rhs.C_;
    r_ = Eigen::Map<Eigen::Vector3d>(&parameters_[0]);
    q_ = Eigen::Map<Quaternion>(&parameters_[3]);
    return *this;
}

inline void Pose::updateC() {
    C_ = quatToRotMat<double>(q_);
}


// apply small update:
template<typename Derived_delta>
inline bool Pose::oplus(
        const Eigen::MatrixBase<Derived_delta> & delta) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
    r_ += delta.template head<3>();
    Quaternion dq;
    double halfnorm = 0.5 * delta.template tail<3>().norm();
    dq.template head<3>() = sinc(halfnorm) * 0.5 * delta.template tail<3>();
    dq[3] = cos(halfnorm);
    q_ = (quatLeftComp(dq) * q_);
    q_.normalize();
    updateC();
    return true;
}

template<typename Derived_delta, typename Derived_jacobian>
inline bool Pose::oplus(
        const Eigen::MatrixBase<Derived_delta> & delta,
        const Eigen::MatrixBase<Derived_jacobian> & jacobian) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
    if (!oplus(delta)) {
        return false;
    }
    return oplusJacobian(jacobian);
}

template<typename Derived_jacobian>
inline bool Pose::oplusJacobian(
        const Eigen::MatrixBase<Derived_jacobian> & jacobian) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
    Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
            .template topLeftCorner<3, 3>().setIdentity();
    S(0, 0) = 0.5;
    S(1, 1) = 0.5;
    S(2, 2) = 0.5;
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
            .template bottomRightCorner<4, 3>() = quatRightComp<double>(q_) * S;
    return true;
}

//use right multiplication to compute $\frac{d\alpha}{d\Delta q}$ where \alpha is defined in leutenegger
// ijrr 15, and q+\Delta q = \delta q(\alpha)\otimes q,
// so (q+\Delta q)\otimes q^{-1} = \delta q(\alpha) = [0.5\alpha; 1]^T
template <typename Derived_jacobian>
inline bool Pose::liftJacobian(const Eigen::MatrixBase<Derived_jacobian> & jacobian) const
{
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 6, 7);
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template topLeftCorner<3,3>()
            = Eigen::Matrix3d::Identity();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template bottomRightCorner<3,4>()
            = 2*quatRightComp<double>(quatInv<double>(q_)).template topLeftCorner<3,4>();
    return true;
}

inline  Quaternion Pose::rotation() const {
    return q_;
}

inline  Eigen::Vector3d Pose::translation() const {
    return r_;
}

