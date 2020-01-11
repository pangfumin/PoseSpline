
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
template <typename T>
__inline__ T sinc(T x) {
    if (fabs(x) > 1e-6) {
        return sin(x) / x;
    } else {
        static const T c_2 = T(1.0 / 6.0);
        static const T c_4 = T(1.0 / 120.0);
        static const T c_6 = T(1.0 / 5040.0);
        const T x_2 = x * x;
        const T x_4 = x_2 * x_2;
        const T x_6 = x_2 * x_2 * x_2;
        return T(1.0) - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
    }
}
template <typename T>
__inline__ Eigen::Matrix<T,4,1> deltaQ(const Eigen::Matrix<T,3,1>& dAlpha)
{
    Eigen::Matrix<T,4,1> dq;
    T halfnorm = T(0.5) * dAlpha.template tail<3>().norm();
    dq.template head<3>() = sinc(halfnorm) * 0.5 * dAlpha.template tail<3>();
    dq[3] = cos(halfnorm);
    return dq;
}

// Right Jacobian, see Forster et al. RSS 2015 eqn. (8)
template <typename T>
__inline__ Eigen::Matrix<T,3,3> rightJacobian(const Eigen::Matrix<T,3,1> & PhiVec) {
    const T Phi = PhiVec.norm();
    Eigen::Matrix<T,3,3> retMat = Eigen::Matrix<T,3,3>::Identity();
    const  Eigen::Matrix<T,3,3> Phi_x = crossMat(PhiVec);
    const  Eigen::Matrix<T,3,3> Phi_x2 = Phi_x*Phi_x;
    if(Phi < T(1.0e-4)) {
        retMat += T(-0.5)*Phi_x + T(1.0/6.0)*Phi_x2;
    } else {
        const T Phi2 = Phi*Phi;
        const T Phi3 = Phi2*Phi;
        retMat += -(T(1.0)-cos(Phi))/(Phi2)*Phi_x + (Phi-sin(Phi))/Phi3*Phi_x2;
    }
    return retMat;
}

template <typename T>
inline Pose<T>::Pose(const Pose & other)
        : parameters_(other.parameters_),
          r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(other.C_) {

}
template <typename T>
inline Pose<T>::Pose(Pose && other)
        : parameters_(std::move(other.parameters_)),
          r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(std::move(other.C_)) {

}

template <typename T>
inline Pose<T>::Pose()
        : r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(Eigen::Matrix<T,3,3>::Identity()) {
    r_ = Eigen::Matrix<T,3,1>(T(0.0), T(0.0), T(0.0));
    q_ = unitQuat<T>();
}

template <typename T>
inline Pose<T>::Pose(const Eigen::Matrix<T,3,1> & r_AB,
                                      const Eigen::Matrix<T,4,1>& q_AB)
        : r_(&parameters_[0]),
          q_(&parameters_[3]) {
    r_ = r_AB;
    q_ = q_AB.normalized();
    updateC();
}
template <typename T>
inline Pose<T>::Pose(const Eigen::Matrix<T,3,1> & r_AB, const Eigen::Quaternion<T>& q_AB) 
        : r_(&parameters_[0]),
          q_(&parameters_[3]) {
    r_ = r_AB;
    q_ = q_AB.coeffs();
    updateC();
}

template <typename T>
inline Pose<T>::Pose(const Eigen::Matrix<T,7,1>& vec): r_(&parameters_[0]),
                                                         q_(&parameters_[3]) {
    r_ = vec.template head<3>();
    q_ = vec.template tail<4>();
    q_ = quatNorm<T>(q_);
    updateC();

}
template <typename T>
inline Pose<T>::Pose(const T* array_ptr): r_(&parameters_[0]),
                                                       q_(&parameters_[3]) {
    r_ << array_ptr[0], array_ptr[1], array_ptr[2];
    q_ << array_ptr[3], array_ptr[4], array_ptr[5], array_ptr[6];
    q_ = quatNorm<T>(q_);
    updateC();

}
template <typename T>
inline Pose<T>::Pose(const Eigen::Matrix<T,4,4> & T_AB)
        : r_(&parameters_[0]),
          q_(&parameters_[3]),
          C_(T_AB.template topLeftCorner<3, 3>()) {
    r_ = (T_AB.template topRightCorner<3, 1>());
    q_ = rotMatToQuat<T>(C_);
//    assert(fabs(T_AB(3, 0)) < 1.0e-12);
//    assert(fabs(T_AB(3, 1)) < 1.0e-12);
//    assert(fabs(T_AB(3, 2)) < 1.0e-12);
//    assert(fabs(T_AB(3, 3) - 1.0) < 1.0e-12);
}
template <typename T>
inline Pose<T>::~Pose() {

}
template<typename T>
template<typename Derived_coeffs>
inline bool Pose<T>::setCoeffs(
        const Eigen::MatrixBase<Derived_coeffs> & coeffs) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_coeffs, 7);
    parameters_ = coeffs;
    updateC();
    return true;
}

// The underlying transformation
template <typename T>
inline Eigen::Matrix<T,4,4> Pose<T>::Transformation() const {
    Eigen::Matrix<T,4,4> T_ret;
    T_ret.template topLeftCorner<3, 3>() = C_;
    T_ret.template topRightCorner<3, 1>() = r_;
    T_ret.template bottomLeftCorner<1, 3>().setZero();
    T_ret(3, 3) = T(1.0);
    return T_ret;
}

// return the rotation matrix
template <typename T>
inline const Eigen::Matrix<T,3,3> & Pose<T>::C() const {
    return C_;
}

// return the translation vector
template <typename T>
inline const Eigen::Map<Eigen::Matrix<T,3,1>> & Pose<T>::r() const {
    return r_;
}
template <typename T>
inline const Eigen::Map<Eigen::Matrix<T,4,1>> & Pose<T>::q() const {
    return q_;
}

template <typename T>
inline Eigen::Matrix<T, 3, 4> Pose<T>::T3x4() const {
    Eigen::Matrix<T, 3, 4> T3x4_ret;
    T3x4_ret.template topLeftCorner<3, 3>() = C_;
    T3x4_ret.template topRightCorner<3, 1>() = r_;
    return T3x4_ret;
}
// Return a copy of the transformation inverted.
template <typename T>
inline Pose<T> Pose<T>::inverse() const {
    return Pose<T>(-(C_.transpose() * r_), quatInv<T>(q_));
}

// Set this to a random transformation.
template <typename T>
inline void Pose<T>::setRandom() {
    setRandom(T(1.0), T(M_PI));
}
// Set this to a random transformation with bounded rotation and translation.
template <typename T>
inline void Pose<T>::setRandom(T translationMaxMeters,
                                      T rotationMaxRadians) {
    // Create a random unit-length axis.
    Eigen::Matrix<T,3,1> axis = rotationMaxRadians * Eigen::Matrix<T,3,1>::Random();
    // Create a random rotation angle in radians.
    Eigen::Matrix<T,3,1> r = translationMaxMeters * Eigen::Matrix<T,3,1>::Random();
    r_ = r;
    C_ = Eigen::AngleAxis<T>(axis.norm(), axis.normalized());
    q_ = rotMatToQuat<T>(C_);
}

// Setters
template <typename T>
inline void Pose<T>::set(const Eigen::Matrix<T,4,4> & T_AB) {
    r_ = (T_AB.template topRightCorner<3, 1>());
    C_ = (T_AB.template topLeftCorner<3, 3>());
    q_ = rotMatToQuat(C_);
}
template <typename T>

inline void Pose<T>::set(const Eigen::Matrix<T,3,1> & r_AB,
                                const Eigen::Matrix<T,4,1>& q_AB) {
    r_ = r_AB;
    q_ = quatNorm<T>(q_AB);

    updateC();
}
// Set this transformation to identity
template <typename T>
inline void Pose<T>::setIdentity() {
    q_ = unitQuat<T>();
    r_.setZero();
    C_.setIdentity();
}

template <typename T>
inline Pose<T> Pose<T>::Identity() {
    return Pose<T>();
}

// operator*
template <typename T>
inline Pose<T> Pose<T>::operator*(
        const Pose & rhs) const {
    return Pose(C_ * rhs.r_ + r_, quatLeftComp<T>(q_) * rhs.q_);
}
template <typename T>
inline Eigen::Matrix<T,3,1> Pose<T>::operator*(
        const Eigen::Matrix<T,3,1> & rhs) const {
    return C_ * rhs + r_;
}
template <typename T>
inline Eigen::Matrix<T,4,1> Pose<T>::operator*(
        const Eigen::Matrix<T,4,1> & rhs) const {
    const T s = rhs[3];
    Eigen::Matrix<T,4,1> retVec;
    retVec.template head<3>() = C_ * rhs.template head<3>() + r_ * s;
    retVec[3] = s;
    return retVec;
}

template <typename T>
inline Pose<T>& Pose<T>::operator=(const Pose<T> & rhs) {
    parameters_ = rhs.parameters_;
    C_ = rhs.C_;
    r_ = Eigen::Map<Eigen::Matrix<T,3,1>>(&parameters_[0]);
    q_ = Eigen::Map<Eigen::Matrix<T,4,1>>(&parameters_[3]);
    return *this;
}


template <typename T>
inline void Pose<T>::updateC() {
    C_ = quatToRotMat<T>(q_);
}


// apply small update:
template <typename T>
template<typename Derived_delta>
inline bool Pose<T>::oplus(
        const Eigen::MatrixBase<Derived_delta> & delta) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
    r_ += delta.template head<3>();
    Eigen::Matrix<T,4,1> dq;
    T halfnorm = T(0.5) * delta.template tail<3>().norm();
    dq.template head<3>() = sinc(halfnorm) * T(0.5) * delta.template tail<3>();
    dq[3] = cos(halfnorm);
    q_ = (quatLeftComp(dq) * q_);
    q_.normalize();
    updateC();
    return true;
}

template <typename T>
template<typename Derived_delta, typename Derived_jacobian>
inline bool Pose<T>::oplus(
        const Eigen::MatrixBase<Derived_delta> & delta,
        const Eigen::MatrixBase<Derived_jacobian> & jacobian) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
    if (!oplus(delta)) {
        return false;
    }
    return oplusJacobian(jacobian);
}
template <typename T>
template<typename Derived_jacobian>
inline bool Pose<T>::oplusJacobian(
        const Eigen::MatrixBase<Derived_jacobian> & jacobian) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
    Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
            .template topLeftCorner<3, 3>().setIdentity();
    S(0, 0) = T(0.5);
    S(1, 1) = T(0.5);
    S(2, 2) = T(0.5);
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
            .template bottomRightCorner<4, 3>() = quatRightComp<double>(q_) * S;
    return true;
}

//use right multiplication to compute $\frac{d\alpha}{d\Delta q}$ where \alpha is defined in leutenegger
// ijrr 15, and q+\Delta q = \delta q(\alpha)\otimes q,
// so (q+\Delta q)\otimes q^{-1} = \delta q(\alpha) = [0.5\alpha; 1]^T
template <typename T>
template <typename Derived_jacobian>
inline bool Pose<T>::liftJacobian(const Eigen::MatrixBase<Derived_jacobian> & jacobian) const
{
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 6, 7);
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template topLeftCorner<3,3>()
            = Eigen::Matrix3d::Identity();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template bottomRightCorner<3,4>()
            = 2*quatRightComp<T>(quatInv<T>(q_)).template topLeftCorner<3,4>();
    return true;
}
template <typename T>
inline  Eigen::Matrix<T,4,1> Pose<T>::rotation() const {
    return q_;
}
template <typename T>

inline  Eigen::Matrix<T,3,1> Pose<T>::translation() const {
    return r_;
}

