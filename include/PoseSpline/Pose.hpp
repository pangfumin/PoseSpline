#ifndef POSE_H
#define POSE_H
#include "Quaternion.hpp"
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
 * @brief Header file for the Pose class. The main difference between original one is we use Quaternion.
 * @author Modified by Pang Fumin  on Sept 25, 2017
 */




/// \brief Implements sin(x)/x for all x in R.
/// @param[in] x The argument of the sinc function.
/// \return The result.
template <typename T>
T sinc(T x);

/// \brief Implements the exponential map for quaternions.
/// @param[in] dAlpha a axis*angle (minimal) input in tangent space.
/// \return The corresponding Quaternion.
template <typename T>
Eigen::Matrix<T,4,1> deltaQ(const Eigen::Matrix<T,3,1>& dAlpha);

/// \brief Right Jacobian, see Forster et al. RSS 2015 eqn. (8)
template <typename T>
Eigen::Matrix<T,3,3> rightJacobian(const  Eigen::Matrix<T,3,1> & PhiVec);

/// \brief A class that does homogeneous transformations.
/// This relates a frame A and B: T_AB; it consists of
///   translation r_AB (represented in frame A) and
///   Quaternion q_AB (as an Eigen Quaternion).
/// see also the RSS'13 / IJRR'14 paper or the Thesis.
/// Follows some of the functionality of the SchweizerMesser library by Paul Furgale,
/// but uses Eigen quaternions underneath.
/// \warning This means the convention is different to SchweizerMesser
///          and the RSS'13 / IJRR'14 paper / the Thesis

template <typename T>
class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Default constructor: initialises a unit transformation.
    Pose();

    /// \brief Copy constructor: nothing fancy but takes care
    ///        of not doing bad stuff with internal caching.
    Pose(const Pose & other);

    /// \brief Move constructor: nothing fancy but takes care
    ///        of not doing bad stuff with internal caching.
    Pose(Pose && other);

    /// \brief Construct from a translation and quaternion.
    /// @param[in] r_AB The translation r_AB (represented in frame A).
    /// @param[in] q_AB The Quaternion q_AB .
    Pose(const Eigen::Matrix<T,3,1> & r_AB, const Eigen::Matrix<T,4,1>& q_AB);

    Pose(const Eigen::Matrix<T,3,1> & r_AB, const Eigen::Quaternion<T>& q_AB);

    /// \brief Construct from a 7x1 vector.
    /// @param[in] vec The 7x1 vector
    Pose(const Eigen::Matrix<T,7,1>& vec);
    Pose(const T* array_ptr);


    /// \brief Construct from a homogeneous transformation matrix.
    /// @param[in] T_AB The homogeneous transformation matrix.
    explicit Pose(const Eigen::Matrix<T,4,4> & T_AB);

    /// \brief Trivial destructor.
    ~Pose();

    /// \brief Parameter setting, all 7.
    /// \tparam Derived_coeffs Deducible matrix type.
    /// @param[in] coeffs The parameters as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal convention).
    template<typename Derived_coeffs>
    bool setCoeffs(const Eigen::MatrixBase<Derived_coeffs> & coeffs);

    /// \brief Parameter setting, all 7.
    /// \tparam Derived_coeffs Deducible matrix type.
    /// @param[in] parameters The parameters as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal convention).
    template<typename Derived_coeffs>
    bool setParameters(const Eigen::MatrixBase<Derived_coeffs> & parameters)
    {
        return setCoeffs(parameters);
    }

    /// \brief The underlying homogeneous transformation matrix.
    Eigen::Matrix<T,4,4> Transformation() const;

    /// \brief Returns the rotation matrix (cached).
    const Eigen::Matrix<T,3,3> & C() const;

    /// \brief Returns the translation vector r_AB (represented in frame A).
    const Eigen::Map<Eigen::Matrix<T,3,1>> & r() const;

    /// \brief Returns the Quaternion q_AB .
    const Eigen::Map<Eigen::Matrix<T,4,1>> & q() const;

    /// \brief Get the upper 3x4 part of the homogeneous transformation matrix T_AB.
    Eigen::Matrix<T, 3, 4> T3x4() const;

    /// \brief The coefficients (parameters) as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal convention).
    const Eigen::Matrix<T, 7, 1> & coeffs() const
    {
        return parameters_;
    }

    /// \brief The parameters (coefficients) as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal convention).
    const Eigen::Matrix<T, 7, 1> & parameters() const
    {
        return parameters_;
    }

    /// \brief Get the parameters --- support for ceres.
    /// \warning USE WITH CARE!
    T* parameterPtr()
    {
        return &parameters_[0];
    }
    T* data()
    {
        return &parameters_[0];
    }

    /// \brief Set this to a random transformation.
    void setRandom();
    /// \brief Set this to a random transformation with bounded rotation and translation.
    /// @param[in] translationMaxMeters Maximum translation [m].
    /// @param[in] rotationMaxRadians Maximum rotation [rad].
    void setRandom(T translationMaxMeters, T rotationMaxRadians);

    /// \brief Set from a homogeneous transformation matrix.
    /// @param[in] T_AB The homogeneous transformation matrix.
    void set(const Eigen::Matrix<T,4,4> & T_AB);

    /// \brief Set from a translation and quaternion.
    /// @param[in] r_AB The translation r_AB (represented in frame A).
    /// @param[in] q_AB The Quaternion q_AB .
    void set(const Eigen::Matrix<T,3,1> & r_AB, const Eigen::Matrix<T,4,1>& q_AB);

    /// \brief Set this transformation to identity
    void setIdentity();

    /// \brief Get an identity transformation
    static Pose Identity();

    /// \brief Returns a copy of the transformation inverted.
    Pose inverse() const;

    // operator* (group operator)
    /// \brief Multiplication with another transformation object.
    /// @param[in] rhs The right-hand side transformation for this to be multiplied with.
    Pose operator*(const Pose & rhs) const;

    /// \brief Transform a direction as v_A = C_AB*v_B (with rhs = hp_B)..
    /// \warning This only applies the rotation!
    /// @param[in] rhs The right-hand side direction for this to be multiplied with.
    Eigen::Matrix<T,3,1> operator*(const Eigen::Matrix<T,3,1> & rhs) const;

    /// \brief Transform a homogenous point as hp_B = T_AB*hp_B (with rhs = hp_B).
    /// @param[in] rhs The right-hand side direction for this to be multiplied with.
    Eigen::Matrix<T,4,1> operator*(const Eigen::Matrix<T,4,1> & rhs) const;

    /// \brief Assignment -- copy. Takes care of proper caching.
    /// @param[in] rhs The rhs for this to be assigned to.
    Pose& operator=(const Pose & rhs);

    /// \brief Apply a small update with delta being 6x1.
    /// \tparam Derived_delta Deducible matrix type.
    /// @param[in] delta The 6x1 minimal update.
    /// \return True on success.
    template<typename Derived_delta>
    bool oplus(const Eigen::MatrixBase<Derived_delta> & delta);

    /// \brief Apply a small update with delta being 6x1 --
    ///        the Jacobian is a 7 by 6 matrix.
    /// @param[in] delta The 6x1 minimal update.
    /// @param[out] jacobian The output Jacobian.
    /// \return True on success.
    template<typename Derived_delta, typename Derived_jacobian>
    bool oplus(const Eigen::MatrixBase<Derived_delta> & delta,
               const Eigen::MatrixBase<Derived_jacobian> & jacobian);

    /// \brief Get the Jacobian of the oplus operation (a 7 by 6 matrix).
    /// @param[out] jacobian The output Jacobian.
    /// \return True on success.
    template<typename Derived_jacobian>
    bool oplusJacobian(
            const Eigen::MatrixBase<Derived_jacobian> & jacobian) const;

    /// \brief Gets the jacobian dx/dChi,
    ///        i.e. lift the minimal Jacobian to a full one (as needed by ceres).
    // @param[out] jacobian The output lift Jacobian (6 by 7 matrix).
    /// \return True on success.
    template<typename Derived_jacobian>
    bool liftJacobian(const Eigen::MatrixBase<Derived_jacobian> & jacobian) const;

    Eigen::Matrix<T,4,1> rotation() const;
    Eigen::Matrix<T,3,1> translation() const;

protected:
    /// \brief Update the caching of the rotation matrix.
    void updateC();
    Eigen::Matrix<T, 7, 1> parameters_;  ///< Concatenated parameters [r;q].
    Eigen::Map<Eigen::Matrix<T,3,1>> r_;  ///< Translation {_A}r_{B}.
    Eigen::Map<Eigen::Matrix<T,4,1>> q_;  ///< Quaternion q_{AB}.
    Eigen::Matrix<T,3,3> C_; ///< The cached DCM C_{AB}.
};

#include "Pose_imp.hpp"

#endif