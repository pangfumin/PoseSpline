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
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file PoseLocalParameterization.cpp
 * @brief Source file for the PoseLocalParameterization class.
 * @author Stefan Leutenegger
 */

#include <okvis_util/assert_macros.hpp>
#include <ceres/PoseLocalParameterization.hpp>
#include <geometry/Pose.hpp>
//#include <okvis/kinematics/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool PoseLocalParameterization::Plus(const double* x, const double* delta,
                                     double* x_plus_delta) const {
  return plus(x, delta, x_plus_delta);
}

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool PoseLocalParameterization::plus(const double* x, const double* delta,
                                     double* x_plus_delta) {

  Eigen::Map<const Eigen::Matrix<double, 6, 1> > delta_(delta);

  // transform to okvis::kinematics framework
      Pose<double> T(
      Eigen::Vector3d(x[0], x[1], x[2]),
      Quaternion( x[3], x[4], x[5], x[6]));

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

  OKVIS_ASSERT_TRUE_DBG(std::runtime_error, T.q().norm()-1.0<1e-15, "damn.");

  return true;
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool PoseLocalParameterization::Minus(const double* x,
                                      const double* x_plus_delta,
                                      double* delta) const {
  return minus(x, x_plus_delta, delta);
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool PoseLocalParameterization::ComputeLiftJacobian(const double* x,
                                                    double* jacobian) const {
  return liftJacobian(x, jacobian);
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool PoseLocalParameterization::minus(const double* x,
                                      const double* x_plus_delta,
                                      double* delta) {

  delta[0] = x_plus_delta[0] - x[0];
  delta[1] = x_plus_delta[1] - x[1];
  delta[2] = x_plus_delta[2] - x[2];
  const Quaternion q_plus_delta_( x_plus_delta[3],
                                         x_plus_delta[4], x_plus_delta[5],x_plus_delta[6]);
  const Quaternion q_( x[3], x[4], x[5], x[6]);
  Eigen::Map<Eigen::Vector3d> delta_q_(&delta[3]);
  delta_q_ = 2 * (quatLeftComp(q_plus_delta_) * quatInv(q_)).template head<3>();
  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool PoseLocalParameterization::plusJacobian(const double* x,
                                             double* jacobian) {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
      Pose<double> T(
      Eigen::Vector3d(x[0], x[1], x[2]),
      Quaternion( x[3], x[4], x[5],x[6]));
  T.oplusJacobian(Jp);

  return true;
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool PoseLocalParameterization::liftJacobian(const double* x,
                                             double* jacobian) {

  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);
  const Quaternion q_inv( -x[3], -x[4], -x[5], x[6]);
  J_lift.setZero();
  J_lift.topLeftCorner<3, 3>().setIdentity();
  Eigen::Matrix4d Qplus = quatRightComp(q_inv);
  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3, 1>().setZero();
  Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;
  J_lift.bottomRightCorner<3, 4>() = Jq_pinv * Qplus;

  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool PoseLocalParameterization::ComputeJacobian(const double* x,
                                                double* jacobian) const {

  return plusJacobian(x, jacobian);
}

bool PoseLocalParameterization::VerifyJacobianNumDiff(const double* x,
                                                      double* jacobian,
                                                      double* jacobianNumDiff) {
  plusJacobian(x, jacobian);
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
}


}  // namespace ceres
}  // namespace okvis
