#ifndef CERES_ERROR_TERMS_PARAMETERIZATION_POSE_PARAM_JPL_H_
#define CERES_ERROR_TERMS_PARAMETERIZATION_POSE_PARAM_JPL_H_

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "PoseSpline/maplab/quaternion_param_jpl.h"

namespace ceres_error_terms {

class JplPoseParameterization : public ceres::ProductParameterization {
 public:
  JplPoseParameterization()
      : ceres::ProductParameterization(
            
            new ceres::IdentityParameterization(3),
            new JplQuaternionParameterization) {}
  virtual ~JplPoseParameterization() {}
};


}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_PARAMETERIZATION_POSE_PARAM_JPL_H_
