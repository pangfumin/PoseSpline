#ifndef OMEGAEXTRINSICTEMPERALERROR_H
#define OMEGAEXTRINSICTEMPERALERROR_H

#include "pose-spline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "pose-spline/ErrorInterface.hpp"

class OmegaExtrinsicTemperalError: public ceres::SizedCostFunction<3,4,1>{

    OmegaExtrinsicTemperalError();
    OmegaExtrinsicTemperalError(const Eigen::Vector3d& omega_meas,
                                const Quaternion& Q_cw,
                                const Quaternion& dotQ_cw,
                                const Quaternion& dotdotQ_cw);

    virtual ~OmegaExtrinsicTemperalError();

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const;

private:
    Eigen::Vector3d omega_meas_;
    Quaternion Q_cw_;
    Quaternion dotQ_cw_;
    Quaternion dotdotQ_cw_;


};

#endif