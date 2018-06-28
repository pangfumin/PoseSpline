#ifndef LINEARACCELERATESAMPLEERROR_H
#define LINEARACCELERATESAMPLEERROR_H

#include "pose-spline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "pose-spline/ErrorInterface.hpp"


class LinearAccelerateSampleError: public ceres::SizedCostFunction<3,7,7,7,7>{
public:
    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    LinearAccelerateSampleError(const double& t_meas, const double& time_interval, const Eigen::Vector3d& a_meas);
    virtual ~LinearAccelerateSampleError();

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const;


private:

    double t_meas_;
    double time_interval_;
    Eigen::Vector3d a_Meas_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif