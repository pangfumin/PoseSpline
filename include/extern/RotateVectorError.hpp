#ifndef RORATEVECTORERROR_H
#define  RORATEVECTORERROR_H


#include "pose-spline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "ceres/ErrorInterface.hpp"


class RoatateVectorError
        : public ceres::SizedCostFunction<3, 4, 4, 4, 4> {
public:

    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    RoatateVectorError(double t_meas, Eigen::Vector3d originalVector,
                       Eigen::Vector3d rotatedVector);
    virtual ~RoatateVectorError();

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const;


private:

    double t_meas_;
    Eigen::Vector3d rotatedVector_Meas_;
    Eigen::Vector3d originalVector_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif