#ifndef RORATEVECTORERROR_H
#define  RORATEVECTORERROR_H


#include "PoseSpline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"


class TransformVectorError
        : public ceres::SizedCostFunction<3, 7, 7, 7, 7> {
public:

    typedef Eigen::Matrix<double, 6, 6> covariance_t;
    typedef covariance_t information_t;

    TransformVectorError(double t_meas, Eigen::Vector3d originalVector,
                       Eigen::Vector3d rotatedVector);
    virtual ~TransformVectorError();

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