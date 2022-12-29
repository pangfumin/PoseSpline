#ifndef  VECTORSPLINESAMPLEVELOCITYERROR_H
#define  VECTORSPLINESAMPLEVELOCITYERROR_H

#include "PoseSpline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"


class VectorSplineSampleVelocityError: public ceres::SizedCostFunction<3,3,3,3,3>{
public:
    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    VectorSplineSampleVelocityError(const double& t_meas, const double& time_interval, const Eigen::Vector3d& V_meas);
    virtual ~VectorSplineSampleVelocityError();

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
    Eigen::Vector3d V_Meas_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif