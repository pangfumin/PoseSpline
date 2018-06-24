#ifndef VECTORSPLINESAMPLEERROR_H
#define  VECTORSPLINESAMPLEERROR_H

#include "pose-spline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "pose-spline/ErrorInterface.hpp"
#include "pose-spline/QuaternionSplineUtility.hpp"


class VectorSplineSampleError: public ceres::SizedCostFunction<3,3,3,3,3>{
public:
    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    VectorSplineSampleError(const double& t_meas, const Eigen::Vector3d& V_meas);
    virtual ~VectorSplineSampleError();

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const;

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
//        auto trajectory = entity::Map<TrajectoryModel, T>(params, meta);
//        Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
//        r = measurement.Error<TrajectoryModel, T>(trajectory);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V0(params[0]);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V1(params[1]);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V2(params[2]);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V3(params[3]);

        T  Beta1 = QSUtility::beta1(t_meas_);
        T  Beta2 = QSUtility::beta2(t_meas_);
        T  Beta3 = QSUtility::beta3(t_meas_);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        Eigen::Matrix<T,3,1> V_hat = V0 + Beta1*(V1 - V0) +  Beta2*(V2 - V1) + Beta3*(V3 - V2);

        Eigen::Map<Eigen::Matrix<T,3,1>> error(residual);

        //squareRootInformation_ = Eigen::Matrix3d::Identity();
        error = V_hat - V_Meas_.cast<T>();

        return true;
    }
private:

    double t_meas_;
    Eigen::Vector3d V_Meas_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif