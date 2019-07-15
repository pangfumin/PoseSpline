#ifndef VECTORSPLINESAMPLEAUTOERROR_H
#define  VECTORSPLINESAMPLEAUTOERROR_H

#include "PoseSpline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"


class VectorSplineSampleAutoError {
public:
    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    VectorSplineSampleAutoError(const double& t_meas, const Eigen::Vector3d& V_meas);
    virtual ~VectorSplineSampleAutoError();



    template <typename T>
    bool operator()(const T* const params0,
                    const T* const params1,
                    const T* const params2,
                    const T* const params3,
                    T* residual) const {

        Eigen::Map<const Eigen::Matrix<T, 3,1>> V0(params0);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V1(params1);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V2(params2);
        Eigen::Map<const Eigen::Matrix<T, 3,1>> V3(params3);

        T  Beta1 = QSUtility::beta1(T(t_meas_));
        T  Beta2 = QSUtility::beta2(T(t_meas_));
        T  Beta3 = QSUtility::beta3(T(t_meas_));

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