#ifndef  QUATERNIONSPLINESAMPLEAUTOERROR_H
#define  QUATERNIONSPLINESAMPLEAUTOERROR_H

#include "PoseSpline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"



class QuaternionSplineSampleAutoError {
public:

    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    QuaternionSplineSampleAutoError(double t_meas,
                                Quaternion Q_meas):
            t_meas_(t_meas),Q_Meas_(Q_meas){

    };

    virtual ~QuaternionSplineSampleAutoError(){};

    template <typename T>
    bool operator()(const T* const params0,
                    const T* const params1,
                    const T* const params2,
                    const T* const params3,
                    T* residual) const {
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q0(params0);
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q1(params0);
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q2(params0);
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q3(params0);

        Eigen::Map<Eigen::Matrix<T,3,1>> error(residual);


        T  Beta1 = QSUtility::beta1((T)t_meas_);
        T  Beta2 = QSUtility::beta2((T)t_meas_);
        T  Beta3 = QSUtility::beta3((T)t_meas_);

        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);

        Eigen::Matrix<T,4,1> r_1 = QSUtility::r(Beta1,phi1);
        Eigen::Matrix<T,4,1> r_2 = QSUtility::r(Beta2,phi2);
        Eigen::Matrix<T,4,1> r_3 = QSUtility::r(Beta3,phi3);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        // delte_Q = Q_hat*inv(Q_meas) = (inv(Q_meas))oplus Q_hat
        Eigen::Matrix<T,4,1> Q_hat = quatLeftComp<T>(Q0)*quatLeftComp<T>(r_1)*quatLeftComp<T>(r_2)*r_3;
        Eigen::Matrix<T,4,1> dQ = quatLeftComp<T>(Q_hat)*quatInv<T>(Q_Meas_.cast<T>());
        error = T(2) * dQ.topLeftCorner(3,1);
        return true;
    }



private:

    double t_meas_;
    Quaternion Q_Meas_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif