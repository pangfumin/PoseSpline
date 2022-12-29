#ifndef POSESPLINESAMPLE_TEMPORAL_ERROR_H
#define  POSESPLINESAMPLE_TEMPORAL_ERROR_H

#include "PoseSpline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"


struct PoseSplineSampleTemporalFunctor {
public:

    typedef Eigen::Matrix<double, 6, 6> covariance_t;
    typedef covariance_t information_t;

    PoseSplineSampleTemporalFunctor(double t_meas, Pose<double> T_meas) :
            t_meas_(t_meas),T_Meas_(T_meas){
    };

    virtual ~PoseSplineSampleTemporalFunctor() {

    }

    template <typename  T>
    bool operator()(const T* const pose0, const T* const pose1,
                    const T* const pose2, const T* const pose3,
                    const T* const bias_t,
                    T* residuals) const
    {


        Eigen::Map<const QuaternionTemplate<T>> Q0(pose0+3);
        Eigen::Map<const QuaternionTemplate<T>> Q1(pose1+3);
        Eigen::Map<const QuaternionTemplate<T>> Q2(pose2+3);
        Eigen::Map<const QuaternionTemplate<T>> Q3(pose3+3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t0(pose0);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t1(pose1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t2(pose2);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t3(pose3);

        Eigen::Map<Eigen::Matrix<T, 6, 1>> error(residuals);

        T dt = bias_t[0];

        T  Beta1 = QSUtility::beta1(T(t_meas_)  + dt);
        T  Beta2 = QSUtility::beta2(T(t_meas_)  + dt);
        T  Beta3 = QSUtility::beta3(T(t_meas_)  + dt);

        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);

        QuaternionTemplate<T> r_1 = QSUtility::r(Beta1,phi1);
        QuaternionTemplate<T> r_2 = QSUtility::r(Beta2,phi2);
        QuaternionTemplate<T> r_3 = QSUtility::r(Beta3,phi3);
//
        // define residual
        // For simplity, we define error  =  /hat - meas.
        // delte_Q = Q_hat*inv(Q_meas) = (inv(Q_meas))oplus Q_hat
        QuaternionTemplate<T> Q_hat = quatLeftComp<T>(Q0)*quatLeftComp<T>(r_1)*quatLeftComp<T>(r_2)*r_3;
        QuaternionTemplate<T> R_meas  = T_Meas_.rotation().cast<T>();
        QuaternionTemplate<T> dQ = quatLeftComp<T>(Q_hat)*quatInv<T>(R_meas);

        Eigen::Matrix<T,3,1> t_hat = t0 + Beta1*(t1 - t0) +  Beta2*(t2 - t1) + Beta3*(t3 - t2);

        error.head(3) =  t_hat - T_Meas_.translation().cast<T>();
        error.tail(3) =  T(2.0) * dQ.template head<3>();
    }
private:

    double t_meas_;
    Pose<double> T_Meas_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};



class PoseSplineSampleTemporalError: public ceres::SizedCostFunction<6,7,7,7,7,1>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseSplineSampleTemporalError() = delete;
    PoseSplineSampleTemporalError(const PoseSplineSampleTemporalFunctor& functor);

    /// \brief Trivial destructor.
    virtual ~PoseSplineSampleTemporalError() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:
    PoseSplineSampleTemporalFunctor functor_;
};

#endif