#ifndef LINEARACCELERATESAMPLEERROR_H
#define LINEARACCELERATESAMPLEERROR_H

#include "pose-spline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "pose-spline/ErrorInterface.hpp"
#include "pose-spline/QuaternionSplineUtility.hpp"



struct LinearAccelerateSampleFunctor{

    LinearAccelerateSampleFunctor(const double ts, const double& deltat,
                                 const Eigen::Vector3d& accelSample,
                                 const double& weightScale)
            : ts_(ts),
              deltaT_(deltat),
              accelSample_(accelSample),
              weightScale_(weightScale){
    }

    template <typename  T>
    bool operator()(const T* const parameters0, const T* const parameters1,
                    const T* const parameters2, const T* const parameters3, T* residuals) const
    {


        Eigen::Map<const Eigen::Matrix<T,3,1>> V0(parameters0);
        Eigen::Map<const Eigen::Matrix<T,3,1>> V1(parameters1);
        Eigen::Map<const Eigen::Matrix<T,3,1>> V2(parameters2);
        Eigen::Map<const Eigen::Matrix<T,3,1>> V3(parameters3);


        Eigen::Map<const Eigen::Matrix<T,4,1>> Q0(parameters0+3);
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q1(parameters1+3);
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q2(parameters2+3);
        Eigen::Map<const Eigen::Matrix<T,4,1>> Q3(parameters3+3);

        T dt_T = T(deltaT_);
        T u_T = T(ts_);

        T  ddBeta1 = QSUtility::dot_dot_beta1(dt_T, u_T);
        T  ddBeta2 = QSUtility::dot_dot_beta2(dt_T, u_T);
        T  ddBeta3 = QSUtility::dot_dot_beta3(dt_T, u_T);

        Eigen::Map<Eigen::Matrix<T,3,1>> error(residuals);
        T  Beta1 = QSUtility::beta1(u_T);
        T  Beta2 = QSUtility::beta2(u_T);
        T  Beta3 = QSUtility::beta3(u_T);

        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);

        Eigen::Matrix<T,4,1> r_1 = QSUtility::r(Beta1,phi1);
        Eigen::Matrix<T,4,1> r_2 = QSUtility::r(Beta2,phi2);
        Eigen::Matrix<T,4,1> r_3 = QSUtility::r(Beta3,phi3);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        Eigen::Matrix<T,4,1> Q_WI = quatLeftComp<T>(Q0)*quatLeftComp(r_1)*quatLeftComp(r_2)*r_3;
        Eigen::Matrix<T,3,3> R_WI = quatToRotMat(Q_WI);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        // /hat = R_WI^T( W_a)
        Eigen::Matrix<T,3,1> Wa = ddBeta1*(V1 - V0) +  ddBeta2*(V2 - V1) + ddBeta3*(V3 - V2);
        Eigen::Matrix<T,3,1> a_hat
                = R_WI.transpose()*Wa;

        error = a_hat - accelSample_.cast<T>();
        Eigen::Matrix<T,3,3> squareInformation_ = (T)weightScale_*Eigen::Matrix<T,3,3>::Identity();
        error = squareInformation_*error;

        return true;
    }



private:
    double ts_;
    double deltaT_;
    Eigen::Vector3d accelSample_;

    double  weightScale_;
};


class LinearAccelerateSampleError: public ceres::SizedCostFunction<3,7,7,7,7>{
public:
    typedef Eigen::Matrix<double, 3, 3> covariance_t;
    typedef covariance_t information_t;

    LinearAccelerateSampleError(const double& t_meas, const double& time_interval, const Eigen::Vector3d& a_meas);
    LinearAccelerateSampleError(LinearAccelerateSampleFunctor* functor);
    virtual ~LinearAccelerateSampleError();

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const;

    bool AutoEvaluateWithMinimalJacobians(double const* const * parameters,
                                                                   double* residuals,
                                                                   double** jacobians,
                                                                   double** jacobiansMinimal) const;


private:

    double t_meas_;
    double time_interval_;
    Eigen::Vector3d a_Meas_;
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.

    LinearAccelerateSampleFunctor* functor_;
};

#endif