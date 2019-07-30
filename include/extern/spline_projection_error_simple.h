
#ifndef SPLINE_PROJECT_FACTOR_SIMPLE_H
#define SPLINE_PROJECT_FACTOR_SIMPLE_H
#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"

struct SplineProjectSimpleFunctor{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SplineProjectSimpleFunctor(const double _t, const Eigen::Vector3d& uv_C,
                       const Eigen::Isometry3d _T_IC) :
                       t_(_t), Cuv_(uv_C),
                       T_IC_(_T_IC) {}
    template <typename  T>
    bool operator()(const T* const T0_param, const T* const T1_param,
                    const T* const T2_param, const T* const T3_param,
                    const T* const Wp_param, T* residuals) const {

        Pose<T> T0(T0_param);
        Pose<T> T1(T1_param);
        Pose<T> T2(T2_param);
        Pose<T> T3(T3_param);

        Eigen::Map<const Eigen::Matrix<T,3,1>> Wp(Wp_param);


        QuaternionTemplate<T> Q0 = T0.rotation();
        QuaternionTemplate<T> Q1 = T1.rotation();
        QuaternionTemplate<T> Q2 = T2.rotation();
        QuaternionTemplate<T> Q3 = T3.rotation();

        Eigen::Matrix<T,3,1> t0 = T0.translation();
        Eigen::Matrix<T,3,1> t1 = T1.translation();
        Eigen::Matrix<T,3,1> t2 = T2.translation();
        Eigen::Matrix<T,3,1> t3 = T3.translation();

        T  Beta01 = QSUtility::beta1(T(t_));
        T  Beta02 = QSUtility::beta2(T(t_));
        T  Beta03 = QSUtility::beta3(T(t_));


        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);

        QuaternionTemplate<T> r_01 = QSUtility::r(Beta01,phi1);
        QuaternionTemplate<T> r_02 = QSUtility::r(Beta02,phi2);
        QuaternionTemplate<T> r_03 = QSUtility::r(Beta03,phi3);



        // define residual
        // For simplity, we define error  =  /hat - meas.
        QuaternionTemplate<T> Q_WI_hat = quatLeftComp(Q0)*quatLeftComp(r_01)*quatLeftComp(r_02)*r_03;
        Eigen::Matrix<T,3,1> t_WI_hat = t0 + Beta01*(t1 - t0) +  Beta02*(t2 - t1) + Beta03*(t3 - t2);


        Eigen::Matrix<T,3,3> R_WI = quatToRotMat(Q_WI_hat);

        Eigen::Matrix<T,3,3> R_IC = T_IC_.matrix().topLeftCorner(3,3).cast<T>();
        Eigen::Matrix<T,3,1> t_IC = T_IC_.matrix().topRightCorner(3,1).cast<T>();
//

        Eigen::Matrix<T,3,1> Ip = R_WI.inverse() * (Wp - t_WI_hat);
        Eigen::Matrix<T,3,1> Cp = R_IC.inverse() * (Ip - t_IC);


        Eigen::Matrix<T, 2, 1> error;
        T inv_z = T(1.0)/Cp(2);
        Eigen::Matrix<T,2,1> hat_Cuv(Cp(0)*inv_z, Cp(1)*inv_z);

        error = hat_Cuv - Cuv_.head<2>().cast<T>();

        // weight it
        Eigen::Map<Eigen::Matrix<T, 2, 1> > weighted_error(residuals);
        weighted_error =  error;
//

        return true;
    }

    double t_;
    Eigen::Vector3d Cuv_;
    Eigen::Isometry3d T_IC_;
};

class SplineProjectSimpleError: public ceres::SizedCostFunction<2,7,7,7,7,3>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SplineProjectSimpleError() = delete;
    SplineProjectSimpleError(const SplineProjectSimpleFunctor& functor);

    /// \brief Trivial destructor.
    virtual ~SplineProjectSimpleError() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:
    SplineProjectSimpleFunctor functor_;
};

#endif
