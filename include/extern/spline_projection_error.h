
#ifndef SPLINe_PROJECT_FACTOR_H
#define SPLINe_PROJECT_FACTOR_H
#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"

struct SplineProjectFunctor{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SplineProjectFunctor(const double _t0, const Eigen::Vector3d& uv_C0,
                       const double _t1, const Eigen::Vector3d& uv_C1,
                       const Eigen::Isometry3d _T_IC) :
                       t0_(_t0), C0uv_(uv_C0), t1_(_t1), C1uv_(uv_C1),
                       T_IC_(_T_IC) {}
    template <typename  T>
    bool operator()(const T* const T0_param, const T* const T1_param,
                    const T* const T2_param, const T* const T3_param,
                    const T* const rho_param,
                    const T* const bias_t,
                    T* residuals) const {

        Pose<T> T0(T0_param);
        Pose<T> T1(T1_param);
        Pose<T> T2(T2_param);
        Pose<T> T3(T3_param);

        T inv_dep = *rho_param;


        QuaternionTemplate<T> Q0 = T0.rotation();
        QuaternionTemplate<T> Q1 = T1.rotation();
        QuaternionTemplate<T> Q2 = T2.rotation();
        QuaternionTemplate<T> Q3 = T3.rotation();

        Eigen::Matrix<T,3,1> t0 = T0.translation();
        Eigen::Matrix<T,3,1> t1 = T1.translation();
        Eigen::Matrix<T,3,1> t2 = T2.translation();
        Eigen::Matrix<T,3,1> t3 = T3.translation();

        T correst_t0 = T(t0_) + *bias_t;
        T correst_t1 = T(t1_) + *bias_t;

//        std::cout << "bias_t[0]: " << bias_t[0] << std::endl;

        T  Beta01 = QSUtility::beta1(correst_t0);
        T  Beta02 = QSUtility::beta2(correst_t0);
        T  Beta03 = QSUtility::beta3(correst_t0);

        T  Beta11 = QSUtility::beta1(correst_t1);
        T  Beta12 = QSUtility::beta2(correst_t1);
        T  Beta13 = QSUtility::beta3(correst_t1);

        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);

        QuaternionTemplate<T> r_01 = QSUtility::r(Beta01,phi1);
        QuaternionTemplate<T> r_02 = QSUtility::r(Beta02,phi2);
        QuaternionTemplate<T> r_03 = QSUtility::r(Beta03,phi3);

        QuaternionTemplate<T> r_11 = QSUtility::r(Beta11,phi1);
        QuaternionTemplate<T> r_12 = QSUtility::r(Beta12,phi2);
        QuaternionTemplate<T> r_13 = QSUtility::r(Beta13,phi3);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        QuaternionTemplate<T> Q_WI0_hat = quatLeftComp(Q0)*quatLeftComp(r_01)*quatLeftComp(r_02)*r_03;
        Eigen::Matrix<T,3,1> t_WI0_hat = t0 + Beta01*(t1 - t0) +  Beta02*(t2 - t1) + Beta03*(t3 - t2);

        QuaternionTemplate<T> Q_WI1_hat = quatLeftComp(Q0)*quatLeftComp(r_11)*quatLeftComp(r_12)*r_13;
        Eigen::Matrix<T,3,1> t_WI1_hat = t0 + Beta11*(t1 - t0) +  Beta12*(t2 - t1) + Beta13*(t3 - t2);

        Eigen::Matrix<T,3,3> R_WI0 = quatToRotMat(Q_WI0_hat);
        Eigen::Matrix<T,3,3> R_WI1 = quatToRotMat(Q_WI1_hat);

//        std::cout << "R_WI0:  " <<  Eigen::Quaternion<T>(R_WI0).coeffs().transpose() << std::endl;
//        std::cout << "R_WI1:  " <<  Eigen::Quaternion<T>(R_WI1).coeffs().transpose() << std::endl;

//        std::cout << "R_WI0: \n" << R_WI0 << std::endl;

        Eigen::Matrix<T,3,3> R_IC = T_IC_.matrix().topLeftCorner(3,3).cast<T>();
        Eigen::Matrix<T,3,1> t_IC = T_IC_.matrix().topRightCorner(3,1).cast<T>();
//
        Eigen::Matrix<T,3,1> C0p = C0uv_.cast<T>() / inv_dep;
        Eigen::Matrix<T,3,1> I0p = R_IC * C0p + t_IC;
        Eigen::Matrix<T,3,1> Wp = R_WI0 * I0p + t_WI0_hat;
        Eigen::Matrix<T,3,1> I1p = R_WI1.inverse() * (Wp - t_WI1_hat);
        Eigen::Matrix<T,3,1> C1p = R_IC.inverse() * (I1p - t_IC);


        Eigen::Matrix<T, 2, 1> error;

        T inv_z = T(1.0)/C1p(2);
        Eigen::Matrix<T,2,1> hat_C1uv(C1p(0)*inv_z, C1p(1)*inv_z);

        error = hat_C1uv - C1uv_.head<2>().cast<T>();

        // weight it
        Eigen::Map<Eigen::Matrix<T, 2, 1> > weighted_error(residuals);
        weighted_error =  error;
//

        return true;
    }

    double t0_,t1_;
    Eigen::Vector3d C0uv_;
    Eigen::Vector3d C1uv_;
    Eigen::Isometry3d T_IC_;
};

class SplineProjectError : public ceres::SizedCostFunction<2,7,7,7,7,1,1>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SplineProjectError() = delete;
    SplineProjectError(const SplineProjectFunctor& functor);

    /// \brief Trivial destructor.
    virtual ~SplineProjectError() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:
    SplineProjectFunctor functor_;
};

#endif
