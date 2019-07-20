
#ifndef SPLINE_PROJECT_FACTOR3_H
#define SPLINE_PROJECT_FACTOR3_H
#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"



struct SplineProjectFunctor3{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SplineProjectFunctor3(const double _t0, const Eigen::Vector3d& uv_C0,
                       const double _t1, const Eigen::Vector3d& uv_C1,
                       const Eigen::Isometry3d _T_IC) :
                       t0_(_t0), C0uv_(uv_C0), t1_(_t1), C1uv_(uv_C1),
                       T_IC_(_T_IC) {}
    template <typename  T>
    bool operator()(const T* const T0_param, const T* const T1_param,
                    const T* const T2_param, const T* const T3_param,
                    const T* const T4_param, const T* const T5_param,
                    const T* const T6_param,
                    const T* const rho_param,
                    T* residuals) const {

        Pose<T> T0(T0_param);
        Pose<T> T1(T1_param);
        Pose<T> T2(T2_param);
        Pose<T> T3(T3_param);
        Pose<T> T4(T4_param);
        Pose<T> T5(T5_param);
        Pose<T> T6(T6_param);

        T inv_dep = *rho_param;

        QuaternionTemplate<T> Q0 = T0.rotation();
        QuaternionTemplate<T> Q1 = T1.rotation();
        QuaternionTemplate<T> Q2 = T2.rotation();
        QuaternionTemplate<T> Q3 = T3.rotation();
        QuaternionTemplate<T> Q4 = T4.rotation();
        QuaternionTemplate<T> Q5 = T5.rotation();
        QuaternionTemplate<T> Q6 = T6.rotation();

        Eigen::Matrix<T,3,1> t0 = T0.translation();
        Eigen::Matrix<T,3,1> t1 = T1.translation();
        Eigen::Matrix<T,3,1> t2 = T2.translation();
        Eigen::Matrix<T,3,1> t3 = T3.translation();
        Eigen::Matrix<T,3,1> t4 = T4.translation();
        Eigen::Matrix<T,3,1> t5 = T5.translation();
        Eigen::Matrix<T,3,1> t6 = T6.translation();

        T  Beta01 = QSUtility::beta1(T(t0_));
        T  Beta02 = QSUtility::beta2(T(t0_));
        T  Beta03 = QSUtility::beta3(T(t0_));

        T  Beta11 = QSUtility::beta1(T(t1_));
        T  Beta12 = QSUtility::beta2(T(t1_));
        T  Beta13 = QSUtility::beta3(T(t1_));
        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);
        Eigen::Matrix<T,3,1> phi4 = QSUtility::Phi<T>(Q3,Q4);
        Eigen::Matrix<T,3,1> phi5 = QSUtility::Phi<T>(Q4,Q5);
        Eigen::Matrix<T,3,1> phi6 = QSUtility::Phi<T>(Q5,Q6);


        QuaternionTemplate<T> r_01 = QSUtility::r(Beta01,phi1);
        QuaternionTemplate<T> r_02 = QSUtility::r(Beta02,phi2);
        QuaternionTemplate<T> r_03 = QSUtility::r(Beta03,phi3);

        QuaternionTemplate<T> r_11 = QSUtility::r(Beta11,phi4);
        QuaternionTemplate<T> r_12 = QSUtility::r(Beta12,phi5);
        QuaternionTemplate<T> r_13 = QSUtility::r(Beta13,phi6);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        QuaternionTemplate<T> Q_WI0_hat = quatLeftComp(Q0)*quatLeftComp(r_01)*quatLeftComp(r_02)*r_03;
        Eigen::Matrix<T,3,1> t_WI0_hat = t0 + Beta01*(t1 - t0) +  Beta02*(t2 - t1) + Beta03*(t3 - t2);

        QuaternionTemplate<T> Q_WI1_hat = quatLeftComp(Q3)*quatLeftComp(r_11)*quatLeftComp(r_12)*r_13;
        Eigen::Matrix<T,3,1> t_WI1_hat = t3 + Beta11*(t4 - t3) +  Beta12*(t5 - t4) + Beta13*(t6 - t5);

//        std::cout << "t_WI0_hat(0): " <<t_WI0_hat(0) << std::endl;
//        std::cout << "t_WI0_hat(1): " <<t_WI0_hat(1) << std::endl;
//        std::cout << "t_WI0_hat(2): " <<t_WI0_hat(2) << std::endl;
//
//        std::cout << "Q_WI0_hat(0): " <<Q_WI0_hat(0) << std::endl;
//        std::cout << "Q_WI0_hat(1): " <<Q_WI0_hat(1) << std::endl;
//        std::cout << "Q_WI0_hat(2): " <<Q_WI0_hat(2) << std::endl;
//        std::cout << "Q_WI0_hat(3): " <<Q_WI0_hat(3) << std::endl;
//
//
//        std::cout << "t_WI1_hat(0): " <<t_WI1_hat(0) << std::endl;
//        std::cout << "t_WI1_hat(1): " <<t_WI1_hat(1) << std::endl;
//        std::cout << "t_WI1_hat(2): " <<t_WI1_hat(2) << std::endl;
//
//        std::cout << "Q_WI1_hat(0): " <<Q_WI1_hat(0) << std::endl;
//        std::cout << "Q_WI1_hat(1): " <<Q_WI1_hat(1) << std::endl;
//        std::cout << "Q_WI1_hat(2): " <<Q_WI1_hat(2) << std::endl;
//        std::cout << "Q_WI1_hat(3): " <<Q_WI1_hat(3) << std::endl;


        Eigen::Matrix<T,3,3> R_WI0 = quatToRotMat(Q_WI0_hat);
        Eigen::Matrix<T,3,3> R_WI1 = quatToRotMat(Q_WI1_hat);

        Eigen::Matrix<T,3,3> R_IC = T_IC_.matrix().topLeftCorner(3,3).cast<T>();
        Eigen::Matrix<T,3,1> t_IC = T_IC_.matrix().topRightCorner(3,1).cast<T>();
        QuaternionTemplate<T> Q_IC = rotMatToQuat(R_IC);

//        std::cout << "t_IC(0): " <<t_IC(0) << std::endl;
//        std::cout << "t_IC(1): " <<t_IC(1) << std::endl;
//        std::cout << "t_IC(2): " <<t_IC(2) << std::endl;
//
//        std::cout << "Q_IC(0): " <<Q_IC(0) << std::endl;
//        std::cout << "Q_IC(1): " <<Q_IC(1) << std::endl;
//        std::cout << "Q_IC(2): " <<Q_IC(2) << std::endl;
//        std::cout << "Q_IC(3): " <<Q_IC(3) << std::endl;

//
        Eigen::Matrix<T,3,1> C0p = C0uv_.cast<T>() / inv_dep;
//        std::cout << "C0p(0): " <<C0p(0) << std::endl;
//        std::cout << "C0p(1): " <<C0p(1) << std::endl;
//        std::cout << "C0p(2): " <<C0p(2) << std::endl;
        Eigen::Matrix<T,3,1> I0p = R_IC * C0p + t_IC;
        Eigen::Matrix<T,3,1> Wp = R_WI0 * I0p + t_WI0_hat;
//        std::cout << "Wp(0): " <<Wp(0) << std::endl;
//        std::cout << "Wp(1): " <<Wp(1) << std::endl;
//        std::cout << "Wp(2): " <<Wp(2) << std::endl;
        Eigen::Matrix<T,3,1> I1p = R_WI1.transpose() * (Wp - t_WI1_hat);
//        std::cout << "I1p(0): " <<I1p(0) << std::endl;
//        std::cout << "I1p(1): " <<I1p(1) << std::endl;
//        std::cout << "I1p(2): " <<I1p(2) << std::endl;
        Eigen::Matrix<T,3,1> C1p = R_IC.transpose() * (I1p - t_IC);
//        std::cout << "C1p(0): " <<C1p(0) << std::endl;
//        std::cout << "C1p(1): " <<C1p(1) << std::endl;
//        std::cout << "C1p(2): " <<C1p(2) << std::endl;


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

class SplineProjectError3{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SplineProjectError3() = delete;
    SplineProjectError3(const SplineProjectFunctor3& functor);

    /// \brief Trivial destructor.
    virtual ~SplineProjectError3() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:
    SplineProjectFunctor3 functor_;
};

#endif
