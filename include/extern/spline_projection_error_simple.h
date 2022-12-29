
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

    bool operator()(const double* const T0_param, const double* const T1_param,
                    const double* const T2_param, const double* const T3_param,
                    const double* const Wp_param, double* residuals) const {

        typedef  double T;
        Pose<T> T0(T0_param);
        Pose<T> T1(T1_param);
        Pose<T> T2(T2_param);
        Pose<T> T3(T3_param);
        Eigen::Map<const Eigen::Matrix<T,3,1>> Wp(Wp_param);
        QuaternionTemplate<T> Q0 = T0.rotation();
        QuaternionTemplate<T> Q1 = T1.rotation();
        QuaternionTemplate<T> Q2 = T2.rotation();
        QuaternionTemplate<T> Q3 = T3.rotation();

//        std::cout << "Q0: " << Q0[0] << std::endl;
//        std::cout << "Q0: " << Q0[1] << std::endl;
//        std::cout << "Q0: " << Q0[2] << std::endl;
//        std::cout << "Q0: " << Q0[3] << std::endl;
//
//        std::cout << "Q1: " << Q1[0] << std::endl;
//        std::cout << "Q1: " << Q1[1] << std::endl;
//        std::cout << "Q1: " << Q1[2] << std::endl;
//        std::cout << "Q1: " << Q1[3] << std::endl;
//
//
//        std::cout << "Q2: " << Q2[0] << std::endl;
//        std::cout << "Q2: " << Q2[1] << std::endl;
//        std::cout << "Q2: " << Q2[2] << std::endl;
//        std::cout << "Q2: " << Q2[3] << std::endl;
//
//
//        std::cout << "Q3: " << Q3[0] << std::endl;
//        std::cout << "Q3: " << Q3[1] << std::endl;
//        std::cout << "Q3: " << Q3[2] << std::endl;
//        std::cout << "Q3: " << Q3[3] << std::endl;
        Eigen::Matrix<T,3,1> t0 = T0.translation();
        Eigen::Matrix<T,3,1> t1 = T1.translation();
        Eigen::Matrix<T,3,1> t2 = T2.translation();
        Eigen::Matrix<T,3,1> t3 = T3.translation();
        T  Beta01 = QSUtility::beta1(T(t_));
        T  Beta02 = QSUtility::beta2(T(t_));
        T  Beta03 = QSUtility::beta3(T(t_));
//        std::cout << "Beta03: " << Beta03 << std::endl;
        Eigen::Matrix<T,3,1> phi1 = QSUtility::Phi<T>(Q0,Q1);
        Eigen::Matrix<T,3,1> phi2 = QSUtility::Phi<T>(Q1,Q2);
        Eigen::Matrix<T,3,1> phi3 = QSUtility::Phi<T>(Q2,Q3);

//        std::cout << "phi3: " << phi3[0] << std::endl;
//        std::cout << "phi3: " << phi3[1] << std::endl;
//        std::cout << "phi3: " << phi3[2] << std::endl;
        QuaternionTemplate<T> r_01 = QSUtility::r(Beta01,phi1);
        QuaternionTemplate<T> r_02 = QSUtility::r(Beta02,phi2);
        QuaternionTemplate<T> r_03 = QSUtility::r(Beta03,phi3);

//        std::cout << "r_01: " << r_01[0] << std::endl;
//        std::cout << "r_01: " << r_01[1] << std::endl;
//        std::cout << "r_01: " << r_01[2] << std::endl;
//        std::cout << "r_01: " << r_01[3] << std::endl;
//
//        std::cout << "r_02: " << r_02[0] << std::endl;
//        std::cout << "r_02: " << r_02[1] << std::endl;
//        std::cout << "r_02: " << r_02[2] << std::endl;
//        std::cout << "r_02: " << r_02[3] << std::endl;
//
//        std::cout << "r_03: " << r_03[0] << std::endl;
//        std::cout << "r_03: " << r_03[1] << std::endl;
//        std::cout << "r_03: " << r_03[2] << std::endl;
//        std::cout << "r_03: " << r_03[3] << std::endl;

        // define residual
        // For simplity, we define error  =  /hat - meas.
        QuaternionTemplate<T> Q_WI_hat = quatLeftComp(Q0)*quatLeftComp(r_01)*quatLeftComp(r_02)*r_03;
        Eigen::Matrix<T,3,1> t_WI_hat = t0 + Beta01*(t1 - t0) +  Beta02*(t2 - t1) + Beta03*(t3 - t2);

//        std::cout << "Q_WI_hat: " << Q_WI_hat[0] << std::endl;
//        std::cout << "Q_WI_hat: " << Q_WI_hat[1] << std::endl;
//        std::cout << "Q_WI_hat: " << Q_WI_hat[2] << std::endl;
//        std::cout << "Q_WI_hat: " << Q_WI_hat[3] << std::endl;

        Eigen::Matrix<T,3,3> R_WI = quatToRotMat(Q_WI_hat);

        Eigen::Matrix<T,3,3> R_IC = T_IC_.matrix().topLeftCorner(3,3).cast<T>();
        Eigen::Matrix<T,3,1> t_IC = T_IC_.matrix().topRightCorner(3,1).cast<T>();
//
//        std::cout << "t: " << T(t_) << std::endl;
//        std::cout << "Cuv_.head<2>().cast<T>(): " << Cuv_.head<2>().cast<T>()[0] << std::endl;
//        std::cout << "Cuv_.head<2>().cast<T>(): " << Cuv_.head<2>().cast<T>()[1] << std::endl;


        Eigen::Matrix<T,3,1> Ip = R_WI.inverse() * (Wp - t_WI_hat);
//        std::cout << "Ip: " << Ip[0] << std::endl;
//        std::cout << "Ip: " << Ip[1] << std::endl;
//        std::cout << "Ip: " << Ip[2] << std::endl;

        Eigen::Matrix<T,3,1> Cp = R_IC.inverse() * (Ip - t_IC);
//        std::cout << "Cp: " << Cp[0] << std::endl;
//        std::cout << "Cp: " << Cp[1] << std::endl;
//        std::cout << "Cp: " << Cp[2] << std::endl;

        Eigen::Matrix<T, 2, 1> error;
        T inv_z = T(1.0)/Cp(2);
        Eigen::Matrix<T,2,1> hat_Cuv(Cp(0)*inv_z, Cp(1)*inv_z);

        error = hat_Cuv - Cuv_.head<2>().cast<T>();
//        std::cout << "error: " << error[0] << std::endl;
//        std::cout << "error: " << error[1] << std::endl;
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
    SplineProjectSimpleError(const double _t, const Eigen::Vector3d& uv_C,
                             const Eigen::Isometry3d _T_IC) :
            t_(_t), Cuv_(uv_C),
            T_IC_(_T_IC) {}


    /// \brief Trivial destructor.
    virtual ~SplineProjectSimpleError() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:
    double t_;
    Eigen::Vector3d Cuv_;
    Eigen::Isometry3d T_IC_;
};


ceres::CostFunction* createSplineProjectSimpleError(const double _t, const Eigen::Vector3d& uv_C,
                                                    const Eigen::Isometry3d _T_IC) {
    return new ceres::NumericDiffCostFunction<SplineProjectSimpleFunctor, ceres::NumericDiffMethodType::CENTRAL,
            2,7,7,7,7,3>(new SplineProjectSimpleFunctor(_t, uv_C, _T_IC));
}
#endif
