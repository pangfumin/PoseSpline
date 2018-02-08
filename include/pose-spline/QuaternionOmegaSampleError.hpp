#ifndef  QUATERNION_OMEGA_SAMPLE_ERROR
#define  QUATERNION_OMEGA_SAMPLE_ERROR


#include "pose-spline/QuaternionSplineUtility.hpp"
class QuaternionOmegaSampleError{


    QuaternionOmegaSampleError(const double ts, const double& deltat,
                               const Eigen::Vector3d& omegaSample,
                                const double& weightScale)
            : deltaT_(deltat),
              omegaSample_(omegaSample),
              weightScale_(weightScale){

    }

    template <typename  T>
    bool operator()(const T* const Q0, const T* const Q1, const T* const Q2, const T* const Q3, T* residuals) const
    {
        Eigen::Map<Eigen::Matrix<T, 4, 1>> Quternion0(Q0);
        Eigen::Map<Eigen::Matrix<T, 4, 1>> Quternion1(Q1);
        Eigen::Map<Eigen::Matrix<T, 4, 1>> Quternion2(Q2);
        Eigen::Map<Eigen::Matrix<T, 4, 1>> Quternion3(Q3);

        Eigen::Matrix<T,3,1> Phi1 = QSUtility::Phi(Quternion0,Quternion1);
        Eigen::Matrix<T,3,1> Phi2 = QSUtility::Phi(Quternion1,Quternion2);
        Eigen::Matrix<T,3,1> Phi3 = QSUtility::Phi(Quternion2,Quternion3);

        T u = ts / T(deltaT_);
        T b1 = QSUtility::beta1(u);
        T b2 = QSUtility::beta2(u);
        T b3 = QSUtility::beta3(u);


        T dot_b1 = QSUtility::dot_beta1(dt,u);
        T dot_b2 = QSUtility::dot_beta2(dt,u);
        T dot_b3 = QSUtility::dot_beta3(dt,u);

        Eigen::Matrix<T, 4, 1> r1 = QSUtility::r(b1,Phi1);
        Eigen::Matrix<T, 4, 1> r2 = QSUtility::r(b2,Phi2);
        Eigen::Matrix<T, 4, 1> r3 = QSUtility::r(b3,Phi3);

        Eigen::Matrix<T, 4, 1> q = quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;


        Eigen::Matrix<T, 4, 1> dot_q, part1, part2,part3;
        part1 = quatLeftComp(dr_dt(dot_b1,b1,Q0,Q1))*quatLeftComp(r2)*r3;
        part2 = quatLeftComp(r1)*quatLeftComp(dr_dt(dot_b2,b2,Q1,Q2))*r3;
        part3 = quatLeftComp(r1)*quatLeftComp(r2)*dr_dt(dot_b3,b3,Q2,Q3);

        dot_q = quatLeftComp(Q0)*(part1 + part2 + part3);

        Eigen::Matrix<T, 3, 1> omega_hat =  w(q,dot_q);

        Eigen::Map<Eigen::Matrix<T,3,1>> error(residuals);
        error = omega_hat - Eigen::Matrix<T, 3, 1>(omegaSample_);

        squareInformation_ = weightScale_*Eigen::Matrix<T,3,3>::Identity();
        error = squareInformation_*error;

        return true;
    }

    // Create  autodiff cost functions
    static ceres::CostFunction* Create(const double ts, const double& deltat,
                                       const Eigen::Vector3d& omegaSample,
                                       const double& weightScale)
    {
        return (new ceres::AutoDiffCostFunction<
                QuaternionOmegaSampleError, 3, 4, 4, 4, 4>(
                new QuaternionOmegaSampleError( ts,  deltat,
                                                 omegaSample,
                                                weightScale)));
    }

private:
    double deltaT_;
    Eigen::Vector3d omegaSample_;
    Eigen::Matrix3d squareInformation_;
    double  weightScale_;
};

#endif
