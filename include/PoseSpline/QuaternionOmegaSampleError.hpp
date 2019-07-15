#ifndef  QUATERNION_OMEGA_SAMPLE_ERROR
#define  QUATERNION_OMEGA_SAMPLE_ERROR

#include <ceres/ceres.h>
#include "PoseSpline/QuaternionSplineUtility.hpp"
struct QuaternionOmegaSampleFunctor{
    QuaternionOmegaSampleFunctor(const double u, const double& deltat,
                               const Eigen::Vector3d& omegaSample,
                                const double& weightScale)
            : u_(u),
              deltaT_(deltat),
              omegaSample_(omegaSample),
              weightScale_(weightScale){
    }

    template <typename  T>
    bool operator()(const T* const Quternion0, const T* const Quternion1,
                    const T* const Quternion2, const T* const Quternion3, T* residuals) const
    {

        Eigen::Matrix<T, 4, 1> Q0(Quternion0[0], Quternion0[1],Quternion0[2],Quternion0[3]);
        Eigen::Matrix<T, 4, 1> Q1(Quternion1[0], Quternion1[1],Quternion1[2],Quternion1[3]);
        Eigen::Matrix<T, 4, 1> Q2(Quternion2[0], Quternion2[1],Quternion2[2],Quternion2[3]);
        Eigen::Matrix<T, 4, 1> Q3(Quternion3[0], Quternion3[1],Quternion3[2],Quternion3[3]);

        Eigen::Matrix<T,3,1> Phi1 = QSUtility::Phi(Q0,Q1);
        Eigen::Matrix<T,3,1> Phi2 = QSUtility::Phi(Q1,Q2);
        Eigen::Matrix<T,3,1> Phi3 = QSUtility::Phi(Q2,Q3);

        T u = T(u_);
        T b1 = QSUtility::beta1(u);
        T b2 = QSUtility::beta2(u);
        T b3 = QSUtility::beta3(u);


        T dot_b1 = QSUtility::dot_beta1(T(deltaT_),u);
        T dot_b2 = QSUtility::dot_beta2(T(deltaT_),u);
        T dot_b3 = QSUtility::dot_beta3(T(deltaT_),u);

        Eigen::Matrix<T, 4, 1> r1 = QSUtility::r(b1,Phi1);
        Eigen::Matrix<T, 4, 1> r2 = QSUtility::r(b2,Phi2);
        Eigen::Matrix<T, 4, 1> r3 = QSUtility::r(b3,Phi3);

        Eigen::Matrix<T, 4, 1> q = quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;


        Eigen::Matrix<T, 4, 1> dot_q, part1, part2,part3;
        part1 = quatLeftComp(QSUtility::dr_dt(dot_b1,b1,Q0,Q1))*quatLeftComp(r2)*r3;
        part2 = quatLeftComp(r1)*quatLeftComp(QSUtility::dr_dt(dot_b2,b2,Q1,Q2))*r3;
        part3 = quatLeftComp(r1)*quatLeftComp(r2)*QSUtility::dr_dt(dot_b3,b3,Q2,Q3);

        dot_q = quatLeftComp(Q0)*(part1 + part2 + part3);

        Eigen::Matrix<T, 3, 1> omega_hat =  QSUtility::w_in_body_frame(q,dot_q);

        Eigen::Map<Eigen::Matrix<T,3,1>> error(residuals);
        //error = omega_hat - Eigen::Matrix<T, 3, 1>(omegaSample_);
        error(0) = omega_hat(0) - T(omegaSample_(0));
        error(1) = omega_hat(1) - T(omegaSample_(1));
        error(2) = omega_hat(2) - T(omegaSample_(2));


        Eigen::Matrix<T,3,3> squareInformation_ = (T)weightScale_*Eigen::Matrix<T,3,3>::Identity();
        error = squareInformation_*error;

        return true;
    }


    double getU() {return u_;};
    double getDeltaT() {return deltaT_;};
    static double num_residuals() {return 3;};

private:
    double u_;
    double deltaT_;
    Eigen::Vector3d omegaSample_;

    double  weightScale_;
};



class QuaternionOmegaSampleAutoError : public ceres::SizedCostFunction<3,
        4, 4, 4, 4> {
public:
    // Takes ownership of functor. Uses the template-provided value for the
    // number of residuals ("kNumResiduals").
    explicit QuaternionOmegaSampleAutoError(QuaternionOmegaSampleFunctor* functor)
            : functor_(functor) {

    }



    virtual ~QuaternionOmegaSampleAutoError() {}

    // Implementation details follow; clients of the autodiff cost function should
    // not have to examine below here.
    //
    // To handle varardic cost functions, some template magic is needed. It's
    // mostly hidden inside autodiff.h.
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const;

private:
    QuaternionOmegaSampleFunctor* functor_;
};

#endif
