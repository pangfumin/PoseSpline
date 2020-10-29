
#ifndef PROJECT_FACTOR_H
#define PROJECT_FACTOR_H
#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

/**
 *
 */
class ProjectError:public ceres::SizedCostFunction<2, /* num of residual */
        3>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief The number of residuals
    static const int kNumResiduals = 2;

    /// \brief The type of the covariance.
    typedef Eigen::Matrix<double, 2, 2> covariance_t;

    /// \brief The type of the information (same matrix dimension as covariance).
    typedef covariance_t information_t;



    ProjectError() = delete;
    ProjectError(const Eigen::Vector3d& uv_C0, const Eigen::Vector3d& pt3d, const Eigen::Quaterniond& Q_WC);

    /// \brief Trivial destructor.
    virtual ~ProjectError() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:

    Eigen::Vector3d C0uv_, Wp_;
    Eigen::Quaterniond Q_WC_;

    // information matrix and its square root
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif


/*
 *
 */