
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



    ProjectError(const Eigen::Vector2d& uv, const Eigen::Vector3d& pt3d, const double cx, const double cy, double focal);

    /// \brief Trivial destructor.
    virtual ~ProjectError() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:

    Eigen::Vector2d uv_;
    Eigen::Vector3d pt3d_;
    double cx_, cy_, focal_;

};

#endif


/*
 *
 */