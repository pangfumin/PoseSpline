#ifndef  ANGULAR_VELOCITY_SAMPLE_ERROR
#define  ANGULAR_VELOCITY_SAMPLE_ERROR

#include <ceres/ceres.h>
#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/QuaternionOmegaSampleError.hpp"

class AngularVelocitySampleAutoError : public ceres::SizedCostFunction<3,
        7, 7, 7, 7,3,3,3,3> {
public:
    explicit AngularVelocitySampleAutoError(QuaternionOmegaSampleFunctor* functor)
            : functor_(functor) {

    }

    virtual ~AngularVelocitySampleAutoError() {}

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
