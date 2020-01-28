//
// Created by root on 1/25/20.
//

#ifndef POSESPLINE_OPTIMIZATIONCHECKER_H
#define POSESPLINE_OPTIMIZATIONCHECKER_H
#include <ceres/ceres.h>
#include <vector>
namespace ceres {
    class OptimizationChecker {
    public:
        OptimizationChecker(ceres::CostFunction *function,
                            const std::vector<ceres::LocalParameterization *> *local_parameterizations,
                            ceres::LossFunction* loss_function);

        // Contains results from a call to Probe for later inspection.
        struct CERES_EXPORT ProbeResults {
            // The return value of the cost function.
            bool return_value;

            // Computed residual vector.
            Vector residuals;

            // The sizes of the Jacobians below are dictated by the cost function's
            // parameter block size and residual block sizes. If a parameter block
            // has a local parameterization associated with it, the size of the "local"
            // Jacobian will be determined by the local parameterization dimension and
            // residual block size, otherwise it will be identical to the regular
            // Jacobian.

            // Derivatives as computed by the cost function.
            std::vector<Matrix> jacobians;

            // Derivatives as computed by the cost function in local space.
            std::vector<Matrix> local_jacobians;

            // Derivatives as computed by nuerical differentiation in local space.
            std::vector<Matrix> numeric_jacobians;

            // Derivatives as computed by nuerical differentiation in local space.
            std::vector<Matrix> local_numeric_jacobians;

            // Contains the maximum relative error found in the local Jacobians.
            double maximum_relative_error;

            // If an error was detected, this will contain a detailed description of
            // that error.
            std::string error_log;
        };

        // Call the cost function, compute alternative Jacobians using finite
        // differencing and compare results. If local parameterizations are given,
        // the Jacobians will be multiplied by the local parameterization Jacobians
        // before performing the check, which effectively means that all errors along
        // the null space of the local parameterization will be ignored.
        // Returns false if the Jacobians don't match, the cost function return false,
        // or if the cost function returns different residual when called with a
        // Jacobian output argument vs. calling it without. Otherwise returns true.
        //
        // parameters: The parameter values at which to probe.
        // relative_precision: A threshold for the relative difference between the
        // Jacobians. If the Jacobians differ by more than this amount, then the
        // probe fails.
        // results: On return, the Jacobians (and other information) will be stored
        // here. May be NULL.
        //
        // Returns true if no problems are detected and the difference between the
        // Jacobians is less than error_tolerance.
        bool Probe(const std::vector<const double* > parameters,
                   double relative_precision,
                   const ceres::Solver::Options& options,
                   ProbeResults* results) const;

    private:
        bool optimizeAtOneParameter(const std::vector<Eigen::VectorXd> eigen_parameters,
                const int& ith,
                const ceres::Solver::Options& options,
                const double threshold) const ;

        void perturbParameter(double * param,
                               ceres::LocalParameterization* localParameterization) const;

        void perturbParameter(double * param,
                                                   const int size) const ;

        ceres::CostFunction *function_;
        const std::vector<ceres::LocalParameterization *> *local_parameterizations_;
         ceres::LossFunction* loss_function_;

    };
}
#endif //POSESPLINE_OPTIMIZATIONCHECKER_H
