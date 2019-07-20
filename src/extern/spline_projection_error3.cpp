#include "extern/spline_projection_error3.h"
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"

SplineProjectError3::SplineProjectError3(const SplineProjectFunctor3& functor):
functor_(functor){}


bool SplineProjectError3::Evaluate(double const *const *parameters,
                                   double *residuals,
                                   double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}


bool SplineProjectError3::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                       double *residuals,
                                                       double **jacobians,
                                                       double **jacobiansMinimal) const {

    if (!jacobians) {
        return ceres::internal::VariadicEvaluate<
                SplineProjectFunctor3, double, 7, 7, 7, 7, 7, 7, 7, 1,0,0>
        ::Call(functor_, parameters, residuals);
    }




    bool success =  ceres::internal::AutoDiff<SplineProjectFunctor3, double,
            7,7,7,7,7,7,7,1>::Differentiate(
            functor_,
            parameters,
            2,
            residuals,
            jacobians);

    if (success && jacobiansMinimal!= NULL) {
        Pose<double> T0(parameters[0]);
        Pose<double> T1(parameters[1]);
        Pose<double> T2(parameters[2]);
        Pose<double> T3(parameters[3]);
        Pose<double> T4(parameters[4]);
        Pose<double> T5(parameters[5]);
        for (int i = 0; i < 7 && jacobiansMinimal[i] != NULL; i++) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_minimal_map(jacobiansMinimal[i]);
            Eigen::Matrix<double, 7, 6, Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(Pose<double>(parameters[i]).parameterPtr(), J_plus.data());
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_map(jacobians[i]);
            J_minimal_map = J_map * J_plus;

        }

        if( jacobiansMinimal[7] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,1>> J7_minimal_map(jacobiansMinimal[7]);
            Eigen::Map<Eigen::Matrix<double,2,1>> J7_map(jacobians[7]);
            J7_minimal_map = J7_map;
        }

    }

    return true;
}