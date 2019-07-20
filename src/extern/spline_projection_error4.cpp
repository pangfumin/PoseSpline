#include "extern/spline_projection_error4.h"
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"

SplineProjectError4::SplineProjectError4(const SplineProjectFunctor4& functor):
functor_(functor){}


bool SplineProjectError4::Evaluate(double const *const *parameters,
                                   double *residuals,
                                   double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}


bool SplineProjectError4::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                       double *residuals,
                                                       double **jacobians,
                                                       double **jacobiansMinimal) const {

    if (!jacobians) {
        return ceres::internal::VariadicEvaluate<
                SplineProjectFunctor4, double, 7, 7, 7, 7, 7, 7, 7, 7, 1,0>
        ::Call(functor_, parameters, residuals);
    }




    bool success =  ceres::internal::AutoDiff<SplineProjectFunctor4, double,
            7,7,7,7,7,7,7,7,1>::Differentiate(
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
        Pose<double> T6(parameters[6]);
        Pose<double> T7(parameters[7]);
        for (int i = 0; i < 8 && jacobiansMinimal[i] != NULL; i++) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_minimal_map(jacobiansMinimal[i]);
            Eigen::Matrix<double, 7, 6, Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(Pose<double>(parameters[i]).parameterPtr(), J_plus.data());
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_map(jacobians[i]);
            J_minimal_map = J_map * J_plus;

        }

        if( jacobiansMinimal[8] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,1>> J8_minimal_map(jacobiansMinimal[8]);
            Eigen::Map<Eigen::Matrix<double,2,1>> J8_map(jacobians[8]);
            J8_minimal_map = J8_map;
        }

    }

    return true;
}