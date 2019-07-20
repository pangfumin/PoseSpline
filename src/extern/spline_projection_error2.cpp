#include "extern/spline_projection_error2.h"
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"

SplineProjectError2::SplineProjectError2(const SplineProjectFunctor2& functor):
functor_(functor){}


bool SplineProjectError2::Evaluate(double const *const *parameters,
                                   double *residuals,
                                   double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}


bool SplineProjectError2::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                       double *residuals,
                                                       double **jacobians,
                                                       double **jacobiansMinimal) const {

    if (!jacobians) {
        return ceres::internal::VariadicEvaluate<
                SplineProjectFunctor2, double, 7, 7, 7, 7, 7, 7, 1,0,0,0>
        ::Call(functor_, parameters, residuals);
    }




    bool success =  ceres::internal::AutoDiff<SplineProjectFunctor2, double,
            7,7,7,7,7,7,1>::Differentiate(
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
        if( jacobiansMinimal[0] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J0_minimal_map(jacobiansMinimal[0]);
            Eigen::Matrix<double,7,6,Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(T0.parameterPtr(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J0_map(jacobians[0]);
            J0_minimal_map = J0_map*J_plus;
        }

        if( jacobiansMinimal[1] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J1_minimal_map(jacobiansMinimal[1]);
            Eigen::Matrix<double,7,6,Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(T1.parameterPtr(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J1_map(jacobians[1]);
            J1_minimal_map = J1_map*J_plus;
        }

        if( jacobiansMinimal[2] != NULL){

            Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J2_minimal_map(jacobiansMinimal[2]);
            Eigen::Matrix<double,7,6,Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(T2.parameterPtr(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J2_map(jacobians[2]);
            J2_minimal_map = J2_map*J_plus;
        }

        if( jacobiansMinimal[3] != NULL){

            Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J3_minimal_map(jacobiansMinimal[3]);
            Eigen::Matrix<double,7,6,Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(T3.parameterPtr(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J3_map(jacobians[3]);
            J3_minimal_map = J3_map*J_plus;
        }

        if( jacobiansMinimal[4] != NULL){

            Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J4_minimal_map(jacobiansMinimal[4]);
            Eigen::Matrix<double,7,6,Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(T4.parameterPtr(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J4_map(jacobians[4]);
            J4_minimal_map = J4_map*J_plus;
        }

        if( jacobiansMinimal[5] != NULL){

            Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J5_minimal_map(jacobiansMinimal[5]);
            Eigen::Matrix<double,7,6,Eigen::RowMajor> J_plus;
            PoseLocalParameter::plusJacobian(T5.parameterPtr(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J5_map(jacobians[5]);
            J5_minimal_map = J5_map*J_plus;
        }

        if( jacobiansMinimal[6] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,1>> J6_minimal_map(jacobiansMinimal[6]);
            Eigen::Map<Eigen::Matrix<double,2,1>> J6_map(jacobians[6]);
            J6_minimal_map = J6_map;
        }

    }

    return true;
}