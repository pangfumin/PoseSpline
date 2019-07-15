#include "PoseSpline/QuaternionOmegaSampleError.hpp"
#include "PoseSpline/QuaternionLocalParameter.hpp"

bool QuaternionOmegaSampleAutoError::Evaluate(double const* const* parameters,
              double* residuals,
              double** jacobians) const {

    return EvaluateWithMinimalJacobians(parameters, residuals,jacobians, NULL);

}

bool QuaternionOmegaSampleAutoError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                  double* residuals,
                                  double** jacobians,
                                  double** jacobiansMinimal) const {

    Eigen::Map<const Quaternion> Q0(parameters[0]);
    Eigen::Map<const Quaternion> Q1(parameters[1]);
    Eigen::Map<const Quaternion> Q2(parameters[2]);
    Eigen::Map<const Quaternion> Q3(parameters[3]);


    if (!jacobians) {
        return ceres::internal::VariadicEvaluate<
                QuaternionOmegaSampleFunctor, double, 4, 4, 4, 4, 0,0,0,0,0,0>
        ::Call(*functor_, parameters, residuals);
    }
    bool success =  ceres::internal::AutoDiff<QuaternionOmegaSampleFunctor, double,
            4,4,4,4>::Differentiate(
            *functor_,
            parameters,
            SizedCostFunction<3,
                    4,4,4,4>::num_residuals(),
            residuals,
            jacobians);

    if (success && jacobiansMinimal!= NULL) {
        if( jacobiansMinimal[0] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0_minimal_map(jacobiansMinimal[0]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q0.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J0_map(jacobians[0]);
            J0_minimal_map = J0_map*J_plus;
        }

        if( jacobiansMinimal[1] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1_minimal_map(jacobiansMinimal[1]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q1.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J1_map(jacobians[1]);
            J1_minimal_map = J1_map*J_plus;
        }

        if( jacobiansMinimal[2] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J2_minimal_map(jacobiansMinimal[2]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q2.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J2_map(jacobians[2]);
            J2_minimal_map = J2_map*J_plus;
        }

        if( jacobiansMinimal[3] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J3_minimal_map(jacobiansMinimal[3]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q3.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J3_map(jacobians[3]);
            J3_minimal_map = J3_map*J_plus;
        }

    }

    return success;
}