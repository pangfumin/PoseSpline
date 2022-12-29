#include "PoseSpline/OmegaExtrinsicTemperalError.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"


OmegaExtrinsicTemperalError::OmegaExtrinsicTemperalError():
        omega_meas_(Eigen::Matrix<double,3,1>::Zero()),
        Q_cw_(unitQuat<double>()),
        dotQ_cw_(unitQuat<double>()),
        dotdotQ_cw_(unitQuat<double>()){


}


OmegaExtrinsicTemperalError::OmegaExtrinsicTemperalError(const Eigen::Vector3d& omega_meas,
                                                         const Quaternion& Q_cw,
                                                         const Quaternion& dotQ_cw,
                                                         const Quaternion& dotdotQ_cw):
        omega_meas_(omega_meas),
        Q_cw_(Q_cw),
        dotQ_cw_(dotQ_cw),
        dotdotQ_cw_(dotdotQ_cw){


}

OmegaExtrinsicTemperalError::~OmegaExtrinsicTemperalError(){

}

bool OmegaExtrinsicTemperalError::Evaluate(double const* const* parameters,
                                          double* residuals,
                                          double** jacobians) const{

    return EvaluateWithMinimalJacobians(parameters,residuals,jacobians,NULL);

}

//todo: finish it
bool OmegaExtrinsicTemperalError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                  double* residuals,
                                  double** jacobians,
                                  double** jacobiansMinimal) const{
    Eigen::Map<Quaternion> param_ext_Q_cs(jacobians[0]);
    double param_delta_t = jacobians[1][0];

    Quaternion dotQinvQ = quatLeftComp(dotQ_cw_)*quatInv(Q_cw_);
    Eigen::Vector4d J_dotQinvQ_t = QSUtility::Jacobian_dotQinvQ_t(Q_cw_,dotQ_cw_,dotdotQ_cw_);
    Quaternion dotQinvQ_linearize = dotQinvQ + J_dotQinvQ_t*param_delta_t; // linearize
    Quaternion invQ_cs = quatInv<double>(param_ext_Q_cs);
    Quaternion invQ_cw = quatInv(Q_cw_);
    Eigen::Vector3d omega_hat = 2.0*(quatLeftComp(invQ_cs)*quatLeftComp(dotQinvQ_linearize)*param_ext_Q_cs).head(3);

    // For simplity, define error = \hat - \meas
    Eigen::Map<Eigen::Vector3d> error(residuals);
    error = omega_hat - omega_meas_;


    // Jacobians
    if(jacobians != NULL){


        if(jacobians[0] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> jacobian0(jacobians[0]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> jacobian0_min;

            jacobian0_min = QSUtility::Jacobian_omega_extrinsicQ(Q_cw_,dotQ_cw_,param_ext_Q_cs);

            QuaternionLocalParameter localPrameter;
            Eigen::Matrix<double,3,4> lift;
            localPrameter.ComputeLiftJacobian(parameters[0],lift.data());
            jacobian0 = jacobian0_min*lift;


            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){

            }


        }
        if(jacobians[1] != NULL){

            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){

            }

        }
    }


    return true;

}
