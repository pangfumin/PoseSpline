#include "project_error.h"
#include "pose_local_parameterization.h"


ProjectError::ProjectError(const Eigen::Vector2d& uv, const Eigen::Vector3d& pt3d,
        const double cx, const double cy, double focal):
        uv_(uv), pt3d_(pt3d), cx_(cx), cy_(cy), focal_(focal){

}
bool ProjectError::Evaluate(double const *const *parameters,
                                    double *residuals,
                                    double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}
bool ProjectError::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                        double *residuals,
                                                        double **jacobians,
                                                        double **jacobiansMinimal) const {

    // T_WC
    Eigen::Vector3d t(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d Cp =  pt3d_ + t;
    Eigen::Matrix<double, 2, 1> error;
    double inv_z = 1/Cp(2);
    Eigen::Vector2d hat_C0uv(focal_ * Cp(0)*inv_z + cx_, focal_ * Cp(1)*inv_z + cy_);
    Eigen::Matrix<double,2,3> H;
    H << 1, 0, -Cp(0)*inv_z,
            0, 1, -Cp(1)*inv_z;
    H *= focal_ * inv_z;
    error = hat_C0uv - uv_;
    // weight it
    Eigen::Map<Eigen::Matrix<double, 2, 1> > weighted_error(residuals);
    weighted_error =  error;
    // calculate jacobians
    if(jacobians != NULL){
        if(jacobians[0] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>> jacobian0(jacobians[0]);
            jacobian0  =  H;

            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>> map_jacobian0_min(jacobiansMinimal[0]);
                map_jacobian0_min = jacobian0;
            }
        }
    }
    return true;
}