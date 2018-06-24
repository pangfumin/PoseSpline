#include "extern/pinhole_project_error.h"
#include "pose-spline/QuaternionSplineUtility.hpp"
#include "pose-spline/Quaternion.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
PinholeProjectError::PinholeProjectError(const Eigen::Vector3d& uv_C0,
                                           const Eigen::Vector3d& uv_C1,
                                           const Eigen::Isometry3d _T_IC):
        C0uv(uv_C0),
        C1uv(uv_C1){
     t_IC = _T_IC.matrix().topRightCorner(3,1);
     R_IC = _T_IC.matrix().topLeftCorner(3,3);
}


bool PinholeProjectError::Evaluate(double const *const *parameters,
                                    double *residuals,
                                    double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}


bool PinholeProjectError::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                        double *residuals,
                                                        double **jacobians,
                                                        double **jacobiansMinimal) const {

    // T_WI0
    Eigen::Vector3d t_WI0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaternion Q_WI0( parameters[0][3], parameters[0][4], parameters[0][5], parameters[0][6]);

    // T_WI1
    Eigen::Vector3d t_WI1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaternion Q_WI1( parameters[1][3], parameters[1][4], parameters[1][5], parameters[1][6]);

    // rho
    double inv_dep = parameters[2][0];


    Eigen::Matrix3d R_WI0 = quatToRotMat(Q_WI0);
    Eigen::Matrix3d R_WI1 = quatToRotMat(Q_WI1);

    Eigen::Vector3d C0p = C0uv / inv_dep;
    Eigen::Vector3d I0p = R_IC * C0p + t_IC;
    Eigen::Vector3d Wp = R_WI0 * I0p + t_WI0;
    Eigen::Vector3d I1p = R_WI1.inverse() * (Wp - t_WI1);
    Eigen::Vector3d C1p = R_IC.inverse() * (I1p - t_IC);

    Eigen::Matrix<double, 2, 1> error;

    double inv_z = 1/C1p(2);
    Eigen::Vector2d hat_C1uv(C1p(0)*inv_z, C1p(1)*inv_z);

    Eigen::Matrix<double,2,3> H;
    H << 1, 0, -C1p(0)*inv_z,
            0, 1, -C1p(1)*inv_z;
    H *= inv_z;

    error = hat_C1uv - C1uv.head<2>();
    squareRootInformation_.setIdentity();
    //squareRootInformation_ = weightScalar_* squareRootInformation_; //Weighted

    // weight it
    Eigen::Map<Eigen::Matrix<double, 2, 1> > weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;




    // calculate jacobians
    if(jacobians != NULL){
        if(jacobians[0] != NULL){
            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian0(jacobians[0]);

            Eigen::Matrix<double, 3, 6> tmp;
            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = R_IC.transpose()*R_WI1.transpose();
            tmp.topRightCorner(3,3) =  R_IC.transpose()*R_WI1.transpose()*crossMat<double>(R_WI0*I0p);

            jacobian0_min  =  H*tmp;

            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
            PoseLocalParameter::liftJacobian(parameters[0], lift.data());
            jacobian0 = squareRootInformation_*jacobian0_min*lift;



            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian0_min(jacobiansMinimal[0]);
                map_jacobian0_min = squareRootInformation_*jacobian0_min;
            }
        }

        if(jacobians[1] != NULL){
            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian1(jacobians[1]);

            Eigen::Matrix<double, 3, 6> tmp;

            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = -R_IC.transpose()*R_WI1.transpose();
            tmp.bottomRightCorner(3,3) =  - R_IC.transpose() * R_WI1.transpose() * crossMat<double>(Wp - t_WI1);


            jacobian1_min = H*tmp;
            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
            PoseLocalParameter::liftJacobian(parameters[1], lift.data());
            jacobian1 = squareRootInformation_*jacobian1_min*lift;

            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian1_min(jacobiansMinimal[1]);
                map_jacobian1_min = squareRootInformation_*jacobian1_min;
            }
        }
        if(jacobians[2] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,1>> jacobian2(jacobians[2]);
            jacobian2 = - H*R_IC.transpose()*R_WI1.transpose()*R_WI0*R_IC*C0uv/(inv_dep*inv_dep);
            jacobian2 = squareRootInformation_*jacobian2;

            if(jacobiansMinimal != NULL && jacobiansMinimal[2] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,1>> map_jacobian2_min(jacobiansMinimal[2]);
                map_jacobian2_min = squareRootInformation_*jacobian2;
            }
        }
    }
    return true;
}