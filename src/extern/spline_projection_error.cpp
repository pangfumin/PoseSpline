#include "extern/spline_projection_error.h"
#include "pose-spline/PoseSplineUtility.hpp"
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "geometry/Pose.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
SplineProjectError::SplineProjectError(const double _t0, const Eigen::Vector3d& uv_C0,
                                        const double _t1, const Eigen::Vector3d& uv_C1,
                                        const Eigen::Isometry3d _T_IC):
t0(_t0), t1(_t1),
        C0uv(uv_C0),
        C1uv(uv_C1){
    t_IC = _T_IC.matrix().topRightCorner(3,1);
    R_IC = _T_IC.matrix().topLeftCorner(3,3);
}


bool SplineProjectError::Evaluate(double const *const *parameters,
                                   double *residuals,
                                   double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}


bool SplineProjectError::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                       double *residuals,
                                                       double **jacobians,
                                                       double **jacobiansMinimal) const {

    Pose<double> T0, T1, T2, T3;
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> map_T0(parameters[0]);
    T0.setParameters(map_T0);

    Eigen::Map<const Eigen::Matrix<double, 7, 1>> map_T1(parameters[1]);
    T1.setParameters(map_T1);

    Eigen::Map<const Eigen::Matrix<double, 7, 1>> map_T2(parameters[2]);
    T2.setParameters(map_T2);

    Eigen::Map<const Eigen::Matrix<double, 7, 1>> map_T3(parameters[3]);
    T0.setParameters(map_T3);


    // rho
    double inv_dep = parameters[4][0];

    PoseSplineEvaluation psv0,psv1;
    Pose<double> T_WI0 = psv0(t0, T0, T1, T2, T3);
    Pose<double> T_WI1 = psv1(t1, T0, T1, T2, T3);

    Quaternion Q0 = psv0.getRotation();
    Quaternion Q1 = psv1.getRotation();

    Eigen::Matrix3d R_WI0 = quatToRotMat(Q0);
    Eigen::Matrix3d R_WI1 = quatToRotMat(Q1);

    Eigen::Vector3d C0p = C0uv / inv_dep;
    Eigen::Vector3d I0p = R_IC * C0p + t_IC;
    Eigen::Vector3d Wp = R_WI0 * I0p + psv0.getTranslation();
    Eigen::Vector3d I1p = R_WI1.inverse() * (Wp - psv1.getTranslation());
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

        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> lift0, lift1;
        QuaternionLocalParameter::liftJacobian(Q0.data(), lift0.data());
        QuaternionLocalParameter::liftJacobian(Q1.data(), lift1.data());

        Eigen::Matrix3d R_C1W = R_IC.transpose()*R_WI1.transpose();

        Eigen::Matrix<double,3,4> J_p_Qspline0 = R_IC.transpose()*R_WI1.transpose()*crossMat<double>(R_WI0*I0p) * lift0;
        Eigen::Matrix<double,3,4> J_p_Qspline1
                = - R_IC.transpose() * R_WI1.transpose() * crossMat<double>(Wp - psv1.getTranslation()) * lift1;

        if(jacobians[0] != NULL){
            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian0(jacobians[0]);

            Eigen::Matrix<double, 3, 6> tmp;
            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = R_C1W
                                     *(psv0.getTranslationJacobian<0>() - psv1.getTranslationJacobian<0>());
            tmp.topRightCorner(3,3)
                    = J_p_Qspline0*psv0.getRotationJacobianMinimal<0>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<0>();

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
            tmp.topLeftCorner(3,3) = R_C1W
                                     *(psv0.getTranslationJacobian<1>() - psv1.getTranslationJacobian<1>());
            tmp.bottomRightCorner(3,3)
                    = J_p_Qspline0*psv0.getRotationJacobianMinimal<1>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<1>();


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
            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian2(jacobians[2]);

            Eigen::Matrix<double, 3, 6> tmp;

            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = R_C1W
                                     *(psv0.getTranslationJacobian<2>() - psv1.getTranslationJacobian<2>());
            tmp.bottomRightCorner(3,3)
                    =  J_p_Qspline0*psv0.getRotationJacobianMinimal<2>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<2>();


            jacobian2_min = H*tmp;
            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
            PoseLocalParameter::liftJacobian(parameters[1], lift.data());
            jacobian2 = squareRootInformation_*jacobian2_min*lift;

            if(jacobiansMinimal != NULL && jacobiansMinimal[2] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian2_min(jacobiansMinimal[2]);
                map_jacobian2_min = squareRootInformation_*jacobian2_min;
            }
        }

        if(jacobians[3] != NULL){
            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian3(jacobians[1]);

            Eigen::Matrix<double, 3, 6> tmp;

            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = R_C1W
                                     *(psv0.getTranslationJacobian<3>() - psv1.getTranslationJacobian<3>());
            tmp.bottomRightCorner(3,3)
                    =  J_p_Qspline0*psv0.getRotationJacobianMinimal<3>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<3>();


            jacobian3_min = H*tmp;
            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
            PoseLocalParameter::liftJacobian(parameters[1], lift.data());
            jacobian3 = squareRootInformation_*jacobian3_min*lift;

            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian3_min(jacobiansMinimal[3]);
                map_jacobian3_min = squareRootInformation_*jacobian3_min;
            }
        }

        if(jacobians[4] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,1>> jacobian4(jacobians[4]);
            jacobian4 = - H*R_IC.transpose()*R_WI1.transpose()*R_WI0*R_IC*C0uv/(inv_dep*inv_dep);
            jacobian4 = squareRootInformation_*jacobian4;

            if(jacobiansMinimal != NULL && jacobiansMinimal[4] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,1>> map_jacobian4_min(jacobiansMinimal[4]);
                map_jacobian4_min = squareRootInformation_*jacobian4;
            }
        }
    }
    return true;
}