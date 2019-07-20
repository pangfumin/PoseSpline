#include "extern/spline_projection_error.h"
#include "PoseSpline/PoseSplineUtility.hpp"
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
SplineProjectError::SplineProjectError(const double _t0, const Eigen::Vector3d& uv_C0,
                                        const double _t1, const Eigen::Vector3d& uv_C1,
                                        const Eigen::Isometry3d _T_IC):
t0_(_t0), t1_(_t1),
        C0uv_(uv_C0),
        C1uv_(uv_C1){
    T_IC_ = _T_IC;
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
    T3.setParameters(map_T3);

    Quaternion Q0 = T0.rotation();
    Quaternion Q1 = T1.rotation();
    Quaternion Q2 = T2.rotation();
    Quaternion Q3 = T3.rotation();

    Eigen::Vector3d t0 = T0.translation();
    Eigen::Vector3d t1 = T1.translation();
    Eigen::Vector3d t2 = T2.translation();
    Eigen::Vector3d t3 = T3.translation();

    // rho
    double inv_dep = parameters[4][0];


    double  Beta01 = QSUtility::beta1(t0_);
    double  Beta02 = QSUtility::beta2(t0_);
    double  Beta03 = QSUtility::beta3(t0_);

    double  Beta11 = QSUtility::beta1(t1_);
    double  Beta12 = QSUtility::beta2(t1_);
    double  Beta13 = QSUtility::beta3(t1_);

    Eigen::Vector3d phi1 = QSUtility::Phi<double>(Q0,Q1);
    Eigen::Vector3d phi2 = QSUtility::Phi<double>(Q1,Q2);
    Eigen::Vector3d phi3 = QSUtility::Phi<double>(Q2,Q3);

    Quaternion r_01 = QSUtility::r(Beta01,phi1);
    Quaternion r_02 = QSUtility::r(Beta02,phi2);
    Quaternion r_03 = QSUtility::r(Beta03,phi3);

    Quaternion r_11 = QSUtility::r(Beta11,phi1);
    Quaternion r_12 = QSUtility::r(Beta12,phi2);
    Quaternion r_13 = QSUtility::r(Beta13,phi3);

    // define residual
    // For simplity, we define error  =  /hat - meas.
    Quaternion Q_WI0_hat = quatLeftComp<double>(Q0)*quatLeftComp(r_01)*quatLeftComp(r_02)*r_03;
    Eigen::Vector3d t_WI0_hat = t0 + Beta01*(t1 - t0) +  Beta02*(t2 - t1) + Beta03*(t3 - t2);

    Quaternion Q_WI1_hat = quatLeftComp<double>(Q0)*quatLeftComp(r_11)*quatLeftComp(r_12)*r_13;
    Eigen::Vector3d t_WI1_hat = t0 + Beta11*(t1 - t0) +  Beta12*(t2 - t1) + Beta13*(t3 - t2);

    Eigen::Matrix3d R_WI0 = quatToRotMat(Q_WI0_hat);
    Eigen::Matrix3d R_WI1 = quatToRotMat(Q_WI1_hat);

    Eigen::Matrix3d R_IC = T_IC_.matrix().topLeftCorner(3,3);
    Eigen::Vector3d t_IC = T_IC_.matrix().topRightCorner(3,1);
//
    Eigen::Vector3d C0p = C0uv_ / inv_dep;
    Eigen::Vector3d I0p = R_IC * C0p + t_IC;
    Eigen::Vector3d Wp = R_WI0 * I0p + t_WI0_hat;
    Eigen::Vector3d I1p = R_WI1.inverse() * (Wp - t_WI1_hat);
    Eigen::Vector3d C1p = R_IC.inverse() * (I1p - t_IC);


    Eigen::Matrix<double, 2, 1> error;

    double inv_z = 1/C1p(2);
    Eigen::Vector2d hat_C1uv(C1p(0)*inv_z, C1p(1)*inv_z);

    Eigen::Matrix<double,2,3> H;
    H << 1, 0, -C1p(0)*inv_z,
            0, 1, -C1p(1)*inv_z;
    H *= inv_z;

    error = hat_C1uv - C1uv_.head<2>();
    squareRootInformation_.setIdentity();
    //squareRootInformation_ = weightScalar_* squareRootInformation_; //Weighted

    // weight it
    Eigen::Map<Eigen::Matrix<double, 2, 1> > weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;

//    // calculate jacobians
//    if(jacobians != NULL){
//
//        Eigen::Matrix<double, 3, 4, Eigen::RowMajor> lift0, lift1;
//        QuaternionLocalParameter::liftJacobian(Q0.data(), lift0.data());
//        QuaternionLocalParameter::liftJacobian(Q1.data(), lift1.data());
//
//        Eigen::Matrix3d R_C1W = R_IC.transpose()*R_WI1.transpose();
//
//        Eigen::Matrix<double,3,4> J_p_Qspline0 = R_IC.transpose()*R_WI1.transpose()*crossMat<double>(R_WI0*I0p) * lift0;
//        Eigen::Matrix<double,3,4> J_p_Qspline1
//                = - R_IC.transpose() * R_WI1.transpose() * crossMat<double>(Wp - psv1.getTranslation()) * lift1;
//
//        if(jacobians[0] != NULL){
//            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
//            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian0(jacobians[0]);
//
//            Eigen::Matrix<double, 3, 6> tmp;
//            tmp.setIdentity();
//            tmp.topLeftCorner(3,3) = R_C1W
//                                     *(psv0.getTranslationJacobian<0>() - psv1.getTranslationJacobian<0>());
//            tmp.topRightCorner(3,3)
//                    = J_p_Qspline0*psv0.getRotationJacobianMinimal<0>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<0>();
//
//            jacobian0_min  =  H*tmp;
//
//            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
//            PoseLocalParameter::liftJacobian(parameters[0], lift.data());
//            jacobian0 = squareRootInformation_*jacobian0_min*lift;
//
//            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){
//                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian0_min(jacobiansMinimal[0]);
//                map_jacobian0_min = squareRootInformation_*jacobian0_min;
//            }
//        }
//
//        if(jacobians[1] != NULL){
//            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
//            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian1(jacobians[1]);
//
//            Eigen::Matrix<double, 3, 6> tmp;
//
//            tmp.setIdentity();
//            tmp.topLeftCorner(3,3) = R_C1W
//                                     *(psv0.getTranslationJacobian<1>() - psv1.getTranslationJacobian<1>());
//            tmp.bottomRightCorner(3,3)
//                    = J_p_Qspline0*psv0.getRotationJacobianMinimal<1>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<1>();
//
//
//            jacobian1_min = H*tmp;
//            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
//            PoseLocalParameter::liftJacobian(parameters[1], lift.data());
//            jacobian1 = squareRootInformation_*jacobian1_min*lift;
//
//            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){
//                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian1_min(jacobiansMinimal[1]);
//                map_jacobian1_min = squareRootInformation_*jacobian1_min;
//            }
//        }
//
//        if(jacobians[2] != NULL){
//            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
//            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian2(jacobians[2]);
//
//            Eigen::Matrix<double, 3, 6> tmp;
//
//            tmp.setIdentity();
//            tmp.topLeftCorner(3,3) = R_C1W
//                                     *(psv0.getTranslationJacobian<2>() - psv1.getTranslationJacobian<2>());
//            tmp.bottomRightCorner(3,3)
//                    =  J_p_Qspline0*psv0.getRotationJacobianMinimal<2>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<2>();
//
//
//            jacobian2_min = H*tmp;
//            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
//            PoseLocalParameter::liftJacobian(parameters[1], lift.data());
//            jacobian2 = squareRootInformation_*jacobian2_min*lift;
//
//            if(jacobiansMinimal != NULL && jacobiansMinimal[2] != NULL){
//                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian2_min(jacobiansMinimal[2]);
//                map_jacobian2_min = squareRootInformation_*jacobian2_min;
//            }
//        }
//
//        if(jacobians[3] != NULL){
//            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
//            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian3(jacobians[1]);
//
//            Eigen::Matrix<double, 3, 6> tmp;
//
//            tmp.setIdentity();
//            tmp.topLeftCorner(3,3) = R_C1W
//                                     *(psv0.getTranslationJacobian<3>() - psv1.getTranslationJacobian<3>());
//            tmp.bottomRightCorner(3,3)
//                    =  J_p_Qspline0*psv0.getRotationJacobianMinimal<3>() + J_p_Qspline1*psv0.getRotationJacobianMinimal<3>();
//
//
//            jacobian3_min = H*tmp;
//            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
//            PoseLocalParameter::liftJacobian(parameters[1], lift.data());
//            jacobian3 = squareRootInformation_*jacobian3_min*lift;
//
//            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){
//                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian3_min(jacobiansMinimal[3]);
//                map_jacobian3_min = squareRootInformation_*jacobian3_min;
//            }
//        }
//
//        if(jacobians[4] != NULL){
//            Eigen::Map<Eigen::Matrix<double,2,1>> jacobian4(jacobians[4]);
//            jacobian4 = - H*R_IC.transpose()*R_WI1.transpose()*R_WI0*R_IC*C0uv/(inv_dep*inv_dep);
//            jacobian4 = squareRootInformation_*jacobian4;
//
//            if(jacobiansMinimal != NULL && jacobiansMinimal[4] != NULL){
//                Eigen::Map<Eigen::Matrix<double,2,1>> map_jacobian4_min(jacobiansMinimal[4]);
//                map_jacobian4_min = squareRootInformation_*jacobian4;
//            }
//        }
//    }
    return true;
}