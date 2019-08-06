
#ifndef INCLUDE_SPLINE_IMUERROR_HPP_
#define INCLUDE_SPLINE_IMUERROR_HPP_


#include "internal/utility.h"

#include <ceres/ceres.h>
#include "PoseSpline/Quaternion.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "extern/JPL_imu_error.h"
#include "PoseSpline/QuaternionSplineUtility.hpp"
namespace  JPL {
    class SplineIMUFactor : public ceres::SizedCostFunction<15, 7, 7, 7, 7, 6, 6, 6, 6> {
    public:
        SplineIMUFactor() = delete;

        SplineIMUFactor(IntegrationBase *_pre_integration,
                const double spline_dt,
                const double& u0, const double& u1) :
                        pre_integration(_pre_integration),
                        spline_dt_(spline_dt),
                        t0_(u0), t1_(u1) {
        }

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
            return EvaluateWithMinimalJacobians(parameters,
                                                residuals,
                                                jacobians, NULL);
        }


        bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                          double *residuals,
                                          double **jacobians,
                                          double **jacobiansMinimal) const {
            Pose<double> T0(parameters[0]);
            Pose<double> T1(parameters[1]);
            Pose<double> T2(parameters[2]);
            Pose<double> T3(parameters[3]);


            Eigen::Map<const Eigen::Matrix<double,6,1>> b0(parameters[4]);
            Eigen::Map<const Eigen::Matrix<double,6,1>> b1(parameters[5]);
            Eigen::Map<const Eigen::Matrix<double,6,1>> b2(parameters[6]);
            Eigen::Map<const Eigen::Matrix<double,6,1>> b3(parameters[7]);


            QuaternionTemplate<double> Q0 = T0.rotation();
            QuaternionTemplate<double> Q1 = T1.rotation();
            QuaternionTemplate<double> Q2 = T2.rotation();
            QuaternionTemplate<double> Q3 = T3.rotation();

            Eigen::Matrix<double,3,1> t0 = T0.translation();
            Eigen::Matrix<double,3,1> t1 = T1.translation();
            Eigen::Matrix<double,3,1> t2 = T2.translation();
            Eigen::Matrix<double,3,1> t3 = T3.translation();

//            std::cout << "t0: " << t0.transpose() << std::endl;
//            std::cout << "t1: " << t1.transpose() << std::endl;
//            std::cout << "t2: " << t2.transpose() << std::endl;
//            std::cout << "t3: " << t3.transpose() << std::endl;

            double  Beta01 = QSUtility::beta1(t0_);
            double  Beta02 = QSUtility::beta2((t0_));
            double  Beta03 = QSUtility::beta3((t0_));

//            std::cout << "U: " << t0_ << " " << Beta01 << " " << Beta02 << " " << Beta03 << std::endl;


            double  Beta11 = QSUtility::beta1((t1_));
            double  Beta12 = QSUtility::beta2((t1_));
            double  Beta13 = QSUtility::beta3((t1_));

            Eigen::Matrix<double,3,1> phi1 = QSUtility::Phi<double>(Q0,Q1);
            Eigen::Matrix<double,3,1> phi2 = QSUtility::Phi<double>(Q1,Q2);
            Eigen::Matrix<double,3,1> phi3 = QSUtility::Phi<double>(Q2,Q3);

            QuaternionTemplate<double> r_01 = QSUtility::r(Beta01,phi1);
            QuaternionTemplate<double> r_02 = QSUtility::r(Beta02,phi2);
            QuaternionTemplate<double> r_03 = QSUtility::r(Beta03,phi3);
            double  dotBeta01 = QSUtility::dot_beta1(spline_dt_, t0_);
            double  dotBeta02 = QSUtility::dot_beta2(spline_dt_, t0_);
            double  dotBeta03 = QSUtility::dot_beta3(spline_dt_, t0_);


            QuaternionTemplate<double> r_11 = QSUtility::r(Beta11,phi1);
            QuaternionTemplate<double> r_12 = QSUtility::r(Beta12,phi2);
            QuaternionTemplate<double> r_13 = QSUtility::r(Beta13,phi3);

            double  dotBeta11 = QSUtility::dot_beta1(spline_dt_, t1_);
            double  dotBeta12 = QSUtility::dot_beta2(spline_dt_, t1_);
            double  dotBeta13 = QSUtility::dot_beta3(spline_dt_, t1_);


            Eigen::Vector3d Pi = t0 + Beta01*(t1 - t0) +  Beta02*(t2 - t1) + Beta03*(t3 - t2);
//            std::cout << "V0: " << t0.transpose() << std::endl;
//            std::cout << "V1: " << (t1 - t0).transpose() << std::endl;
//            std::cout << "V2: " << (t2 - t1).transpose() << std::endl;
//            std::cout << "V3: " << (t3 - t2).transpose() << std::endl;

            QuaternionTemplate<double> Qi = quatLeftComp(Q0)*quatLeftComp(r_01)*quatLeftComp(r_02)*r_03;
            Eigen::Vector3d Vi = dotBeta01*(t1 - t0) +  dotBeta02*(t2 - t1) + dotBeta03*(t3 - t2);
            Eigen::Matrix<double,6,1> bias_i = b0 + Beta01*(b1 - b0) +  Beta02*(b2 - b1) + Beta03*(b3 - b2);
            Eigen::Vector3d Bai = bias_i.head<3>();
            Eigen::Vector3d Bgi = bias_i.tail<3>();

            Eigen::Vector3d Pj = t0 + Beta11*(t1 - t0) +  Beta12*(t2 - t1) + Beta13*(t3 - t2);
            QuaternionTemplate<double> Qj = quatLeftComp(Q0)*quatLeftComp(r_11)*quatLeftComp(r_12)*r_13;
            Eigen::Vector3d Vj = dotBeta11*(t1 - t0) +  dotBeta12*(t2 - t1) + dotBeta13*(t3 - t2);

            Eigen::Matrix<double,6,1> bias_j = b0 + Beta11*(b1 - b0) +  Beta12*(b2 - b1) + Beta13*(b3 - b2);
            Eigen::Vector3d Baj = bias_j.head<3>();
            Eigen::Vector3d Bgj = bias_j.tail<3>();

//            std::cout << "** Pi: " << Pi.transpose() << std::endl;
//            std::cout << "** Qi: " << Qi.transpose() << std::endl;
//            std::cout << "** Vi: " << Vi.transpose() << std::endl;
//            std::cout << "** Bai: " << Bai.transpose() << std::endl;
//            std::cout << "** Bgi: " << Bgi.transpose() << std::endl;
//
//            std::cout << "** Pj: " << Pj.transpose() << std::endl;
//            std::cout << "** Qj: " << Qj.transpose() << std::endl;
//            std::cout << "** Vj: " << Vj.transpose() << std::endl;
//            std::cout << "** Baj: " << Baj.transpose() << std::endl;
//            std::cout << "** Bgj: " << Bgj.transpose() << std::endl;


            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_t_WI0;
            Eigen::Matrix<double, 15, 4, Eigen::RowMajor> J_r_q_WI0;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_v_WI0;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_ba0;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_bg0;

            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_t_WI1;
            Eigen::Matrix<double, 15, 4, Eigen::RowMajor> J_r_q_WI1;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_v_WI1;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_ba1;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_bg1;


            double* internal_jacobians[10] = {J_r_t_WI0.data(), J_r_q_WI0.data(),
                                              J_r_v_WI0.data(), J_r_ba0.data(), J_r_bg0.data(),
                                              J_r_t_WI1.data(), J_r_q_WI1.data(),
                                              J_r_v_WI1.data(), J_r_ba1.data(), J_r_bg1.data()};

            Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
            residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                                 Pj, Qj, Vj, Baj, Bgj,
                                                 internal_jacobians);

            Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                    pre_integration->covariance.inverse()).matrixL().transpose();
            sqrt_info.setIdentity();

            residual = sqrt_info * residual;

            Eigen::Vector3d G{0.0, 0.0, 9.8};
            if (jacobians)
            {
                Eigen::Matrix<double,4,3> Vee = QSUtility::V<double>();

                Eigen::Vector3d BetaPhi01 = Beta01*phi1;
                Eigen::Vector3d BetaPhi02 = Beta02*phi2;
                Eigen::Vector3d BetaPhi03 = Beta03*phi3;

                Eigen::Vector3d BetaPhi11 = Beta11*phi1;
                Eigen::Vector3d BetaPhi12 = Beta12*phi2;
                Eigen::Vector3d BetaPhi13 = Beta13*phi3;

                Eigen::Matrix3d S01 = quatS(BetaPhi01);
                Eigen::Matrix3d S02 = quatS(BetaPhi02);
                Eigen::Matrix3d S03 = quatS(BetaPhi03);

                Eigen::Matrix3d S11 = quatS(BetaPhi11);
                Eigen::Matrix3d S12 = quatS(BetaPhi12);
                Eigen::Matrix3d S13 = quatS(BetaPhi13);


                Quaternion invQ0Q1 = quatLeftComp(quatInv<double>(Q0))*Q1;
                Quaternion invQ1Q2 = quatLeftComp(quatInv<double>(Q1))*Q2;
                Quaternion invQ2Q3 = quatLeftComp(quatInv<double>(Q2))*Q3;
                Eigen::Matrix3d L1 = quatL(invQ0Q1);
                Eigen::Matrix3d L2 = quatL(invQ1Q2);
                Eigen::Matrix3d L3 = quatL(invQ2Q3);

                Eigen::Matrix3d C0 = quatToRotMat<double>(Q0);
                Eigen::Matrix3d C1 = quatToRotMat<double>(Q1);
                Eigen::Matrix3d C2 = quatToRotMat<double>(Q2);


                Eigen::Matrix<double,4,3> temp0_0, temp0_1;
                Eigen::Matrix<double,4,3> temp01_0, temp01_1;
                Eigen::Matrix<double,4,3> temp12_0, temp12_1;
                Eigen::Matrix<double,4,3> temp23_0, temp23_1;


                temp0_0 = quatRightComp<double>(quatLeftComp<double>(r_01)*quatLeftComp<double>(r_02)*r_03)*quatRightComp<double>(Q0)*Vee;
                temp01_0 = quatLeftComp<double>(Q0)*quatRightComp<double>(quatLeftComp<double>(r_02)*r_03)*quatRightComp<double>(r_01)*Vee*S01*Beta01*L1*C0.transpose();
                temp12_0 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r_01)*quatRightComp<double>(r_03)*quatRightComp<double>(r_02)*Vee*S02*Beta02*L2*C1.transpose();
                temp23_0 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r_01)*quatLeftComp<double>(r_02)*quatRightComp<double>(r_03)*Vee*S03*Beta03*L3*C2.transpose();

                temp0_1 = quatRightComp<double>(quatLeftComp<double>(r_11)*quatLeftComp<double>(r_12)*r_13)*quatRightComp<double>(Q0)*Vee;
                temp01_1 = quatLeftComp<double>(Q0)*quatRightComp<double>(quatLeftComp<double>(r_12)*r_13)*quatRightComp<double>(r_11)*Vee*S11*Beta11*L1*C0.transpose();
                temp12_1 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r_11)*quatRightComp<double>(r_13)*quatRightComp<double>(r_12)*Vee*S12*Beta12*L2*C1.transpose();
                temp23_1 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r_11)*quatLeftComp<double>(r_12)*quatRightComp<double>(r_13)*Vee*S13*Beta13*L3*C2.transpose();


                Eigen::Matrix<double,7,6,Eigen::RowMajor> J_T_WI0_T0, J_T_WI0_T1, J_T_WI0_T2, J_T_WI0_T3;
                Eigen::Matrix<double,7,6,Eigen::RowMajor> J_T_WI1_T0, J_T_WI1_T1, J_T_WI1_T2, J_T_WI1_T3;
                Eigen::Matrix<double,3,6,Eigen::RowMajor> J_v_WI0_T0, J_v_WI0_T1, J_v_WI0_T2, J_v_WI0_T3;
                Eigen::Matrix<double,3,6,Eigen::RowMajor> J_v_WI1_T0, J_v_WI1_T1, J_v_WI1_T2, J_v_WI1_T3;

                J_T_WI0_T0.setZero();
                J_T_WI0_T0.topLeftCorner(3,3) = (1 - Beta01)*Eigen::Matrix3d::Identity();
                J_T_WI0_T0.bottomRightCorner(4,3) = temp0_0 - temp01_0;


                J_T_WI0_T1.setZero();
                J_T_WI0_T1.topLeftCorner(3,3) = (Beta01 - Beta02)*Eigen::Matrix3d::Identity();
                J_T_WI0_T1.bottomRightCorner(4,3) = temp01_0 - temp12_0;

                J_T_WI0_T2.setZero();
                J_T_WI0_T2.topLeftCorner(3,3) = (Beta02 - Beta03)*Eigen::Matrix3d::Identity();
                J_T_WI0_T2.bottomRightCorner(4,3) = temp12_0 - temp23_0;

                J_T_WI0_T3.setZero();
                J_T_WI0_T3.topLeftCorner(3,3) = Beta03*Eigen::Matrix3d::Identity();;
                J_T_WI0_T3.bottomRightCorner(4,3) = temp23_0;


                J_v_WI0_T0.setZero();
                J_v_WI0_T0.block<3,3>(0,0) = -dotBeta01*Eigen::Matrix3d::Identity();

                J_v_WI0_T1.setZero();
                J_v_WI0_T1.block<3,3>(0,0) = (dotBeta01 - dotBeta02)*Eigen::Matrix3d::Identity();

                J_v_WI0_T2.setZero();
                J_v_WI0_T2.block<3,3>(0,0) = (dotBeta02 - dotBeta03)*Eigen::Matrix3d::Identity();

                J_v_WI0_T3.setZero();
                J_v_WI0_T3.block<3,3>(0,0) = dotBeta03*Eigen::Matrix3d::Identity();

                J_T_WI1_T0.setZero();
                J_T_WI1_T0.topLeftCorner(3,3) = (1 - Beta11)*Eigen::Matrix3d::Identity();
                J_T_WI1_T0.bottomRightCorner(4,3) = temp0_1 - temp01_1;


                J_T_WI1_T1.setZero();
                J_T_WI1_T1.topLeftCorner(3,3) = (Beta11 - Beta12)*Eigen::Matrix3d::Identity();
                J_T_WI1_T1.bottomRightCorner(4,3) = temp01_1 - temp12_1;

                J_T_WI1_T2.setZero();
                J_T_WI1_T2.topLeftCorner(3,3) = (Beta12 - Beta13)*Eigen::Matrix3d::Identity();
                J_T_WI1_T2.bottomRightCorner(4,3) = temp12_1 - temp23_1;

                J_T_WI1_T3.setZero();
                J_T_WI1_T3.topLeftCorner(3,3) = Beta13*Eigen::Matrix3d::Identity();;
                J_T_WI1_T3.bottomRightCorner(4,3) = temp23_1;


                J_v_WI1_T0.setZero();
                J_v_WI1_T0.block<3,3>(0,0) = -dotBeta11*Eigen::Matrix3d::Identity();

                J_v_WI1_T1.setZero();
                J_v_WI1_T1.block<3,3>(0,0) = (dotBeta11 - dotBeta12)*Eigen::Matrix3d::Identity();

                J_v_WI1_T2.setZero();
                J_v_WI1_T2.block<3,3>(0,0) = (dotBeta12 - dotBeta13)*Eigen::Matrix3d::Identity();

                J_v_WI1_T3.setZero();
                J_v_WI1_T3.block<3,3>(0,0) = dotBeta13*Eigen::Matrix3d::Identity();


                Eigen::Matrix<double,15,7,Eigen::RowMajor> J_r_pose_i;
                Eigen::Matrix<double,15,7,Eigen::RowMajor> J_r_pose_j;
                Eigen::Matrix<double,15,6,Eigen::RowMajor> J_r_bias_i;
                Eigen::Matrix<double,15,6,Eigen::RowMajor> J_r_bias_j;
                J_r_pose_i << J_r_t_WI0, J_r_q_WI0;
                J_r_pose_j << J_r_t_WI1, J_r_q_WI1;
                J_r_bias_i << J_r_ba0, J_r_bg0;
                J_r_bias_j << J_r_ba1, J_r_bg1;


                Eigen::Matrix<double,6,6,Eigen::RowMajor> J_bias_i_b0, J_bias_i_b1, J_bias_i_b2, J_bias_i_b3;
                Eigen::Matrix<double,6,6,Eigen::RowMajor> J_bias_j_b0, J_bias_j_b1, J_bias_j_b2, J_bias_j_b3;
                Eigen::Matrix<double,6,6,Eigen::RowMajor> I6;
                I6.setIdentity();
                J_bias_i_b0 = (1 - Beta01)*I6;
                J_bias_i_b1 = (Beta01 - Beta02)*I6;
                J_bias_i_b2 = (Beta02 - Beta03)*I6;
                J_bias_i_b3 = (Beta03)*I6;

                J_bias_j_b0 = (1 - Beta11)*I6;
                J_bias_j_b1 = (Beta11 - Beta12)*I6;
                J_bias_j_b2 = (Beta12 - Beta13)*I6;
                J_bias_j_b3 = (Beta13)*I6;

                if (jacobians[0])
                {
                    Eigen::Matrix<double,6,7,Eigen::RowMajor> lift;
                    PoseLocalParameter::liftJacobian(parameters[0], lift.data());
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
                    jacobian = (J_r_pose_i*J_T_WI0_T0 + J_r_v_WI0*J_v_WI0_T0 + J_r_pose_j*J_T_WI1_T0 + J_r_v_WI1*J_v_WI1_T0)*lift;
                }
                if (jacobians[1])
                {
                    Eigen::Matrix<double,6,7,Eigen::RowMajor> lift;
                    PoseLocalParameter::liftJacobian(parameters[1], lift.data());
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian(jacobians[1]);
                    jacobian = (J_r_pose_i*J_T_WI0_T1 + J_r_v_WI0*J_v_WI0_T1 + J_r_pose_j*J_T_WI1_T1 + J_r_v_WI1*J_v_WI1_T1)*lift;
                }
                if (jacobians[2])
                {
                    Eigen::Matrix<double,6,7,Eigen::RowMajor> lift;
                    PoseLocalParameter::liftJacobian(parameters[2], lift.data());
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian(jacobians[2]);
                    jacobian = (J_r_pose_i*J_T_WI0_T2 + J_r_v_WI0*J_v_WI0_T2 + J_r_pose_j*J_T_WI1_T2 + J_r_v_WI1*J_v_WI1_T2)*lift;
                }
                if (jacobians[3])
                {
                    Eigen::Matrix<double,6,7,Eigen::RowMajor> lift;
                    PoseLocalParameter::liftJacobian(parameters[3], lift.data());
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian(jacobians[3]);
                    jacobian = (J_r_pose_i*J_T_WI0_T3 + J_r_v_WI0*J_v_WI0_T3 + J_r_pose_j*J_T_WI1_T3 + J_r_v_WI1*J_v_WI1_T3)*lift;
                }

                if (jacobians[4])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian(jacobians[4]);
                    jacobian = J_r_bias_i * J_bias_i_b0 + J_r_bias_j * J_bias_j_b0;
                }

                if (jacobians[5])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian(jacobians[5]);
                    jacobian = J_r_bias_i * J_bias_i_b1 + J_r_bias_j * J_bias_j_b1;
                }

                if (jacobians[6])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian(jacobians[6]);
                    jacobian = J_r_bias_i * J_bias_i_b2 + J_r_bias_j * J_bias_j_b2;
                }

                if (jacobians[7])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian(jacobians[7]);
                    jacobian = J_r_bias_i * J_bias_i_b3 + J_r_bias_j * J_bias_j_b3;
                }

            }

            return true;
        }
        IntegrationBase *pre_integration;
        double spline_dt_;
        double t0_;
        double t1_;
    };
}
#endif
