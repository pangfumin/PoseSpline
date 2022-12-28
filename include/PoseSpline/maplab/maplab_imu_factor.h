
#ifndef MAPLAB_JPL_IMUERROR_HPP_
#define MAPLAB_JPL_IMUERROR_HPP_


#include "internal/utility.h"

#include <ceres/ceres.h>
// #include "PoseSpline/Quaternion.hpp"
// #include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/maplab/quaternion_param_jpl.h"
#include "quaternion-math.h"


namespace maplab {

    struct ImuParam {
        double ACC_N = 0.1;
        double GYR_N = 0.01;
        double ACC_W = 0.001;
        double GYR_W = 0.0001;
    };

    enum StateOrder {
        O_P = 0,
        O_R = 3,
        O_V = 6,
        O_BA = 9,
        O_BG = 12
    };

    enum NoiseOrder {
        O_AN = 0,
        O_GN = 3,
        O_AW = 6,
        O_GW = 9
    };

class IntegrationBase{
    public:
        IntegrationBase() = delete;

        IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                        const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg,
                        const ImuParam &imuParam)
                : acc_0_{_acc_0}, gyr_0_{_gyr_0}, linearized_acc_{_acc_0}, linearized_gyr_{_gyr_0},
                  linearized_ba_{_linearized_ba}, linearized_bg_{_linearized_bg},
                  jacobian_{Eigen::Matrix<double, 15, 15>::Identity()},
                  covariance_{Eigen::Matrix<double, 15, 15>::Zero()},
                  sum_dt_{0.0}, 
                  delta_p_{Eigen::Vector3d::Zero()}, 
                  delta_q_{unitQuat<double>()},
                  delta_v_{Eigen::Vector3d::Zero()} {
            noise_ = Eigen::Matrix<double, 18, 18>::Zero();
            noise_.block<3, 3>(0, 0) = (imuParam.ACC_N * imuParam.ACC_N) * Eigen::Matrix3d::Identity();
            noise_.block<3, 3>(3, 3) = (imuParam.GYR_N * imuParam.GYR_N) * Eigen::Matrix3d::Identity();
            noise_.block<3, 3>(6, 6) = (imuParam.ACC_N * imuParam.ACC_N) * Eigen::Matrix3d::Identity();
            noise_.block<3, 3>(9, 9) = (imuParam.GYR_N * imuParam.GYR_N) * Eigen::Matrix3d::Identity();
            noise_.block<3, 3>(12, 12) = (imuParam.ACC_W * imuParam.ACC_W) * Eigen::Matrix3d::Identity();
            noise_.block<3, 3>(15, 15) = (imuParam.GYR_W * imuParam.GYR_W) * Eigen::Matrix3d::Identity();
        }

        void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
            dt_buf_.push_back(dt);
            acc_buf_.push_back(acc);
            gyr_buf_.push_back(gyr);
            propagate(dt, acc, gyr);
        }

        void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg) {
            sum_dt_ = 0.0;
            acc_0_ = linearized_acc_;
            gyr_0_ = linearized_gyr_;
            delta_p_.setZero();
            delta_q_.setIdentity();
            delta_v_.setZero();
            linearized_ba_ = _linearized_ba;
            linearized_bg_ = _linearized_bg;
            jacobian_.setIdentity();
            covariance_.setZero();
            for (int i = 0; i < static_cast<int>(dt_buf_.size()); i++) {
              propagate(dt_buf_[i], acc_buf_[i], gyr_buf_[i]);
            }
                
        }

        void midPointIntegration(double _dt,
                                 const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                 const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                 const Eigen::Vector3d &delta_p, const Eigen::Vector4d &delta_q,
                                 const Eigen::Vector3d &delta_v,
                                 const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                                 Eigen::Vector3d &result_delta_p, Eigen::Vector4d &result_delta_q,
                                 Eigen::Vector3d &result_delta_v,
                                 Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg,
                                 bool update_jacobian) {
            //ROS_INFO("midpoint integration");
          Eigen::Matrix3d R_b_I0,  R_b_I1;
          common::toRotationMatrixJPL(common::quaternionInverseJPL(delta_q), &R_b_I0);
          Eigen::Vector3d un_acc_0 = R_b_I0 * (_acc_0 - linearized_ba);
          Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
          Eigen::Vector3d delta(un_gyr * _dt);
          common::positiveQuaternionProductJPL(deltaQuat<double>(delta), delta_q, result_delta_q);
          common::toRotationMatrixJPL(common::quaternionInverseJPL(result_delta_q), &R_b_I1);

          Eigen::Vector3d un_acc_1 = R_b_I1 * (_acc_1 - linearized_ba);
          Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
          result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
          result_delta_v = delta_v + un_acc * _dt;
          result_linearized_ba = linearized_ba;
          result_linearized_bg = linearized_bg;

          if(update_jacobian)
          {
              Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
              Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
              Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
              Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

              R_w_x<<0, -w_x(2), w_x(1),
                      w_x(2), 0, -w_x(0),
                      -w_x(1), w_x(0), 0;
              R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                      a_0_x(2), 0, -a_0_x(0),
                      -a_0_x(1), a_0_x(0), 0;
              R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                      a_1_x(2), 0, -a_1_x(0),
                      -a_1_x(1), a_1_x(0), 0;

              Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
              F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
              F.block<3, 3>(0, 3) = -0.25 * R_b_I0 * R_a_0_x * _dt * _dt +
                                    -0.25 * R_b_I1 * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
              F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3,3) * _dt;
              F.block<3, 3>(0, 9) = -0.25 * (R_b_I0 + R_b_I1) * _dt * _dt;
              F.block<3, 3>(0, 12) = -0.25 * R_b_I1 * R_a_1_x * _dt * _dt * -_dt;
              F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
              F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3,3) * _dt;
              F.block<3, 3>(6, 3) = -0.5 * R_b_I0 * R_a_0_x * _dt +
                                    -0.5 * R_b_I1 * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
              F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
              F.block<3, 3>(6, 9) = -0.5 * (R_b_I0 + R_b_I1) * _dt;
              F.block<3, 3>(6, 12) = -0.5 * R_b_I1 * R_a_1_x * _dt * -_dt;
              F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
              F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
              //cout<<"A"<<endl<<A<<endl;

              Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
              V.block<3, 3>(0, 0) =  0.25 * R_b_I0 * _dt * _dt;
              V.block<3, 3>(0, 3) =  0.25 * -R_b_I1 * R_a_1_x  * _dt * _dt * 0.5 * _dt;
              V.block<3, 3>(0, 6) =  0.25 * R_b_I1 * _dt * _dt;
              V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
              V.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
              V.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
              V.block<3, 3>(6, 0) =  0.5 * R_b_I0 * _dt;
              V.block<3, 3>(6, 3) =  0.5 * -R_b_I1 * R_a_1_x  * _dt * 0.5 * _dt;
              V.block<3, 3>(6, 6) =  0.5 * R_b_I1 * _dt;
              V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
              V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3,3) * _dt;
              V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * _dt;

              //step_jacobian = F;
              //step_V = V;
              jacobian_ = F * jacobian_;
              covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();

              sqrt_Sigma_ = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                      covariance_.inverse()).matrixL().transpose();
          }

        }

        void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1) {
            dt_ = _dt;
            acc_1_ = _acc_1;
            gyr_1_ = _gyr_1;
            Eigen::Vector3d result_delta_p;
            Eigen::Vector4d result_delta_q;
            Eigen::Vector3d result_delta_v;
            Eigen::Vector3d result_linearized_ba;
            Eigen::Vector3d result_linearized_bg;

            midPointIntegration(_dt, acc_0_, gyr_0_, _acc_1, _gyr_1, 
                delta_p_, delta_q_, delta_v_,
                linearized_ba_, linearized_bg_,
                result_delta_p, result_delta_q, result_delta_v,
                result_linearized_ba, result_linearized_bg, 1);

            //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
            //                    linearized_ba, linearized_bg);
            delta_p_ = result_delta_p;
            delta_q_ = result_delta_q;
            delta_v_ = result_delta_v;
            linearized_ba_ = result_linearized_ba;
            linearized_bg_ = result_linearized_bg;
            delta_q_.normalize();
            sum_dt_ += dt_;
            acc_0_ = acc_1_;
            gyr_0_ = gyr_1_;

        }

        template <typename  T>
        Eigen::Matrix<T, 15, 1>
        evaluate(const Eigen::Matrix<T,3,1> &Pi, const Eigen::Matrix<T,4,1> &Q_IiW,
                 const Eigen::Matrix<T,3,1> &Vi,
                 const Eigen::Matrix<T,3,1> &Bai, const Eigen::Matrix<T,3,1> &Bgi,
                 const Eigen::Matrix<T,3,1> &Pj, const Eigen::Matrix<T,4,1> &Q_IjW,
                 const Eigen::Matrix<T,3,1> &Vj,
                 const Eigen::Matrix<T,3,1> &Baj, const Eigen::Matrix<T,3,1> &Bgj,
                 T **jacobians = NULL) {
            Eigen::Matrix<T,3,1> G{T(0.0), T(0.0), T(9.8)};
            Eigen::Matrix<T, 15, 1> residuals;

            Eigen::Matrix<T,3,3> dp_dba = jacobian_.block<3, 3>(O_P, O_BA).cast<T>();
            Eigen::Matrix<T,3,3> dp_dbg = jacobian_.block<3, 3>(O_P, O_BG).cast<T>();

            Eigen::Matrix<T,3,3> dq_dbg = jacobian_.block<3, 3>(O_R, O_BG).cast<T>();

            Eigen::Matrix<T,3,3> dv_dba = jacobian_.block<3, 3>(O_V, O_BA).cast<T>();
            Eigen::Matrix<T,3,3> dv_dbg = jacobian_.block<3, 3>(O_V, O_BG).cast<T>();

            Eigen::Matrix<T,3,1> dba = Bai - linearized_ba_.cast<T>();
            Eigen::Matrix<T,3,1> dbg = Bgi - linearized_bg_.cast<T>();

            Eigen::Matrix<T,3,1> temp = dq_dbg * dbg;
            Eigen::Matrix<T,4,1> corrected_delta_q;
            common::positiveQuaternionProductJPL(deltaQuat<T>(temp), delta_q_.cast<T>(), corrected_delta_q);

            Eigen::Matrix<T,3,1> corrected_delta_v = delta_v_.cast<T>() + dv_dba * dba + dv_dbg * dbg;
            Eigen::Matrix<T,3,1> corrected_delta_p = delta_p_.cast<T>() + dp_dba * dba + dp_dbg * dbg;


            T _sum_dt = T(sum_dt_);
            Eigen::Matrix<T,3,3> R_IiW;
            common::toRotationMatrixJPL(Q_IiW, &R_IiW);
            Eigen::Matrix<T,3,1> temp_p = T(0.5) * G * _sum_dt * _sum_dt + Pj - Pi - Vi * _sum_dt;
            Eigen::Matrix<T,3,1> temp_v = G * _sum_dt + Vj - Vi;
            residuals.template block<3, 1>(O_P, 0) = R_IiW * temp_p - corrected_delta_p;
            Eigen::Matrix<T,4,1> temp0, temp1;
            
            common::positiveQuaternionProductJPL(Q_IiW, common::quaternionInverseJPL(Q_IjW), temp0);
            common::positiveQuaternionProductJPL(corrected_delta_q, temp0, temp1);
            residuals.template block<3, 1>(O_R, 0) = T(2) * temp1.template head<3>();
            residuals.template block<3, 1>(O_V, 0) = R_IiW * temp_v - corrected_delta_v;
            residuals.template block<3, 1>(O_BA, 0) = Baj - Bai;
            residuals.template block<3, 1>(O_BG, 0) = Bgj - Bgi;

            if (jacobians != nullptr) {
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_t_WI0(jacobians[0]);
                Eigen::Map<Eigen::Matrix<T, 15, 4, Eigen::RowMajor>> J_r_q_I0W(jacobians[1]); // JPL
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_v_WI0(jacobians[2]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_ba0(jacobians[3]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_bg0(jacobians[4]);

                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_t_WI1(jacobians[5]);
                Eigen::Map<Eigen::Matrix<T, 15, 4, Eigen::RowMajor>> J_r_q_I1W(jacobians[6]); // JPL
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_v_WI1(jacobians[7]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_ba1(jacobians[8]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_bg1(jacobians[9]);

                J_r_t_WI0.setZero();
                J_r_t_WI0.template block<3,3>(O_P, 0) = - R_IiW;

                Eigen::Matrix<T,3,4,Eigen::RowMajor> lift0, lift1;
                ceres_error_terms::JplQuaternionParameterization::liftJacobian<T>(Q_IiW.data(), lift0.data());
                ceres_error_terms::JplQuaternionParameterization::liftJacobian<T>(Q_IjW.data(), lift1.data());
                J_r_q_I0W.setZero();
                J_r_q_I0W.template block<3,4>(O_P, 0) = common::skew(R_IiW * temp_p) * lift0;
                J_r_q_I0W.template block<3,4>(O_R, 0) = (quatLeftComp<T>(corrected_delta_q)  * quatRightComp<T>(temp0)).topLeftCorner(3,3) * lift0;
                J_r_q_I0W.template block<3,4>(O_V, 0) = common::skew(R_IiW * temp_v) * lift0;

                J_r_v_WI0.setZero();
                J_r_v_WI0.template block<3,3>(O_P, 0) = - R_IiW*_sum_dt;
                J_r_v_WI0.template block<3,3>(O_V, 0) = - R_IiW;

                J_r_ba0.setZero();
                J_r_ba0.template block<3,3>(O_P, 0) = - dp_dba;
                J_r_ba0.template block<3,3>(O_V, 0) = - dv_dba;
                J_r_ba0.template block<3,3>(O_BA, 0) = - Eigen::Matrix<T,3,3>::Identity();

                J_r_bg0.setZero();
                J_r_bg0.template block<3,3>(O_P, 0) = - dp_dbg;
                Eigen::Vector4d temp2;
                common::positiveQuaternionProductJPL(delta_q_.cast<T>(), temp0, temp2);
                J_r_bg0.template block<3,3>(O_R, 0) = quatRightComp<T>(temp2).topLeftCorner(3,3) * dq_dbg;
                J_r_bg0.template block<3,3>(O_V, 0) = - dv_dbg;
                J_r_bg0.template block<3,3>(O_BG, 0) = - Eigen::Matrix<T,3,3>::Identity();

                J_r_t_WI1.setZero();
                J_r_t_WI1.template block<3,3>(O_P, 0) = R_IiW;

                J_r_q_I1W.setZero();
                J_r_q_I1W.template block<3,4>(O_R, 0) = - quatLeftComp<T>(temp1).topLeftCorner(3,3) * lift1;

                J_r_v_WI1.setZero();
                J_r_v_WI1.template block<3,3>(O_V, 0) = R_IiW;

                J_r_ba1.setZero();
                J_r_ba1.template block<3,3>(O_BA, 0) = Eigen::Matrix<T,3,3>::Identity();

                J_r_bg1.setZero();
                J_r_bg1.template block<3,3>(O_BG, 0) = Eigen::Matrix<T,3,3>::Identity();

            }

            return residuals;
        }

        double dt_;
        Eigen::Vector3d acc_0_, gyr_0_;
        Eigen::Vector3d acc_1_, gyr_1_;

        const Eigen::Vector3d linearized_acc_, linearized_gyr_;
        Eigen::Vector3d linearized_ba_, linearized_bg_;

        Eigen::Matrix<double, 15, 15> jacobian_, covariance_;
        Eigen::Matrix<double, 15, 15> sqrt_Sigma_;

        Eigen::Matrix<double, 15, 15> step_jacobian_;
        Eigen::Matrix<double, 15, 18> step_V_;
        Eigen::Matrix<double, 18, 18> noise_;

        double sum_dt_;
        Eigen::Vector3d delta_p_;
        Eigen::Vector4d delta_q_;  // Q_Ikp1_Ik in JPL
        Eigen::Vector3d delta_v_;

        std::vector<double> dt_buf_;
        std::vector<Eigen::Vector3d> acc_buf_;
        std::vector<Eigen::Vector3d> gyr_buf_;

    };


    class IMUFactor : public ceres::SizedCostFunction<15, 7, 3,3,3, 7, 3,3,3> {
    public:
        IMUFactor() = delete;

        IMUFactor(std::shared_ptr<IntegrationBase>& pre_integration) : pre_integration_(pre_integration) {
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

            Eigen::Vector4d Q_IiW(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
            Eigen::Vector3d Pi(parameters[0][4], parameters[0][5], parameters[0][6]);

            Eigen::Vector3d Bgi(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
            Eigen::Vector3d Bai(parameters[3][0], parameters[3][1], parameters[3][2]);

            Eigen::Vector4d Q_IjW(parameters[4][0], parameters[4][1], parameters[4][2], parameters[4][3]);
            Eigen::Vector3d Pj(parameters[4][4], parameters[4][5], parameters[4][6]);

            Eigen::Vector3d Bgj(parameters[5][0], parameters[5][1], parameters[5][2]);
            Eigen::Vector3d Vj(parameters[6][0], parameters[6][1], parameters[6][2]);
            Eigen::Vector3d Baj(parameters[7][0], parameters[7][1], parameters[7][2]);


            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_t_WI0;
            Eigen::Matrix<double, 15, 4, Eigen::RowMajor> J_r_q_I0W;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_v_WI0;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_ba0;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_bg0;

            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_t_WI1;
            Eigen::Matrix<double, 15, 4, Eigen::RowMajor> J_r_q_I1W;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_v_WI1;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_ba1;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_r_bg1;


            double* internal_jacobians[10] = {J_r_t_WI0.data(), J_r_q_I0W.data(),
                                              J_r_v_WI0.data(), J_r_ba0.data(), J_r_bg0.data(),
                                              J_r_t_WI1.data(), J_r_q_I1W.data(),
                                              J_r_v_WI1.data(), J_r_ba1.data(), J_r_bg1.data()};

            Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
            residual = pre_integration_->evaluate(Pi, Q_IiW, Vi, Bai, Bgi,
                                                 Pj, Q_IjW, Vj, Baj, Bgj,
                                                 internal_jacobians);

            Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                    pre_integration_->covariance_.inverse()).matrixL().transpose();

            sqrt_info.setIdentity();  // todo: remove

            residual = sqrt_info * residual;

            Eigen::Vector3d G{0.0, 0.0, 9.8};
            if (jacobians)
            {

                if (jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i << J_r_q_I0W, J_r_t_WI0;
                    jacobian_pose_i  = sqrt_info * jacobian_pose_i;
                }

                if (jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_biasgyro_i(jacobians[1]);
                    jacobian_biasgyro_i.setZero();
                    jacobian_biasgyro_i = J_r_bg0;
                    jacobian_biasgyro_i  = sqrt_info * jacobian_biasgyro_i;

                }

                if (jacobians[2])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_velocity_i(jacobians[2]);
                    jacobian_velocity_i.setZero();

                    jacobian_velocity_i = J_r_v_WI0;
                    jacobian_velocity_i  = sqrt_info * jacobian_velocity_i;

                }

                if (jacobians[3])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_biasaccl_i(jacobians[3]);
                    jacobian_biasaccl_i.setZero();

                    jacobian_biasaccl_i = J_r_ba0;
                    jacobian_biasaccl_i  = sqrt_info * jacobian_biasaccl_i;

                }

                if (jacobians[4])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[4]);
                    jacobian_pose_j.setZero();
                    jacobian_pose_j << J_r_q_I1W, J_r_t_WI1;
                    jacobian_pose_j  = sqrt_info * jacobian_pose_j;
                }

                if (jacobians[5])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_biasgyro_j(jacobians[5]);
                    jacobian_biasgyro_j.setZero();
                    jacobian_biasgyro_j = J_r_bg1;
                    jacobian_biasgyro_j  = sqrt_info * jacobian_biasgyro_j;

                }

                if (jacobians[6])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_velocity_j(jacobians[6]);
                    jacobian_velocity_j.setZero();

                    jacobian_velocity_j = J_r_v_WI1;
                    jacobian_velocity_j  = sqrt_info * jacobian_velocity_j;

                }

                if (jacobians[7])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_biasaccl_j(jacobians[7]);
                    jacobian_biasaccl_j.setZero();

                    jacobian_biasaccl_j = J_r_ba1;
                    jacobian_biasaccl_j  = sqrt_info * jacobian_biasaccl_j;

                }
            }

            return true;
        }

        private:
        std::shared_ptr<IntegrationBase> pre_integration_;

    };

}
#endif
