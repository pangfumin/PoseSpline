
#ifndef INCLUDE_JPL_IMUERROR_HPP_
#define INCLUDE_JPL_IMUERROR_HPP_


#include "internal/utility.h"

#include <ceres/ceres.h>
#include "PoseSpline/Quaternion.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/QuaternionLocalParameter.hpp"


namespace  JPL {
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
                : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
                  linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
                  jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
                  covariance{Eigen::Matrix<double, 15, 15>::Zero()},
                  sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{unitQuat<double>()},
                  delta_v{Eigen::Vector3d::Zero()} {
            noise = Eigen::Matrix<double, 18, 18>::Zero();
            noise.block<3, 3>(0, 0) = (imuParam.ACC_N * imuParam.ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) = (imuParam.GYR_N * imuParam.GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(6, 6) = (imuParam.ACC_N * imuParam.ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(9, 9) = (imuParam.GYR_N * imuParam.GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(12, 12) = (imuParam.ACC_W * imuParam.ACC_W) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(15, 15) = (imuParam.GYR_W * imuParam.GYR_W) * Eigen::Matrix3d::Identity();
        }

        void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
            dt_buf.push_back(dt);
            acc_buf.push_back(acc);
            gyr_buf.push_back(gyr);
            propagate(dt, acc, gyr);
        }

        void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg) {
            sum_dt = 0.0;
            acc_0 = linearized_acc;
            gyr_0 = linearized_gyr;
            delta_p.setZero();
            delta_q.setIdentity();
            delta_v.setZero();
            linearized_ba = _linearized_ba;
            linearized_bg = _linearized_bg;
            jacobian.setIdentity();
            covariance.setZero();
            for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
                propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        }

        void midPointIntegration(double _dt,
                                 const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                 const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                 const Eigen::Vector3d &delta_p, const QuaternionTemplate<double> &delta_q,
                                 const Eigen::Vector3d &delta_v,
                                 const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                                 Eigen::Vector3d &result_delta_p, QuaternionTemplate<double> &result_delta_q,
                                 Eigen::Vector3d &result_delta_v,
                                 Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg,
                                 bool update_jacobian) {
            //ROS_INFO("midpoint integration");
            Eigen::Matrix3d R_b_I0 = quatToRotMat(quatInv(delta_q));
            Eigen::Vector3d un_acc_0 = R_b_I0 * (_acc_0 - linearized_ba);
            Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Eigen::Vector3d delta(un_gyr * _dt);
            result_delta_q = quatMult(deltaQuat<double>(delta), delta_q);
            Eigen::Matrix3d R_b_I1 = quatToRotMat(quatInv(result_delta_q));

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
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();

            sqrt_Sigma = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                    covariance.inverse()).matrixL().transpose();
        }

        }

        void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1) {
            dt = _dt;
            acc_1 = _acc_1;
            gyr_1 = _gyr_1;
            Eigen::Vector3d result_delta_p;
            QuaternionTemplate<double> result_delta_q;
            Eigen::Vector3d result_delta_v;
            Eigen::Vector3d result_linearized_ba;
            Eigen::Vector3d result_linearized_bg;

            midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, 
                delta_p, delta_q, delta_v,
                linearized_ba, linearized_bg,
                result_delta_p, result_delta_q, result_delta_v,
                result_linearized_ba, result_linearized_bg, 1);

            //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
            //                    linearized_ba, linearized_bg);
            delta_p = result_delta_p;
            delta_q = result_delta_q;
            delta_v = result_delta_v;
            linearized_ba = result_linearized_ba;
            linearized_bg = result_linearized_bg;
            delta_q.normalize();
            sum_dt += dt;
            acc_0 = acc_1;
            gyr_0 = gyr_1;

        }

        template <typename  T>
        Eigen::Matrix<T, 15, 1>
        evaluate(const Eigen::Matrix<T,3,1> &Pi, const QuaternionTemplate<T> &Qi,
                 const Eigen::Matrix<T,3,1> &Vi,
                 const Eigen::Matrix<T,3,1> &Bai, const Eigen::Matrix<T,3,1> &Bgi,
                 const Eigen::Matrix<T,3,1> &Pj, const QuaternionTemplate<T> &Qj,
                 const Eigen::Matrix<T,3,1> &Vj,
                 const Eigen::Matrix<T,3,1> &Baj, const Eigen::Matrix<T,3,1> &Bgj,
                 T **jacobians = NULL) {
            Eigen::Matrix<T,3,1> G{T(0.0), T(0.0), T(9.8)};
            Eigen::Matrix<T, 15, 1> residuals;

            Eigen::Matrix<T,3,3> dp_dba = jacobian.block<3, 3>(O_P, O_BA).cast<T>();
            Eigen::Matrix<T,3,3> dp_dbg = jacobian.block<3, 3>(O_P, O_BG).cast<T>();

            Eigen::Matrix<T,3,3> dq_dbg = jacobian.block<3, 3>(O_R, O_BG).cast<T>();

            Eigen::Matrix<T,3,3> dv_dba = jacobian.block<3, 3>(O_V, O_BA).cast<T>();
            Eigen::Matrix<T,3,3> dv_dbg = jacobian.block<3, 3>(O_V, O_BG).cast<T>();

            Eigen::Matrix<T,3,1> dba = Bai - linearized_ba.cast<T>();
            Eigen::Matrix<T,3,1> dbg = Bgi - linearized_bg.cast<T>();

            Eigen::Matrix<T,3,1> temp = dq_dbg * dbg;
            QuaternionTemplate<T> corrected_delta_q = quatMult<T>(deltaQuat<T>(temp), delta_q.cast<T>());

            Eigen::Matrix<T,3,1> corrected_delta_v = delta_v.cast<T>() + dv_dba * dba + dv_dbg * dbg;
            Eigen::Matrix<T,3,1> corrected_delta_p = delta_p.cast<T>() + dp_dba * dba + dp_dbg * dbg;


            T _sum_dt = T(sum_dt);
            Eigen::Matrix<T,3,3> R_WIi = quatToRotMat(Qi);
            Eigen::Matrix<T,3,1> temp_p = T(0.5) * G * _sum_dt * _sum_dt + Pj - Pi - Vi * _sum_dt;
            Eigen::Matrix<T,3,1> temp_v = G * _sum_dt + Vj - Vi;
            residuals.template block<3, 1>(O_P, 0) = R_WIi.transpose() * temp_p - corrected_delta_p;
            QuaternionTemplate<T> temp0 = quatMult<T>(quatInv(Qi), Qj);
            // std::cout << "JPL  temp0: " << temp0.transpose() << std::endl; 
            residuals.template block<3, 1>(O_R, 0) = T(2) * quatMult<T>( (corrected_delta_q), temp0).template head<3>();
            residuals.template block<3, 1>(O_V, 0) = R_WIi.transpose() * temp_v - corrected_delta_v;
            residuals.template block<3, 1>(O_BA, 0) = Baj - Bai;
            residuals.template block<3, 1>(O_BG, 0) = Bgj - Bgi;

            if (jacobians != nullptr) {
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_t_WI0(jacobians[0]);
                Eigen::Map<Eigen::Matrix<T, 15, 4, Eigen::RowMajor>> J_r_q_WI0(jacobians[1]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_v_WI0(jacobians[2]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_ba0(jacobians[3]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_bg0(jacobians[4]);

                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_t_WI1(jacobians[5]);
                Eigen::Map<Eigen::Matrix<T, 15, 4, Eigen::RowMajor>> J_r_q_WI1(jacobians[6]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_v_WI1(jacobians[7]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_ba1(jacobians[8]);
                Eigen::Map<Eigen::Matrix<T, 15, 3, Eigen::RowMajor>> J_r_bg1(jacobians[9]);

                J_r_t_WI0.setZero();
                J_r_t_WI0.template block<3,3>(O_P, 0) = - R_WIi.transpose();

                Eigen::Matrix<T,3,4,Eigen::RowMajor> lift0, lift1;
                QuaternionLocalParameter::liftJacobian<T>(Qi.data(), lift0.data());
                QuaternionLocalParameter::liftJacobian<T>(Qj.data(), lift1.data());
                J_r_q_WI0.setZero();
                J_r_q_WI0.template block<3,4>(O_P, 0) = - R_WIi.transpose() * crossMat(temp_p) * lift0;
                J_r_q_WI0.template block<3,4>(O_R, 0) = - (quatLeftComp<T>(quatMult<T>(corrected_delta_q, quatInv<T>(Qi))) * quatRightComp<T>(Qj)).topLeftCorner(3,3) * lift0;
                J_r_q_WI0.template block<3,4>(O_V, 0) = - R_WIi.transpose() * crossMat(temp_v) * lift0;

                J_r_v_WI0.setZero();
                J_r_v_WI0.template block<3,3>(O_P, 0) = - R_WIi.transpose()*_sum_dt;
                J_r_v_WI0.template block<3,3>(O_V, 0) = - R_WIi.transpose();

                J_r_ba0.setZero();
                J_r_ba0.template block<3,3>(O_P, 0) = - dp_dba;
                J_r_ba0.template block<3,3>(O_V, 0) = - dv_dba;
                J_r_ba0.template block<3,3>(O_BA, 0) = - Eigen::Matrix<T,3,3>::Identity();

                J_r_bg0.setZero();
                J_r_bg0.template block<3,3>(O_P, 0) = - dp_dbg;
                // J_r_bg0.template block<3,3>(O_R, 0) = - (quatLeftComp<T>(quatInv<T>(delta_q.cast<T>())) * quatRightComp<T>(quatMult<T>(quatInv<T>(Qj), Qi))).topLeftCorner(3,3) * dq_dbg;
                J_r_bg0.template block<3,3>(O_R, 0) = (quatRightComp<T>(quatMult<T>(delta_q.cast<T>(), quatMult<T>(quatInv<T>(Qi), Qj)))).topLeftCorner(3,3) * dq_dbg;
                J_r_bg0.template block<3,3>(O_V, 0) = - dv_dbg;
                J_r_bg0.template block<3,3>(O_BG, 0) = - Eigen::Matrix<T,3,3>::Identity();

                J_r_t_WI1.setZero();
                J_r_t_WI1.template block<3,3>(O_P, 0) = R_WIi.transpose();

                J_r_q_WI1.setZero();
                J_r_q_WI1.template block<3,4>(O_R, 0) = (quatLeftComp<T>(quatMult<T>(corrected_delta_q, quatInv<T>(Qi))) * quatRightComp<T>(Qj)).topLeftCorner(3,3) * lift1;

                J_r_v_WI1.setZero();
                J_r_v_WI1.template block<3,3>(O_V, 0) = R_WIi.transpose();

                J_r_ba1.setZero();
                J_r_ba1.template block<3,3>(O_BA, 0) = Eigen::Matrix<T,3,3>::Identity();

                J_r_bg1.setZero();
                J_r_bg1.template block<3,3>(O_BG, 0) = Eigen::Matrix<T,3,3>::Identity();

            }

            return residuals;
        }

        double dt;
        Eigen::Vector3d acc_0, gyr_0;
        Eigen::Vector3d acc_1, gyr_1;

        const Eigen::Vector3d linearized_acc, linearized_gyr;
        Eigen::Vector3d linearized_ba, linearized_bg;

        Eigen::Matrix<double, 15, 15> jacobian, covariance;
        Eigen::Matrix<double, 15, 15> sqrt_Sigma;

        Eigen::Matrix<double, 15, 15> step_jacobian;
        Eigen::Matrix<double, 15, 18> step_V;
        Eigen::Matrix<double, 18, 18> noise;

        double sum_dt;
        Eigen::Vector3d delta_p;
        QuaternionTemplate<double> delta_q;  // Q_Ikp1_Ik in JPL
        Eigen::Vector3d delta_v;

        std::vector<double> dt_buf;
        std::vector<Eigen::Vector3d> acc_buf;
        std::vector<Eigen::Vector3d> gyr_buf;

    };


    class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
    public:
        IMUFactor() = delete;

        IMUFactor(IntegrationBase *_pre_integration) : pre_integration(_pre_integration) {
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

            Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
            QuaternionTemplate<double> Qi(parameters[0][3], parameters[0][4], parameters[0][5], parameters[0][6]);

            Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
            Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

            Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
            QuaternionTemplate<double> Qj(parameters[2][3], parameters[2][4], parameters[2][5], parameters[2][6]);

            Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
            Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
            Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);


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

            sqrt_info.setIdentity();  // todo: remove

            residual = sqrt_info * residual;

            Eigen::Vector3d G{0.0, 0.0, 9.8};
            if (jacobians)
            {

                if (jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i << J_r_t_WI0, J_r_q_WI0;
                    jacobian_pose_i  = sqrt_info * jacobian_pose_i;
                }
                if (jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                    jacobian_speedbias_i.setZero();
                    jacobian_speedbias_i << J_r_v_WI0, J_r_ba0, J_r_bg0;
                    jacobian_speedbias_i  = sqrt_info * jacobian_speedbias_i;

                }
                if (jacobians[2])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                    jacobian_pose_j.setZero();

                    jacobian_pose_j << J_r_t_WI1, J_r_q_WI1;
                    jacobian_pose_j  = sqrt_info * jacobian_pose_j;

                }
                if (jacobians[3])
                {
                    Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                    jacobian_speedbias_j.setZero();

                    jacobian_speedbias_j << J_r_v_WI1, J_r_ba1, J_r_bg1;
                    jacobian_speedbias_j  = sqrt_info * jacobian_speedbias_j;
                }
            }

            return true;
        }


        IntegrationBase *pre_integration;

    };

}
#endif
