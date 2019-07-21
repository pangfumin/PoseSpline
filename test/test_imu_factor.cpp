#include "extern/spline_imu_error.h"
#include "PoseSpline/Time.hpp"
#include "PoseSpline/Pose.hpp"
#include <iostream>
struct ImuMeas {
    Time ts_;
    Eigen::Vector3d acc_;
    Eigen::Vector3d gyro_;
};

typedef Eigen::Matrix<double,9,1> SpeedAndBias;

double sinc_test(double x){
    if(fabs(x)>1e-10) {
        return sin(x)/x;
    }
    else{
        static const double c_2=1.0/6.0;
        static const double c_4=1.0/120.0;
        static const double c_6=1.0/5040.0;
        const double x_2 = x*x;
        const double x_4 = x_2*x_2;
        const double x_6 = x_2*x_2*x_2;
        return 1.0 - c_2*x_2 + c_4*x_4 - c_6*x_6;
    }
}
std::vector<ImuMeas> simulateImuMeas(const Pose<double> T_WS0, Pose<double>& T_WS1,
        const Eigen::Vector3d& v_WS0, Eigen::Vector3d& v_WS1 ) {

    // generate random motion
    const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
    const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
    const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
    const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
    const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
    const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
    const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
    const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
    const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
    const double w_a_W_x = Eigen::internal::random(0.1,10.0);
    const double w_a_W_y = Eigen::internal::random(0.1,10.0);
    const double w_a_W_z = Eigen::internal::random(0.1,10.0);
    const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
    const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
    const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
    const double m_a_W_x = Eigen::internal::random(0.1,10.0);
    const double m_a_W_y = Eigen::internal::random(0.1,10.0);
    const double m_a_W_z = Eigen::internal::random(0.1,10.0);

    // generate randomized measurements - duration 10 seconds
    const double duration = 1.0;
    std::vector<ImuMeas> imuMeasurements;

    int rate = 200;
    // time increment
    const double dt=1.0/double(rate); // time discretization


    // states
    Eigen::Matrix3d R_WS =T_WS0.C();
    Eigen::Matrix<double,4,1> q_WS =T_WS0.q();
    Eigen::Vector3d r_WS=T_WS0.r();

    Eigen::Vector3d v=v_WS0;

    for(size_t i=0; i<size_t(duration*rate); ++i){
        double time = double(i)/rate;


        Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                                m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                                m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
        Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                            m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                            m_a_W_z*sin(w_a_W_z*time+p_a_W_z));
        Quaternion dq;
        // propagate orientation
        const double theta_half = omega_S.norm()*dt*0.5;
        const double sinc_theta_half = sinc_test(theta_half);
        const double cos_theta_half = cos(theta_half);
        dq.head(3)=sinc_theta_half*0.5*dt*omega_S;
        dq(3)=cos_theta_half;
        q_WS = quatMult(q_WS, dq);

        // propagate speed
        v+=dt*a_W;

        v_WS1 = v;

        // propagate position
        r_WS+=dt*v;

        // T_WS
        T_WS1 = Pose<double>(r_WS,q_WS);

        // speedAndBias - v only, obviously, since this is the Ground Truth


        double g = 9.81;
        // generate measurements
        Eigen::Vector3d gyr = omega_S ;
        Eigen::Vector3d acc = T_WS1.inverse().C()*(a_W+Eigen::Vector3d(0,0,g));
        imuMeasurements.push_back({Time(time), acc, gyr});
    }
    return  imuMeasurements;
};

void T2double(Eigen::Isometry3d& T,double* ptr){

    Eigen::Vector3d trans = T.matrix().topRightCorner(3,1);
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Quaterniond q(R);


    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);
    ptr[3] = q.x();
    ptr[4] = q.y();
    ptr[5] = q.z();
    ptr[6] = q.w();
}
int main () {
    Pose<double> T_WS0, T_WS1;
    T_WS0.setRandom();
    Eigen::Vector3d v0(0,0,0), v1;
    std::vector<ImuMeas> imuMeas_vec = simulateImuMeas(T_WS0, T_WS1, v0, v1);
//    for (auto i : imuMeas_vec) {
//        std::cout << i.ts_.toNSec() <<" " << i.gyro_.transpose() << " " << i.acc_.transpose() << std::endl;
//    }

    std::cout << T_WS0.parameters().transpose() << std::endl;
    std::cout << T_WS1.parameters().transpose() << std::endl;

    ImuParam imuParam;
    Eigen::Vector3d gyro_bias, acc_bias;
    gyro_bias.setZero();
    acc_bias.setZero();
    IntegrationBase imuIntegrate(imuMeas_vec.front().acc_, imuMeas_vec.front().gyro_, gyro_bias, acc_bias, imuParam);
    for (int i = 1; i < imuMeas_vec.size(); i ++) {
        double dt = imuMeas_vec.at(i).ts_.toSec() - imuMeas_vec.at(i-1).ts_.toSec();
        imuIntegrate.push_back(dt, imuMeas_vec.at(i).acc_, imuMeas_vec.at(i).gyro_);
    }

    IMUFactor imuFactor(&imuIntegrate);

    Eigen::Matrix4d pose0 = T_WS0.Transformation();
    Eigen::Matrix4d pose1 = T_WS1.Transformation();
    Eigen::Isometry3d eigen_T_WS0, eigen_T_WS1;
    eigen_T_WS0.matrix() = pose0;
    eigen_T_WS1.matrix() = pose1;

    SpeedAndBias sb0, sb1;
    sb0.setZero(); sb0.head(3) = v0;
    sb1.setZero(); sb1.head(3) = v1;

    double* param_T_WS0 = new double[7];
    double* param_T_WS1 = new double[7];

    T2double(eigen_T_WS0, param_T_WS0);
    T2double(eigen_T_WS1, param_T_WS1);

    double* paramters[4] = {param_T_WS0, sb0.data(), param_T_WS1, sb1.data()};
    Eigen::Matrix<double, 15,1> residual;
    imuFactor.Evaluate(paramters, residual.data(), NULL);
    std::cout << residual.transpose() << std::endl;









    return 0;
}