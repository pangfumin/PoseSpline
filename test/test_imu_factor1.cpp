#include <iostream>
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/PoseSpline.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"
#include "extern/ImuFactor.hpp"

struct State{
    double ts_;
    Pose<double> T_;
    Eigen::Vector3d v_;
    Eigen::Vector3d acc_;
    Eigen::Vector3d gyr_;
};

std::vector<State> generateImuReadings () {
    std::vector<State> states;

    PoseSpline  poseSpline(1.0);
    double dt  = 0.2;
    std::vector<std::pair<double, Pose<double>>> meas;

    for (int i = 0 ; i < 4; i ++ ) {
        Pose<double> T_WS_k;
        T_WS_k.setRandom(1, 1.0);
        meas.push_back(std::make_pair(7.0 + i * dt, T_WS_k));
    }

    poseSpline.initialPoseSpline(meas);

    int rate = 200;
    double sample_dt = 1.0/200;
    double start_ts = poseSpline.t_min();
    for (int i = 1; i < rate - 1; i ++ ) {
        Pose<double> T = poseSpline.evalPoseSpline(start_ts + i * sample_dt);
        Eigen::Vector3d v = poseSpline.evalLinearVelocity(start_ts + i * sample_dt);
        Eigen::Vector3d acc = poseSpline.evalLinearAccelerator(start_ts + i * sample_dt);
        Eigen::Vector3d gyr = poseSpline.evalOmega(start_ts + i * sample_dt);
//        std::cout << acc.transpose() << std::endl;
//
        State state{start_ts + i * sample_dt, T, v, acc, gyr};
        states.push_back(state);
    }

    return states;


}
int main() {
    std::cout << "test imu factor " << std::endl;
    std::vector<State> states = generateImuReadings();


    auto start_ts = Time(states.front().ts_);
    auto end_ts = Time(states.back().ts_);
    okvis::ImuMeasurementDeque imuMeasurementDeque;
    for (auto i : states) {
        okvis::ImuMeasurement imuMeasurement;
        imuMeasurement.timeStamp  = Time(i.ts_);
        imuMeasurement.measurement.accelerometers = i.acc_;
        imuMeasurement.measurement.gyroscopes = i.gyr_;
        imuMeasurementDeque.push_back(imuMeasurement);
    }
    okvis::ImuParameters imuParameters;
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    imuParameters.rate = 1000; // 1 kHz
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;
    imuParameters.tau = 3600.0;

    okvis::kinematics::Transformation T_WS;
    T_WS.set(states.front().T_.Transformation());
    std::cout << T_WS.parameters().transpose() << std::endl;

    okvis::SpeedAndBias speedAndBias;
    speedAndBias << states.front().v_, 0,0,0,0,0,0;
    okvis::ceres::ImuError::propagation(imuMeasurementDeque, imuParameters,T_WS,speedAndBias,start_ts, end_ts);

    std::cout << states.front().T_.parameters().transpose() << std::endl;
    std::cout << T_WS.parameters().transpose() << std::endl;
    std::cout << states.back().T_.parameters().transpose() << std::endl;



    return 0;
}