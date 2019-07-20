#include "extern/project_error.h"
#include "extern/pinhole_project_error.h"
#include "extern/spline_projection_error.h"

#include <iostream>
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>

int main () {
    Pose<double> T0, T1, T2, T3;
    T0.setRandom();
    T1.setRandom();
    T2.setRandom();
    T3.setRandom();

    std::cout << T0.coeffs().transpose() << std::endl;
    std::cout << T1.coeffs().transpose() << std::endl;
    std::cout << T2.coeffs().transpose() << std::endl;
    std::cout << T3.coeffs().transpose() << std::endl;

    double u0 = 0.5;
    double u1 = 0.75;
    Pose<double> T_WI0 = PSUtility::EvaluatePS(u0, T0, T1, T2, T3);
    Pose<double> T_WI1 = PSUtility::EvaluatePS(u1, T0, T1, T2, T3);

    Pose<double> T_IC;
    T_IC.setRandom();

    Eigen::Vector3d C0p = Eigen::Vector3d::Random();
    C0p(2) = std::fabs(C0p(2));

    Pose<double> T_WC0 = T_WI0 * T_IC;
    Pose<double> T_WC1 = T_WI1 * T_IC;



    Eigen::Vector3d Wp = T_WC0 * C0p;
    Eigen::Vector3d C1p = T_WC1.inverse() * Wp;

    Eigen::Vector3d uv0(C0p(0)/ C0p(2), C0p(1)/C0p(2), 1);
    Eigen::Vector3d uv1(C1p(0)/ C1p(2), C1p(1)/C1p(2), 1);

    double rho = 1.0 / C0p(2);

    /*
    * Zero Test
    * Passed!
    */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    Eigen::Isometry3d ext_T_IC;
    ext_T_IC.matrix() = T_IC.Transformation();
    SplineProjectError* splineProjectError = new SplineProjectError(u0, uv0, u1, uv1, ext_T_IC);

    double* param_rho;
    param_rho = &rho;

    double* paramters[5] = {T0.parameterPtr(), T1.parameterPtr(),
                            T2.parameterPtr(), T3.parameterPtr(), param_rho};

    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian3_min;
    Eigen::Matrix<double,2,1> jacobian4_min;
    double* jacobians_min[5] = {jacobian0_min.data(), jacobian1_min.data(),
                                jacobian2_min.data(), jacobian3_min.data(),
                                jacobian4_min.data()};


    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian3;
    Eigen::Matrix<double,2,1> jacobian4;
    double* jacobians[5] = {jacobian0.data(), jacobian1.data(),
                            jacobian2.data(), jacobian3.data(),
                            jacobian4.data()};

    splineProjectError->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";

    return 0;
}