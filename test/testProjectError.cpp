#include "extern/project_error.h"
#include "extern/pinhole_project_error.h"

#include <iostream>
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>
void T2double(Eigen::Isometry3d& T,double* ptr){

    Eigen::Vector3d trans = T.matrix().topRightCorner(3,1);
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Quaternion q = rotMatToQuat<double>(R);


    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);
    ptr[3] = q(0);
    ptr[4] = q(1);
    ptr[5] = q(2);
    ptr[6] = q(3);
}

void applyNoise(const Eigen::Isometry3d& Tin,Eigen::Isometry3d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.5*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.16*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;

    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3) = Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}

TEST(ceres, projectError){
    // simulate
    Eigen::Isometry3d T_WI0, T_WI1, T_IC;
    Eigen::Vector3d C0p(4, 3, 10);
    T_WI0 = T_WI1 = T_IC = Eigen::Isometry3d::Identity();

    T_WI1.matrix().topRightCorner(3,1) = Eigen::Vector3d(1,0,0);
    T_IC.matrix().topRightCorner(3,1) = Eigen::Vector3d(0.1,0.10,0);

    Eigen::Isometry3d T_WC0, T_WC1;
    T_WC0 = T_WI0 * T_IC;
    T_WC1 = T_WI1 * T_IC;

    Eigen::Isometry3d T_C0C1 = T_WC0.inverse()*T_WC1;
    Eigen::Isometry3d T_C1C0 = T_C0C1.inverse();

    Eigen::Vector3d Wp = T_WC0.matrix().topLeftCorner(3,3)*C0p+T_WC0.matrix().topRightCorner(3,1);

    Eigen::Vector3d C1p = T_C1C0.matrix().topLeftCorner(3,3)*C0p+T_C1C0.matrix().topRightCorner(3,1);

    Eigen::Vector3d p0(C0p(0)/C0p(2), C0p(1)/C0p(2), 1);
    double z = C0p(2);
    double rho = 1.0/z;

    Eigen::Vector3d p1(C1p(0)/C1p(2), C1p(1)/C1p(2), 1);



    /*
     * Zero Test
     * Passed!
     */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    ProjectError* projectFactor = new ProjectError(p0);

    double* param_T_WC0 = new double[7];



    T2double(T_WC0,param_T_WC0);

    double* paramters[2] = {param_T_WC0, Wp.data()};

    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian1_min;
    double* jacobians_min[2] = {jacobian0_min.data(), jacobian1_min.data()};


    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian1;
    double* jacobians[2] = {jacobian0.data(), jacobian1.data()};

    projectFactor->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */


    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;

    Eigen::Isometry3d T_WC0_noised;

    applyNoise(T_WC0,T_WC0_noised);

    double* param_T_WC0_noised = new double[7];



    T2double(T_WC0_noised,param_T_WC0_noised);


    double* parameters_noised[2] = {param_T_WC0_noised,Wp.data()};

    projectFactor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(),jacobians,jacobians_min);


    std::cout<<"residual: "<<residual.transpose()<<std::endl;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian0_min;
    Eigen::Matrix<double,2,3,Eigen::RowMajor> num_jacobian1_min;

    NumbDifferentiator<ProjectError,2> num_differ(projectFactor);

    num_differ.df_r_xi<2,7,6,PoseLocalParameter>(parameters_noised,0,num_jacobian0_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;


    GTEST_ASSERT_LT((jacobian0_min - num_jacobian0_min).norm(), 1e6);

//
    num_differ.df_r_xi<2,3>(parameters_noised,1,num_jacobian1_min.data());

    std::cout<<"jacobian1_min: "<<std::endl<<jacobian1_min<<std::endl;
    std::cout<<"num_jacobian1_min: "<<std::endl<<num_jacobian1_min<<std::endl;
    GTEST_ASSERT_LT((jacobian1_min - num_jacobian1_min).norm(), 1e6);

}

TEST (ceres, PinholePRojectError) {
    // simulate

    Eigen::Isometry3d T_WI0, T_WI1, T_IC;
    Eigen::Vector3d C0p(4, 3, 10);
    T_WI0 = T_WI1 = T_IC = Eigen::Isometry3d::Identity();

    T_WI1.matrix().topRightCorner(3,1) = Eigen::Vector3d(1,0,0);
    T_IC.matrix().topRightCorner(3,1) = Eigen::Vector3d(0.1,0.10,0);

    Eigen::Isometry3d T_WC0, T_WC1;
    T_WC0 = T_WI0 * T_IC;
    T_WC1 = T_WI1 * T_IC;

    Eigen::Isometry3d T_C0C1 = T_WC0.inverse()*T_WC1;
    Eigen::Isometry3d T_C1C0 = T_C0C1.inverse();

    Eigen::Vector3d C1p = T_C1C0.matrix().topLeftCorner(3,3)*C0p+T_C1C0.matrix().topRightCorner(3,1);

    Eigen::Vector3d p0(C0p(0)/C0p(2), C0p(1)/C0p(2), 1);
    double z = C0p(2);
    double rho = 1.0/z;

    Eigen::Vector3d p1(C1p(0)/C1p(2), C1p(1)/C1p(2), 1);



    /*
     * Zero Test
     * Passed!
     */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    PinholeProjectError* pinholeProjectFactor = new PinholeProjectError(p0,p1,T_IC);

    double* param_T_WI0 = new double[7];
    double* param_T_WI1 = new double[7];
    double* param_rho;


    T2double(T_WI0,param_T_WI0);
    T2double(T_WI1,param_T_WI1);
    param_rho = &rho;



    double* paramters[3] = {param_T_WI0, param_T_WI1, param_rho};

    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
    Eigen::Matrix<double,2,1> jacobian2_min;
    double* jacobians_min[3] = {jacobian0_min.data(), jacobian1_min.data(),
                                jacobian2_min.data()};


    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
    Eigen::Matrix<double,2,1> jacobian2;
    double* jacobians[3] = {jacobian0.data(), jacobian1.data(), jacobian2.data()};

    pinholeProjectFactor->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */


    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;

    Eigen::Isometry3d T_WI0_noised,  T_WI1_noised;
    double rho_noised;

    applyNoise(T_WI0,T_WI0_noised);
    applyNoise(T_WI1,T_WI1_noised);
    rho_noised = rho ;

    double* param_T_WI0_noised = new double[7];
    double* param_T_WI1_noised = new double[7];
    double* param_rho_noised;



    T2double(T_WI0_noised,param_T_WI0_noised);
    T2double(T_WI1_noised,param_T_WI1_noised);

    param_rho_noised = &rho_noised;

    double* parameters_noised[3] = {param_T_WI0_noised ,param_T_WI1_noised,
                                    param_rho_noised};

    pinholeProjectFactor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(),jacobians,jacobians_min);


    std::cout<<"residual: "<<residual.transpose()<<std::endl;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian1_min;
    Eigen::Matrix<double,2,1> num_jacobian2_min;

    NumbDifferentiator<PinholeProjectError,3> localizer_num_differ(pinholeProjectFactor);

    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameter>(parameters_noised,0,num_jacobian0_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;

    GTEST_ASSERT_LT((jacobian0_min - num_jacobian0_min).norm(), 1e6);


//    std::cout<<"Check jacobian0: "<<std::endl;
//    localizer_num_differ.isJacobianEqual<2,6>(jacobian0_min.data(),num_jacobian0_min.data(),1e-2);
//
    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameter>(parameters_noised,1,num_jacobian1_min.data());

    std::cout<<"jacobian1_min: "<<std::endl<<jacobian1_min<<std::endl;
    std::cout<<"num_jacobian1_min: "<<std::endl<<num_jacobian1_min<<std::endl;

    GTEST_ASSERT_LT((jacobian1_min - num_jacobian1_min).norm(), 1e6);


//
//    std::cout<<"Check jacobian1: "<<std::endl;
//    localizer_num_differ.isJacobianEqual<2,6>(jacobian1_min.data(),num_jacobian1_min.data(),1e-2);
//
    localizer_num_differ.df_r_1D_x<2>(parameters_noised,2,num_jacobian2_min.data());

    std::cout<<"jacobian2_min: "<<std::endl<<jacobian2_min<<std::endl;
    std::cout<<"num_jacobian2_min: "<<std::endl<<num_jacobian2_min<<std::endl;

    GTEST_ASSERT_LT((jacobian2_min - num_jacobian2_min).norm(), 1e6);

}
