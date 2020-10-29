#include "project_error.h"
#include <iostream>
#include "NumbDifferentiator.hpp"
#include "pose_local_parameterization.h"
void T2double(Eigen::Isometry3d& T,double* ptr){

    Eigen::Vector3d trans = T.matrix().topRightCorner(3,1);
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Quaterniond q(R);

    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);

}

void applyNoise(const Eigen::Isometry3d& Tin,Eigen::Isometry3d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.5*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.16*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;

    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3) = Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}

int main(){

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

    Eigen::Matrix3d R = T_WC0.matrix().topLeftCorner(3,3);
    Eigen::Quaterniond Q_QC(R);



    /*
     * Zero Test
     * Passed!
     */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    ProjectError* projectFactor = new ProjectError(p0, Wp,Q_QC);

    double* param_T_WC0 = new double[3];



    T2double(T_WC0,param_T_WC0);

    double* paramters[1] = {param_T_WC0};

    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian0_min;

    double* jacobians_min[1] = {jacobian0_min.data()};


    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian0;
    double* jacobians[1] = {jacobian0.data()};

    projectFactor->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */
//
//
    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;

    Eigen::Isometry3d T_WC0_noised;

    applyNoise(T_WC0,T_WC0_noised);

    double* param_T_WC0_noised = new double[3];



    T2double(T_WC0_noised,param_T_WC0_noised);


    double* parameters_noised[1] = {param_T_WC0_noised};

    projectFactor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(),jacobians,jacobians_min);


    std::cout<<"residual: "<<residual.transpose()<<std::endl;

    Eigen::Matrix<double,2,3,Eigen::RowMajor> num_jacobian0_min;

    NumbDifferentiator<ProjectError,1> num_differ(projectFactor);

    num_differ.df_r_xi<2,3>(parameters_noised,0,num_jacobian0_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;



    return 0;
}
