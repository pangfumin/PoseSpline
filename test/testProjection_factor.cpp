#include "cv/projection_factor.hpp"
#include <iostream>
#include "QuaternionSpline/NumbDifferentiator.hpp"
#include "../src/factor/pose_local_parameterization.h"
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

void applyNoise(const Eigen::Isometry3d& Tin,Eigen::Isometry3d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.5*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.16*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;

    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3) =   Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}

int main(){

    // simulate
    Eigen::Isometry3d T_WI, T_IC;
    T_WI.setIdentity();
    Eigen::Quaterniond q_WI(1,3,2,-9);
    q_WI.normalize();

    T_WI.matrix().topLeftCorner(3,3) =q_WI.toRotationMatrix(); // any pose
    T_WI.matrix().topRightCorner(3,1) = Eigen::Vector3d(2,6,1);

    std::cout<<"T_WI: "<<std::endl<<T_WI.matrix()<<std::endl;

    T_IC.setIdentity();
    Eigen::Quaterniond q_IC(-2,50,-1,3);
    q_IC.normalize();

    T_IC.matrix().topLeftCorner(3,3) = q_IC.toRotationMatrix();  // any pose
    T_IC.matrix().topRightCorner(3,1) = Eigen::Vector3d(0.02,0.06,-0.011);
    std::cout<<"T_IC: "<<std::endl<<T_IC.matrix()<<std::endl;



    Eigen::Isometry3d T_WC = T_WI*T_IC;
    Eigen::Matrix3d R_WC = T_WC.matrix().topLeftCorner(3,3);
    Eigen::Vector3d W_t_WC = T_WC.matrix().topRightCorner(3,1);
    std::cout<<"T_WC: "<<std::endl<<T_WC.matrix()<<std::endl;




    Eigen::Vector3d Cp(0.3,-0.6,3); // feature in camera frame
    Eigen::Vector3d Wp = W_t_WC + R_WC*Cp;
    std::cout<<"Wp: "<<Wp.transpose()<<std::endl;



    /*
     * Zero Test
     * Passed!
     */

    Eigen::Vector2d uv_ideal_meas;
    uv_ideal_meas << Cp(0)/Cp(2),Cp(1)/Cp(2);

    Localization_2d3d_factor* localization2d3dFactor = new Localization_2d3d_factor(uv_ideal_meas,Wp,1,1);

    double* param_T_WI = new double[7];
    double* param_T_IC = new double[7];

    T2double(T_WI,param_T_WI);
    T2double(T_IC,param_T_IC);

    double* paramters[2] = {param_T_WI, param_T_IC};
    Eigen::Vector2d residual;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;

    double* jacobians_min[2] = {jacobian0_min.data(), jacobian1_min.data()};
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;

    double* jacobians[2] = {jacobian0.data(), jacobian1.data()};

    localization2d3dFactor->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
    std::cout<<"residual: "<<residual.transpose()<<std::endl;


    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */

    Eigen::Isometry3d T_WI_noised, T_IC_noised;

    applyNoise(T_WI,T_WI_noised);
    applyNoise(T_IC,T_IC_noised);

    double* param_T_WI_noised = new double[7];
    double* param_T_IC_noised = new double[7];

    T2double(T_WI_noised,param_T_WI_noised);
    T2double(T_IC_noised,param_T_IC_noised);

    double* parameters_noised[2] = {param_T_WI_noised, param_T_IC_noised };

    localization2d3dFactor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(),jacobians,jacobians_min);


    std::cout<<"residual: "<<residual.transpose()<<std::endl;


    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian1_min;


    NumbDifferentiiator<Localization_2d3d_factor,2> localizer_num_differ(localization2d3dFactor);

    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameterization>(parameters_noised,0,num_jacobian0_min.data());
    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameterization>(parameters_noised,1,num_jacobian1_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"jacobian1_min: "<<std::endl<<jacobian1_min<<std::endl;

    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;
    std::cout<<"num_jacobian1_min: "<<std::endl<<num_jacobian1_min<<std::endl;

    localizer_num_differ.isJacobianEqual<2,6>(jacobian0_min.data(),num_jacobian0_min.data(),1e-2);
    localizer_num_differ.isJacobianEqual<2,6>(jacobian1_min.data(),num_jacobian1_min.data(),1e-2);




    return 0;
}