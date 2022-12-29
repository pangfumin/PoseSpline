#include "PoseSpline/AngularVelocitySampleError.hpp"
#include "PoseSpline/NumbDifferentiator.hpp"
#include "PoseSpline/PoseSplineUtility.hpp"

#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "geometry/Pose.hpp"

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);

    double u = 0.6;
    Quaternion Q_meas(0.0,1.0,0.0,1.9);
    Q_meas = quatNorm(Q_meas);

    double u1 = 0.4;
    Quaternion Q_meas1(0.1,1.0,0.0,1.9);
    Q_meas1 = quatNorm(Q_meas1);

    double u2 = 0.1;
    Quaternion Q_meas2(0.1,1.3,0.0,1.9);
    Q_meas2 = quatNorm(Q_meas2);

    double u3 = 0.9;
    Quaternion Q_meas3(0.1,1.3,-0.7,1.9);
    Q_meas3 = quatNorm(Q_meas3);

    double u4 = 0.3;
    Quaternion Q_meas4(0.1,1.3,-0.732,1.9);
    Q_meas4 = quatNorm(Q_meas4);

    double u5 = 0.5;
    Quaternion Q_meas5(0.12,1.3,-0.7,1.9);
    Q_meas5 = quatNorm(Q_meas5);


    Quaternion Cp0,Cp1,Cp2,Cp3;
    Cp0 = Cp1 = Cp2 = Cp3 = unitQuat<double>();
    Cp0 = Quaternion(-0.0233343  ,0.538966 ,  0.805091,   0.246575); Cp0 = quatNorm(Cp0);
    Cp1 = Quaternion(0.142278 ,  0.44318 ,-0.513372 ,  0.72097);  Cp1 = quatNorm(Cp1);
    Cp2 = Quaternion(-0.112329,  0.379688,   0.34445,  0.851219);  Cp2 = quatNorm(Cp2);
    Cp3 = Quaternion(-0.164781, -0.303314,  0.876392, -0.335836);  Cp3 = quatNorm(Cp3);

    Quaternion Cp0_init,Cp1_init,Cp2_init,Cp3_init;
    Cp0_init = Cp0;
    Cp1_init = Cp1;
    Cp2_init = Cp2;
    Cp3_init = Cp3;

    Pose<double> pose0(Eigen::Vector3d::Zero(), Cp0);
    Pose<double> pose1(Eigen::Vector3d::Zero(), Cp1);
    Pose<double> pose2(Eigen::Vector3d::Zero(), Cp2);
    Pose<double> pose3(Eigen::Vector3d::Zero(), Cp3);

    Eigen::Vector3d bw0 = Eigen::Vector3d::Random();
    Eigen::Vector3d bw1 = Eigen::Vector3d::Random();
    Eigen::Vector3d bw2 = Eigen::Vector3d::Random();
    Eigen::Vector3d bw3 = Eigen::Vector3d::Random();

    Eigen::Vector3d bias_omega = PSUtility::EvaluatePosition(0.5, bw0,bw1,bw2,bw3);
    std::cout<<"bias_omega: "<<bias_omega.transpose() << std::endl;





    Quaternion Q_ba = QSUtility::EvaluateQS(0.5,Cp0,Cp1,Cp2,Cp3);
    Quaternion dot_Q_ba = QSUtility::Evaluate_dot_QS(1, 0.5,Cp0,Cp1,Cp2,Cp3);
    Eigen::Vector3d omega = QSUtility::w_in_body_frame(Q_ba, dot_Q_ba) + bias_omega;
    std::cout<<"Omega: "<< omega.transpose() << std::endl;


    double* paramters[8] = {pose0.data(),pose1.data(),pose2.data(),pose3.data(),
                            bw0.data(), bw1.data(), bw2.data(), bw3.data()};
    Eigen::Vector3d Residual;

    QuaternionOmegaSampleFunctor* quaternionOmegaSampleFunctor
            = new QuaternionOmegaSampleFunctor(0.5, 1.0, omega, 1.0);

    AngularVelocitySampleAutoError* angularVelocitySampleAutoError
            = new AngularVelocitySampleAutoError(quaternionOmegaSampleFunctor);



    Eigen::Matrix<double,3,6,Eigen::RowMajor> AnaliJacobian_minimal0,AnaliJacobian_minimal1,AnaliJacobian_minimal2,AnaliJacobian_minimal3;
    Eigen::Matrix<double,3,3,Eigen::RowMajor> AnaliJacobian_minimal4,AnaliJacobian_minimal5,AnaliJacobian_minimal6,AnaliJacobian_minimal7;
    double* AnaliJacobians_minimal[8] = {AnaliJacobian_minimal0.data(),
                                         AnaliJacobian_minimal1.data(),
                                         AnaliJacobian_minimal2.data(),
                                         AnaliJacobian_minimal3.data(),
                                         AnaliJacobian_minimal4.data(),
                                         AnaliJacobian_minimal5.data(),
                                         AnaliJacobian_minimal6.data(),
                                         AnaliJacobian_minimal7.data()};
    Eigen::Matrix<double,3,7,Eigen::RowMajor> AnaliJacobian0,AnaliJacobian1,AnaliJacobian2,AnaliJacobian3;
    Eigen::Matrix<double,3,3,Eigen::RowMajor> AnaliJacobian4,AnaliJacobian5,AnaliJacobian6,AnaliJacobian7;
    double* AnaliJacobians[8] = {AnaliJacobian0.data(),
                                 AnaliJacobian1.data(),
                                 AnaliJacobian2.data(),
                                 AnaliJacobian3.data(),
                                 AnaliJacobian4.data(),
                                 AnaliJacobian5.data(),
                                 AnaliJacobian6.data(),
                                 AnaliJacobian7.data()};


    /*
     *  Test Zero
     */

    angularVelocitySampleAutoError->EvaluateWithMinimalJacobians(paramters,Residual.data(),
                                                                 AnaliJacobians,AnaliJacobians_minimal);

    std::cout<<"Residual: "<<Residual.transpose() << std::endl;
//



//
    /*
     *  Test jacobians
     */
    Quaternion noise0, noise1, noise2, noise3;
    Quaternion CP_noise0, CP_noise1, CP_noise2, CP_noise3;
    noise0 = Quaternion(0.1,0,0,1);
    noise1 = Quaternion(0.1,0.1,0,1);
    noise2 = Quaternion(0.1,0,0.1,1);
    noise3 = Quaternion(0,0,0.1,1);
    CP_noise0 = quatMult(Cp0,noise0);CP_noise0 = quatNorm(CP_noise0);
    CP_noise1 = quatMult(Cp1,noise1);CP_noise1 = quatNorm(CP_noise1);
    CP_noise2 = quatMult(Cp2,noise2);CP_noise2 = quatNorm(CP_noise2);
    CP_noise3 = quatMult(Cp3,noise3);CP_noise3 = quatNorm(CP_noise3);

    Pose<double> pose0_noised(Eigen::Vector3d::Zero(), CP_noise0);
    Pose<double> pose1_noised(Eigen::Vector3d::Zero(), CP_noise1);
    Pose<double> pose2_noised(Eigen::Vector3d::Zero(), CP_noise2);
    Pose<double> pose3_noised(Eigen::Vector3d::Zero(), CP_noise3);

    Eigen::Vector3d bw0_noised = Eigen::Vector3d::Random();
    Eigen::Vector3d bw1_noised = Eigen::Vector3d::Random();
    Eigen::Vector3d bw2_noised = Eigen::Vector3d::Random();
    Eigen::Vector3d bw3_noised = Eigen::Vector3d::Random();


    double* paramters_noised[8] = {pose0_noised.data(),pose1_noised.data(),pose2_noised.data(),pose3_noised.data(),
                                   bw0_noised.data(), bw1_noised.data(), bw2_noised.data(), bw3_noised.data()};

    angularVelocitySampleAutoError->EvaluateWithMinimalJacobians(paramters_noised,Residual.data(),
                                                                 AnaliJacobians,AnaliJacobians_minimal);


//
    // check jacobian_minimal0
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<AngularVelocitySampleAutoError,8> numbDifferentiator(angularVelocitySampleAutoError);
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0: "<<
                                      std::endl<<AnaliJacobian_minimal0<<std::endl<<std::endl;
//
     // check jacobian_minimal1
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<
             std::endl<<AnaliJacobian_minimal1<<std::endl<<std::endl;


////
    // check jacobian_minimal2
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<
             std::endl<<AnaliJacobian_minimal2<<std::endl<<std::endl;

    // check jacobian_minimal3
    Eigen::Matrix<double,3,6,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,7,6,PoseLocalParameter>(paramters_noised,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<
             std::endl<<AnaliJacobian_minimal3<<std::endl<<std::endl;

    // check jacobian_minimal4
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min4;
    numbDifferentiator.df_r_xi<3,3>(paramters_noised,4,numJacobian_min4.data());

    std::cout<<"numJacobian_min4: "<<std::endl<<numJacobian_min4<<std::endl;
    std::cout<<"AnaliJacobian_minimal4: "<<
             std::endl<<AnaliJacobian_minimal4<<std::endl<<std::endl;

    // check jacobian_minimal5
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min5;
    numbDifferentiator.df_r_xi<3,3>(paramters_noised,5,numJacobian_min5.data());

    std::cout<<"numJacobian_min5: "<<std::endl<<numJacobian_min5<<std::endl;
    std::cout<<"AnaliJacobian_minimal5: "<<
             std::endl<<AnaliJacobian_minimal5<<std::endl<<std::endl;

    // check jacobian_minimal6
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min6;
    numbDifferentiator.df_r_xi<3,3>(paramters_noised,6,numJacobian_min6.data());

    std::cout<<"numJacobian_min6: "<<std::endl<<numJacobian_min6<<std::endl;
    std::cout<<"AnaliJacobian_minimal6: "<<
             std::endl<<AnaliJacobian_minimal6<<std::endl<<std::endl;

    // check jacobian_minimal7
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min7;
    numbDifferentiator.df_r_xi<3,3>(paramters_noised,7,numJacobian_min7.data());

    std::cout<<"numJacobian_min7: "<<std::endl<<numJacobian_min7<<std::endl;
    std::cout<<"AnaliJacobian_minimal7: "<<
             std::endl<<AnaliJacobian_minimal7<<std::endl<<std::endl;



    return 0;
}