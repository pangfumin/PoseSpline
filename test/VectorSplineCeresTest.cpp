#include "pose-spline/VectorSplineSampleError.hpp"
#include "pose-spline/NumbDifferentiator.hpp"

#include "pose-spline/QuaternionSplineUtility.hpp"
#include "pose-spline/VectorSpaceSpline.hpp"

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);

    double u = 0.6;
    Eigen::Vector3d V_meas(0.0,1.0,0.0);


    double u1 = 0.4;
    Eigen::Vector3d V_meas1(0.1,1.0,0.0);

    double u2 = 0.1;
    Eigen::Vector3d V_meas2(0.1,1.3,0.0);


    double u3 = 0.9;
    Eigen::Vector3d V_meas3(0.1,1.3,-0.7);


    double u4 = 0.3;
    Eigen::Vector3d V_meas4(0.1,1.3,-0.732);


    double u5 = 0.5;
    Eigen::Vector3d V_meas5(0.12,1.3,-0.7);



    Eigen::Vector3d Cp0,Cp1,Cp2,Cp3;
    Cp0 = Cp1 = Cp2 = Cp3 = Eigen::Vector3d::Zero();
    Cp0 = Eigen::Vector3d(-0.0233343  ,0.538966 ,  0.805091);
    Cp1 = Eigen::Vector3d(0.142278 ,  0.44318 ,-0.513372);
    Cp2 = Eigen::Vector3d(-0.112329,  0.379688,   0.34445);
    Cp3 = Eigen::Vector3d(-0.164781, -0.303314,  0.876392);

    Eigen::Vector3d Cp0_init,Cp1_init,Cp2_init,Cp3_init;
    Cp0_init = Cp0;
    Cp1_init = Cp1;
    Cp2_init = Cp2;
    Cp3_init = Cp3;

    double* paramters[4] = {Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data()};
    Eigen::Vector3d Residual;

    real_t sample_t = 0.32;
    Eigen::Vector3d sample_V = VectorSpaceSpline::evaluateSpline(sample_t,Cp0,Cp1,Cp2,Cp3);
    std::cout<<"Sample_V: "<<sample_V.transpose()<<std::endl;

    /*
     * Zero test
     */

    Eigen::Matrix<double,3,3,Eigen::RowMajor> AnaliJacobian_minimal0,AnaliJacobian_minimal1,
            AnaliJacobian_minimal2,AnaliJacobian_minimal3;
    double* AnaliJacobians_minimal[4] = {AnaliJacobian_minimal0.data(),
                                         AnaliJacobian_minimal1.data(),
                                         AnaliJacobian_minimal2.data(),
                                         AnaliJacobian_minimal3.data()};
    Eigen::Matrix<double,3,3,Eigen::RowMajor> AnaliJacobian0,AnaliJacobian1,AnaliJacobian2,AnaliJacobian3;
    double* AnaliJacobians[4] = {AnaliJacobian0.data(),
                                 AnaliJacobian1.data(),
                                 AnaliJacobian2.data(),
                                 AnaliJacobian3.data()};

    VectorSplineSampleError* vectorSplineSampleError =  new VectorSplineSampleError(sample_t,sample_V);
    vectorSplineSampleError->EvaluateWithMinimalJacobians(paramters,Residual.data(),AnaliJacobians,AnaliJacobians_minimal);

    std::cout<<"Residual: "<<Residual.transpose()<<std::endl;



    /*
     *  Test jacobians
     */
    std::cout<<"AnaliJacobian_minimal0: "<<std::endl<<AnaliJacobian_minimal0<<std::endl;
    std::cout<<"AnaliJacobian_minimal1: "<<std::endl<<AnaliJacobian_minimal1<<std::endl;
    std::cout<<"AnaliJacobian_minimal2: "<<std::endl<<AnaliJacobian_minimal2<<std::endl;
    std::cout<<"AnaliJacobian_minimal3: "<<std::endl<<AnaliJacobian_minimal3<<std::endl;

    // check jacobian_minimal0
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min0;
    NumbDifferentiator<VectorSplineSampleError,4> numbDifferentiator(vectorSplineSampleError);
    numbDifferentiator.df_r_xi<3,3>(paramters,0,numJacobian_min0.data());

    std::cout<<"numJacobian_min0: "<<std::endl<<numJacobian_min0<<std::endl;
    std::cout<<"AnaliJacobian_minimal0*numJacobian_min0: "<<
                                      std::endl<<AnaliJacobian_minimal0*numJacobian_min0.inverse()<<std::endl;

    // check jacobian_minimal1
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min1;
    numbDifferentiator.df_r_xi<3,3>(paramters,1,numJacobian_min1.data());

    std::cout<<"numJacobian_min1: "<<std::endl<<numJacobian_min1<<std::endl;
    std::cout<<"AnaliJacobian_minimal1*numJacobian_min1: "<<
             std::endl<<AnaliJacobian_minimal1*numJacobian_min1.inverse()<<std::endl;

    // check jacobian_minimal2
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min2;
    numbDifferentiator.df_r_xi<3,3>(paramters,2,numJacobian_min2.data());

    std::cout<<"numJacobian_min2: "<<std::endl<<numJacobian_min2<<std::endl;
    std::cout<<"AnaliJacobian_minimal2*numJacobian_min2: "<<
             std::endl<<AnaliJacobian_minimal2*numJacobian_min2.inverse()<<std::endl;

    // check jacobian_minimal3
    Eigen::Matrix<double,3,3,Eigen::RowMajor> numJacobian_min3;
    numbDifferentiator.df_r_xi<3,3>(paramters,3,numJacobian_min3.data());

    std::cout<<"numJacobian_min3: "<<std::endl<<numJacobian_min3<<std::endl;
    std::cout<<"AnaliJacobian_minimal3*numJacobian_min3: "<<
             std::endl<<AnaliJacobian_minimal3*numJacobian_min3.inverse()<<std::endl;

    /*
     * Test opt
     */
    // solve
    ceres::Problem problem;
    ceres::LossFunction* loss_function =  new ceres::HuberLoss(1.0);

    problem.AddParameterBlock(Cp0.data(), 3);
    problem.AddParameterBlock(Cp1.data(), 3);
    problem.AddParameterBlock(Cp2.data(), 3);
    problem.AddParameterBlock(Cp3.data(), 3);

    VectorSplineSampleError* vectorSplineSampleError0 = new VectorSplineSampleError(u1,V_meas);
    VectorSplineSampleError* vectorSplineSampleError1 = new VectorSplineSampleError(u1,V_meas1);
    VectorSplineSampleError* vectorSplineSampleError2 = new VectorSplineSampleError(u2,V_meas2);
    VectorSplineSampleError* vectorSplineSampleError3 = new VectorSplineSampleError(u3,V_meas3);
    VectorSplineSampleError* vectorSplineSampleError4 = new VectorSplineSampleError(u4,V_meas4);
    VectorSplineSampleError* vectorSplineSampleError5 = new VectorSplineSampleError(u5,V_meas5);

    problem.AddResidualBlock(vectorSplineSampleError0, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(vectorSplineSampleError1, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(vectorSplineSampleError2, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(vectorSplineSampleError3, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(vectorSplineSampleError4, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());
    problem.AddResidualBlock(vectorSplineSampleError5, loss_function, Cp0.data(),Cp1.data(),Cp2.data(),Cp3.data());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout<<"--------------"<<std::endl;
    std::cout << summary.FullReport() << std::endl;

    std::cout<<"CP0_init: "<<Cp0_init.transpose()<<std::endl<<"After opt: "<<Cp0.transpose()<<std::endl;
    std::cout<<"CP1_init: "<<Cp1_init.transpose()<<std::endl<<"After opt: "<<Cp1.transpose()<<std::endl;
    std::cout<<"CP2_init: "<<Cp2_init.transpose()<<std::endl<<"After opt: "<<Cp2.transpose()<<std::endl;
    std::cout<<"CP3_init: "<<Cp3_init.transpose()<<std::endl<<"After opt: "<<Cp3.transpose()<<std::endl;

    Eigen::Vector3d Vhat0 = VectorSpaceSpline::evaluateSpline(u,Cp0,Cp1,Cp2,Cp3);
    Eigen::Vector3d Vhat1 = VectorSpaceSpline::evaluateSpline(u1,Cp0,Cp1,Cp2,Cp3);
    Eigen::Vector3d Vhat2 = VectorSpaceSpline::evaluateSpline(u2,Cp0,Cp1,Cp2,Cp3);
    Eigen::Vector3d Vhat3 = VectorSpaceSpline::evaluateSpline(u3,Cp0,Cp1,Cp2,Cp3);

    std::cout<<"Vmeas0: "<<V_meas.transpose()<<std::endl<<"Vhat0: "<<Vhat0.transpose()<<std::endl;
    std::cout<<"Vmeas1: "<<V_meas1.transpose()<<std::endl<<"Vhat1: "<<Vhat1.transpose()<<std::endl;
    std::cout<<"Vmeas2: "<<V_meas2.transpose()<<std::endl<<"Vhat2: "<<Vhat2.transpose()<<std::endl;
    std::cout<<"Vmeas3: "<<V_meas3.transpose()<<std::endl<<"Vhat3: "<<Vhat3.transpose()<<std::endl;

    return 0;
}