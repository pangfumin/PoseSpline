#include "pose-spline/AngularVelocitySampleError.hpp"
#include "pose-spline/QuaternionLocalParameter.hpp"
#include "pose-spline/PoseSplineUtility.hpp"

bool AngularVelocitySampleAutoError::Evaluate(double const* const* parameters,
              double* residuals,
              double** jacobians) const {

    return EvaluateWithMinimalJacobians(parameters, residuals,jacobians, NULL);

}

bool AngularVelocitySampleAutoError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                  double* residuals,
                                  double** jacobians,
                                  double** jacobiansMinimal) const {

    Eigen::Map<const Quaternion> Q0(parameters[0] + 3);
    Eigen::Map<const Quaternion> Q1(parameters[1] + 3);
    Eigen::Map<const Quaternion> Q2(parameters[2] + 3);
    Eigen::Map<const Quaternion> Q3(parameters[3] + 3);

    Eigen::Map<const Eigen::Vector3d> bw0(parameters[4]);
    Eigen::Map<const Eigen::Vector3d> bw1(parameters[5]);
    Eigen::Map<const Eigen::Vector3d> bw2(parameters[6]);
    Eigen::Map<const Eigen::Vector3d> bw3(parameters[7]);


    double u = functor_->getU();
    //std::cout<<"u " << u << std::endl;
    double  Beta1 = QSUtility::beta1(u);
    double  Beta2 = QSUtility::beta2(u);
    double  Beta3 = QSUtility::beta3(u);

    // define residual
    // For simplity, we define error  =  /hat - meas.
    Eigen::Vector3d bias =  bw0 + Beta1*(bw1 - bw0) +  Beta2*(bw2 - bw1) + Beta3*(bw3 - bw2);

    //std::cout<<"bias: "<<bias.transpose() << std::endl;


    const double* temp_parameters[4] = {Q0.data(), Q1.data(), Q2.data(), Q3.data()};
    Eigen::Vector3d temp_residual;
    Eigen::Map<Eigen::Vector3d> error(residuals);


    if (!jacobians) {
        bool success = ceres::internal::VariadicEvaluate<
                QuaternionOmegaSampleFunctor, double, 4, 4, 4, 4, 0,0,0,0,0,0>
        ::Call(*functor_, temp_parameters, temp_residual.data());


        error = temp_residual + bias;

        return success;
    }

    Eigen::Matrix<double, 3,4, Eigen::RowMajor> quaternion_jacobian0,quaternion_jacobian1,
                                                 quaternion_jacobian2, quaternion_jacobian3;
    double* quaternion_jacobians[4]
                = {quaternion_jacobian0.data(), quaternion_jacobian1.data(),
                    quaternion_jacobian2.data(), quaternion_jacobian3.data()};
    

    bool success =  ceres::internal::AutoDiff<QuaternionOmegaSampleFunctor, double,
            4,4,4,4>::Differentiate(
            *functor_,
            temp_parameters,
            QuaternionOmegaSampleFunctor::num_residuals(),
            temp_residual.data(),
            quaternion_jacobians);

    error = temp_residual + bias;

    if (success && jacobiansMinimal!= NULL) {
        if( jacobiansMinimal[0] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J0_minimal_map(jacobiansMinimal[0]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q0.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J0_map(jacobians[0]);
            J0_minimal_map << Eigen::Matrix3d::Zero(), quaternion_jacobian0*J_plus;
        }

        if( jacobiansMinimal[1] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J1_minimal_map(jacobiansMinimal[1]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q1.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J1_map(jacobians[1]);
            J1_minimal_map << Eigen::Matrix3d::Zero(), quaternion_jacobian1*J_plus;
        }

        if( jacobiansMinimal[2] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J2_minimal_map(jacobiansMinimal[2]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q2.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J2_map(jacobians[2]);
            J2_minimal_map  << Eigen::Matrix3d::Zero(), quaternion_jacobian2*J_plus;
        }

        if( jacobiansMinimal[3] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J3_minimal_map(jacobiansMinimal[3]);
            Eigen::Matrix<double,4,3,Eigen::RowMajor> J_plus;
            QuaternionLocalParameter::plusJacobian(Q3.data(),J_plus.data());
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J3_map(jacobians[3]);
            J3_minimal_map  << Eigen::Matrix3d::Zero(), quaternion_jacobian3*J_plus;
        }

        //

        if (jacobians[4] != NULL) {
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J4(jacobians[0]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J4_minimal;

            J4_minimal  = (1 - Beta1)*Eigen::Matrix3d::Identity();
            J4 = J4_minimal ;
            if(jacobiansMinimal != NULL && jacobiansMinimal[4] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J4_minimal_map(jacobiansMinimal[4]);
                J4_minimal_map = J4_minimal;

                //std::cout<<"J0_minimal_map: "<<std::endl<<J0_minimal_map<<std::endl;

            }
        }

        if(jacobians[5] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J5(jacobians[5]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J5_minimal;

            J5_minimal = (Beta1 - Beta2)*Eigen::Matrix3d::Identity();

            J5 = J5_minimal ;

            //std::cout<<"J1: "<<std::endl<<J1<<std::endl;

            if(jacobiansMinimal != NULL && jacobiansMinimal[5] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J5_minimal_map(jacobiansMinimal[5]);
                J5_minimal_map = J5_minimal;

            }

        }
        if(jacobians[6] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J6(jacobians[6]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J6_minimal;

            J6_minimal = (Beta2 - Beta3)*Eigen::Matrix3d::Identity();

            J6 = J6_minimal;


            if(jacobiansMinimal != NULL &&  jacobiansMinimal[6] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J6_minimal_map(jacobiansMinimal[6]);
                J6_minimal_map = J6_minimal;

            }

        }
        if(jacobians[7] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J7(jacobians[7]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J7_minimal;
            //
            J7_minimal = Beta3*Eigen::Matrix3d::Identity();
            J7 = J7_minimal;

            //std::cout<<"J3: "<<std::endl<<J3<<std::endl;

            if(jacobiansMinimal != NULL &&  jacobiansMinimal[7] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J7_minimal_map(jacobiansMinimal[7]);
                J7_minimal_map = J7_minimal;

            }

        }

    }

    return success;
}