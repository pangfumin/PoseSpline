#include "PoseSpline/VectorSplineSampleVelocityError.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"
#include "PoseSpline/VectorSpaceSpline.hpp"


VectorSplineSampleVelocityError::VectorSplineSampleVelocityError(const double& t_meas,
                                                                 const double& time_interval,
                                                 const Eigen::Vector3d& V_meas):
        t_meas_(t_meas),time_interval_(time_interval), V_Meas_(V_meas){

};

VectorSplineSampleVelocityError::~VectorSplineSampleVelocityError(){

}
bool VectorSplineSampleVelocityError::Evaluate(double const* const* parameters,
                                           double* residuals,
                                           double** jacobians) const{

    return EvaluateWithMinimalJacobians(parameters,residuals,jacobians,NULL);

}

bool VectorSplineSampleVelocityError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                                               double* residuals,
                                                               double** jacobians,
                                                               double** jacobiansMinimal) const {

    Eigen::Map<const Eigen::Vector3d> V0(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> V1(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> V2(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> V3(parameters[3]);

    double  dotBeta1 = QSUtility::dot_beta1(time_interval_, t_meas_);
    double  dotBeta2 = QSUtility::dot_beta2(time_interval_, t_meas_);
    double  dotBeta3 = QSUtility::dot_beta3(time_interval_,t_meas_);

    // define residual
    // For simplity, we define error  =  /hat - meas.
    Eigen::Vector3d V_hat =  dotBeta1*(V1 - V0) +  dotBeta2*(V2 - V1) + dotBeta3*(V3 - V2);

    Eigen::Map<Eigen::Vector3d> error(residuals);

    squareRootInformation_ = Eigen::Matrix3d::Identity();
    error = squareRootInformation_*(V_hat - V_Meas_);

    if(jacobians != NULL){


        if(jacobians[0] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0(jacobians[0]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J0_minimal;

            J0_minimal = -dotBeta1*Eigen::Matrix3d::Identity();
            J0 = J0_minimal;
            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0_minimal_map(jacobiansMinimal[0]);
                J0_minimal_map = J0_minimal;
                //std::cout<<"J0_minimal_map: "<<std::endl<<J0_minimal_map<<std::endl;
            }
        }
        if(jacobians[1] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1(jacobians[1]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J1_minimal;

            J1_minimal = (dotBeta1 - dotBeta2)*Eigen::Matrix3d::Identity();

            J1 = J1_minimal ;

            //std::cout<<"J1: "<<std::endl<<J1<<std::endl;

            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1_minimal_map(jacobiansMinimal[1]);
                J1_minimal_map = J1_minimal;

            }

        }
        if(jacobians[2] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J2(jacobians[2]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J2_minimal;

            J2_minimal = (dotBeta2 - dotBeta3)*Eigen::Matrix3d::Identity();

            J2 = J2_minimal;


            if(jacobiansMinimal != NULL &&  jacobiansMinimal[2] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J2_minimal_map(jacobiansMinimal[2]);
                J2_minimal_map = J2_minimal;

            }

        }
        if(jacobians[3] != NULL){

            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J3(jacobians[3]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> J3_minimal;
            //
            J3_minimal = dotBeta3*Eigen::Matrix3d::Identity();
            J3 = J3_minimal;

            //std::cout<<"J3: "<<std::endl<<J3<<std::endl;


            if(jacobiansMinimal != NULL &&  jacobiansMinimal[3] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J3_minimal_map(jacobiansMinimal[3]);
                J3_minimal_map = J3_minimal;

            }

        }
    }

    return true;
}


