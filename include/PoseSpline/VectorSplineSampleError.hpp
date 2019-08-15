#ifndef VECTORSPLINESAMPLEERROR_H
#define  VECTORSPLINESAMPLEERROR_H

#include "PoseSpline/QuaternionSpline.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"

template <int Dim>
class VectorSplineSampleError: public ceres::SizedCostFunction<Dim,Dim,Dim,Dim,Dim>{
public:

    VectorSplineSampleError(const double& t_meas, const Eigen::Matrix<double, Dim, 1>& V_meas):
            t_meas_(t_meas),V_Meas_(V_meas){
    }
    virtual ~VectorSplineSampleError() {

    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        return EvaluateWithMinimalJacobians(parameters,residuals,jacobians,NULL);
    }
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const {
        Eigen::Map<const Eigen::Matrix<double, Dim, 1>> V0(parameters[0]);
        Eigen::Map<const Eigen::Matrix<double, Dim, 1>> V1(parameters[1]);
        Eigen::Map<const Eigen::Matrix<double, Dim, 1>> V2(parameters[2]);
        Eigen::Map<const Eigen::Matrix<double, Dim, 1>> V3(parameters[3]);

        double  Beta1 = QSUtility::beta1(t_meas_);
        double  Beta2 = QSUtility::beta2(t_meas_);
        double  Beta3 = QSUtility::beta3(t_meas_);

        // define residual
        // For simplity, we define error  =  /hat - meas.
        Eigen::Matrix<double, Dim, 1> V_hat = V0 + Beta1*(V1 - V0) +  Beta2*(V2 - V1) + Beta3*(V3 - V2);

        Eigen::Map<Eigen::Matrix<double, Dim, 1>> error(residuals);

        error = (V_hat - V_Meas_);

        if(jacobians != NULL){

            Eigen::Matrix<double, Dim, Dim > I = Eigen::Matrix<double, Dim, Dim >::Identity();
            if(jacobians[0] != NULL){
                Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J0(jacobians[0]);
                Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor> J0_minimal;

                J0_minimal  = (1 - Beta1)*I;
                J0 = J0_minimal ;
                if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){

                    Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J0_minimal_map(jacobiansMinimal[0]);
                    J0_minimal_map = J0_minimal;

                    //std::cout<<"J0_minimal_map: "<<std::endl<<J0_minimal_map<<std::endl;

                }

            }
            if(jacobians[1] != NULL){
                Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J1(jacobians[1]);
                Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor> J1_minimal;

                J1_minimal = (Beta1 - Beta2)*I;

                J1 = J1_minimal ;

                //std::cout<<"J1: "<<std::endl<<J1<<std::endl;

                if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){

                    Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J1_minimal_map(jacobiansMinimal[1]);
                    J1_minimal_map = J1_minimal;

                }

            }
            if(jacobians[2] != NULL){
                Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J2(jacobians[2]);
                Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor> J2_minimal;

                J2_minimal = (Beta2 - Beta3)*I;

                J2 = J2_minimal;


                if(jacobiansMinimal != NULL &&  jacobiansMinimal[2] != NULL){

                    Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J2_minimal_map(jacobiansMinimal[2]);
                    J2_minimal_map = J2_minimal;

                }

            }
            if(jacobians[3] != NULL){

                Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J3(jacobians[3]);
                Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor> J3_minimal;
                //
                J3_minimal = Beta3*I;
                J3 = J3_minimal;

                //std::cout<<"J3: "<<std::endl<<J3<<std::endl;

                if(jacobiansMinimal != NULL &&  jacobiansMinimal[3] != NULL){

                    Eigen::Map<Eigen::Matrix<double,Dim,Dim,Eigen::RowMajor>> J3_minimal_map(jacobiansMinimal[3]);
                    J3_minimal_map = J3_minimal;

                }

            }
        }

        return true;
    }


private:

    double t_meas_;
    Eigen::Matrix<double, Dim, 1> V_Meas_;
};

#endif