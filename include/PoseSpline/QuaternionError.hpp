#ifndef QUTERNIONERROR_H
#define QUTERNIONERROR_H

#include <ceres/ceres.h>
#include <iostream>
#include "PoseSpline/Quaternion.hpp"
#include "PoseSpline/QuaternionLocalParameter.hpp"
#include "PoseSpline/ErrorInterface.hpp"
#include "PoseSpline/NumbDifferentiator.hpp"

class QuaternionErrorCostFunction: public  ceres::SizedCostFunction<3,4> ,ErrorInterface{

public:

    QuaternionErrorCostFunction(){};

    QuaternionErrorCostFunction(Quaternion Q_target):Q_target_(Q_target){};
    ~QuaternionErrorCostFunction(){};

    /// \brief The base class type.
    typedef ceres::SizedCostFunction<3, 4> base_t;

    /// \brief The number of residuals .
    static const int kNumResiduals = 3;


    /// @brief Get dimension of residuals.
    /// @return The residual dimension.
    virtual size_t residualDim() const{
        return kNumResiduals;

    };

    /// @brief Get the number of parameter blocks this is connected to.
    /// @return The number of parameter blocks.
    virtual size_t parameterBlocks() const{
        return base_t::parameter_block_sizes().size();
    };

    /**
     * @brief get the dimension of a parameter block this is connected to.
     * @param parameterBlockId The ID of the parameter block of interest.
     * @return Its dimension.
     */
    virtual size_t parameterBlockDim(size_t parameterBlockId) const {
        return base_t::parameter_block_sizes().at(parameterBlockId);
    };
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const{

        return EvaluateWithMinimalJacobians(parameters,residuals,jacobians,NULL);

    };

    virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                              double* residuals,
                                              double** jacobians,
                                              double** jacobiansMinimal) const{

        Eigen::Map<const Quaternion> Q(parameters[0]);

        Eigen::Map<Eigen::Vector3d> error(residuals);


        // calculate residual

        Quaternion dq = quatMult(Q_target_,quatInv<double>(Q));
        error = 2.0*dq.head<3>();

        // calculate  Jacobians
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> Jacobian(jacobians[0]);
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J_minimal;
                J_minimal.setZero();
                J_minimal = -quatLeftComp(dq).topLeftCorner(3, 3);

                // pseudo inverse of the local parametrization Jacobian:
                Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_lift;
                QuaternionLocalParameter::liftJacobian(parameters[0], J_lift.data());

                Jacobian = J_minimal * J_lift;


                if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){
                    Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> Jacobian_minimal(jacobiansMinimal[0]);
                    Jacobian_minimal = J_minimal;

                }
                //std::cout<<"Residual: "<<error.transpose()<<std::endl;
                //std::cout<<"Jacobian: "<<std::endl<<Jacobian<<std::endl<<std::endl;

            }
        }

        return true;
    };



    bool VerifyJacobianNumDiff(double const* const* parameters){


        //
        Eigen::Vector3d residual;
        Eigen::Matrix<double,3,4,Eigen::RowMajor> jacobian;
        Eigen::Matrix<double,3,3,Eigen::RowMajor> jacobian_minimal;
        double* jacobians[1] = {jacobian.data()};
        double* jacobians_minimal[1] = {jacobian_minimal.data()};
        EvaluateWithMinimalJacobians(parameters,residual.data(),jacobians,jacobians_minimal);


        double dx=1e-6;
        QuaternionLocalParameter quaternionLocalParam;
        Eigen::Matrix<double,3,3> J0_numDiff;
        for(size_t i=0; i<3; ++i) {
            Eigen::Matrix<double, 3, 1> dp_0;
            Eigen::Matrix<double, 3, 1> residuals_p;
            Eigen::Matrix<double, 3, 1> residuals_m;
            dp_0.setZero();
            dp_0[i] = dx;

            Eigen::Matrix<double, 4, 1> parameter_plus;

            quaternionLocalParam.Plus(parameters[0],dp_0.data(),parameter_plus.data());
            const double* parameters_plus[1] = {parameter_plus.data()};
            Evaluate(parameters_plus,residuals_p.data(),NULL);

            dp_0.setZero();
            dp_0[i] = -dx;
            Eigen::Matrix<double, 4, 1> parameter_minus;

            quaternionLocalParam.Plus(parameters[0],dp_0.data(),parameter_minus.data());
            const double* parameters_minus[1] = {parameter_minus.data()};
            Evaluate(parameters_minus,residuals_m.data(),NULL);
            J0_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));

        }

        std::cout<<"Analytic Jacobian minimal: "<<std::endl<<jacobian_minimal<<std::endl;

        std::cout<<"Numdiff Jacobian minimal: "<<std::endl<<J0_numDiff<<std::endl;


        CHECK_EQ((jacobian_minimal - J0_numDiff).norm() < 0.00001, true );

        return (jacobian_minimal - J0_numDiff).norm() < 0.00001;

    };

private:
    Quaternion Q_target_;

};

#endif
