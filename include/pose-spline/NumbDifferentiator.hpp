#ifndef NUMBDIFFERENTIATOR_H
#define NUMBDIFFERENTIATOR_H
#include <ceres/ceres.h>
#include <memory>



/*
 *  A naive numberic differential tool to do finite differenceing  to calculate
 *  num-jacobian.
 *
 *  This num-differ is able to lightly calculate local parameter jacobian.
 *
 *  Some alternative tools are availibale in Ceres-solver, e.g.
 *  a. num_diff
 *  b. gradient_checher
 *
 *  @autor: Pang Fumin
 */
#define Eps  (1e-6)

template<typename Functor,int ParamBlockSize>
class NumbDifferentiiator{

public:
    NumbDifferentiiator(Functor* ptrErrorFunctor):
            ptrErrorFunctor_(ptrErrorFunctor){

        if(ptrErrorFunctor_ == NULL){
            LOG(FATAL)<<"Error functor pointor is NULL!";
        }

    };

    template <int ResidualDim,
              int ParamDim,
              int MinimalParamDim,
              typename LoaclPrameter>
    bool df_r_xi(double** parameters,
                 unsigned int paramId,
                 double* jacobiansMinimal){

        std::shared_ptr<LoaclPrameter> ptrlocalParemeter(new LoaclPrameter);
        Eigen::Map<Eigen::Matrix<double,ResidualDim,MinimalParamDim,Eigen::RowMajor>> miniJacobian(jacobiansMinimal);
        Eigen::Map<Eigen::Matrix<double,ParamDim,1>> xi(parameters[paramId]);
        Eigen::Matrix<double,ResidualDim,1> residual_plus;
        Eigen::Matrix<double,ResidualDim,1> residual_minus;

        Eigen::Matrix<double,ParamDim,1>  xi_plus_delta, xi_minus_delta;
        Eigen::Matrix<double,MinimalParamDim,1> delta;

        for(unsigned int i = 0; i < MinimalParamDim; i++){

            // plus
            delta.setZero();
            delta(i) = Eps;
            ptrlocalParemeter->Plus(xi.data(),delta.data(),xi_plus_delta.data());
            double* parameter_plus[ParamBlockSize];
            applyDistribance(parameters,xi_plus_delta.data(),parameter_plus,paramId);
            ptrErrorFunctor_->Evaluate(parameter_plus,residual_plus.data(),NULL);

            // minus
            delta.setZero();
            delta(i) = -Eps;
            ptrlocalParemeter->Plus(xi.data(),delta.data(),xi_minus_delta.data());
            double* parameter_minus[ParamBlockSize];
            applyDistribance(parameters,xi_minus_delta.data(),parameter_minus,paramId);
            ptrErrorFunctor_->Evaluate(parameter_minus,residual_minus.data(),NULL);

            // diff
            miniJacobian.col(i) = (residual_plus - residual_minus)/(2.0*Eps);


        }

        return true;
    };

    template <int ResidualDim,
              int ParamDim>
    bool df_r_xi(double** parameters,
                 unsigned int paramId,
                 double* jacobian){

        Eigen::Map<Eigen::Matrix<double,ResidualDim,ParamDim,Eigen::RowMajor>> Jacobian(jacobian);
        Eigen::Map<Eigen::Matrix<double,ParamDim,1>> xi(parameters[paramId]);
        Eigen::Matrix<double,ResidualDim,1> residual_plus;
        Eigen::Matrix<double,ResidualDim,1> residual_minus;

        Eigen::Matrix<double,ParamDim,1>  xi_plus_delta, xi_minus_delta;
        Eigen::Matrix<double,ParamDim,1> delta;

        for(unsigned int i = 0; i < ParamDim; i++){

            delta.setZero();
            delta(i) = Eps;
            xi_plus_delta = xi + delta;
            double* parameter_plus[ParamBlockSize];
            applyDistribance(parameters,xi_plus_delta.data(),parameter_plus,paramId);
            ptrErrorFunctor_->Evaluate(parameter_plus,residual_plus.data(),NULL);

            delta.setZero();
            delta(i) = -Eps;
            xi_minus_delta = xi + delta;
            double* parameter_minus[ParamBlockSize];
            applyDistribance(parameters,xi_minus_delta.data(),parameter_minus,paramId);
            ptrErrorFunctor_->Evaluate(parameter_minus,residual_minus.data(),NULL);

            Jacobian.col(i) = (residual_plus - residual_minus)/(2.0*Eps);

        }

        return true;
    };

    template <int ResidualDim,
            int ParamDim>
    static bool isJacobianEqual(double* Jacobian_a,
                                double* Jacobian_b,
                                double relTol = 1e-4) {

        Eigen::Map<Eigen::Matrix<double,ResidualDim,ParamDim,Eigen::RowMajor>> jacobian_a(Jacobian_a);
        Eigen::Map<Eigen::Matrix<double,ResidualDim,ParamDim,Eigen::RowMajor>> jacobian_b(Jacobian_b);


        bool isCorrect = true;
        // check
        double norm = jacobian_a.norm();
        Eigen::MatrixXd J_diff = jacobian_a - jacobian_b;
        double maxDiff = std::max(-J_diff.minCoeff(), J_diff.maxCoeff());
        if (maxDiff / norm > relTol) {
            LOG(INFO) << "Jacobian inconsistent: ";
            LOG(INFO) << " Jacobian a: ";
            LOG(INFO) << jacobian_a;
            LOG(INFO) << "provided Jacobian b: ";
            LOG(INFO) << jacobian_b;
            LOG(INFO) << "relative error: " << maxDiff / norm
                      << ", relative tolerance: " << relTol;
            isCorrect = false;
        }

        return isCorrect;

    }

private:
    Functor* ptrErrorFunctor_;

    void applyDistribance(double** parameters,
                          double* parameter_i,
                          double** parameters_plus, unsigned int ith){

        for(unsigned int i = 0; i < ith; i++){
            parameters_plus[i] = parameters[i];
        }
        parameters_plus[ith] = parameter_i;
        for(unsigned int i = ith + 1; i < ParamBlockSize; i++){
            parameters_plus[i] = parameters[i];
        }

    }

};

//const int n = sizeof...(T)

#endif