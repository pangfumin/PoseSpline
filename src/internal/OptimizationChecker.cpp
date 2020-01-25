#include "internal/OptimizationChecker.h"

namespace ceres {
    OptimizationChecker::OptimizationChecker(ceres::CostFunction *function,
                                             const std::vector<ceres::LocalParameterization *> *local_parameterizations,
                                             ceres::LossFunction* loss_function)
            :
            function_(function), local_parameterizations_(local_parameterizations),
            loss_function_(loss_function){

    }

    bool OptimizationChecker::Probe(const std::vector<const double* > parameters,
               double relative_precision,
               const ceres::Solver::Options& options,
               ProbeResults* results) const {
        Eigen::VectorXd init_cost(function_->num_residuals());
        Eigen::VectorXd final_cost(function_->num_residuals());
        function_->Evaluate(parameters.data(), init_cost.data(), nullptr);
        int num_params = parameters.size();
        std::vector<Eigen::VectorXd> eigen_parameters(num_params);
        std::vector<double * > mutable_parameters(num_params);
        for (int i = 0; i < num_params; i++) {
            Eigen::Map<const Eigen::VectorXd> param_i(parameters.at(i), function_->parameter_block_sizes().at(i));
            eigen_parameters.at(i) = param_i;
            mutable_parameters.at(i) = const_cast<double *>(param_i.data());
        }

        ceres::Problem problem;

        int num_local_params = local_parameterizations_->size();
        for (int i = 0; i < parameters.size(); i++) {
            if (i < num_local_params) {
                problem.AddParameterBlock(eigen_parameters.at(i).data(),
                                          function_->parameter_block_sizes().at(i),
                                          local_parameterizations_->at(i));
            } else {
                problem.AddParameterBlock(eigen_parameters.at(i).data(),
                                          function_->parameter_block_sizes().at(i));
            }

        }


        // add residuals
        problem.AddResidualBlock(function_, loss_function_, mutable_parameters);

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout<<"--------------"<<std::endl;
        std::cout << summary.FullReport() << std::endl;


        function_->Evaluate(mutable_parameters.data(), final_cost.data(), nullptr);

        std::cout << "init cost: " << init_cost.transpose() << std::endl;
        std::cout << "final cost: " << final_cost.transpose() << std::endl;



        // go

        for (int i = 0; i < parameters.size(); i++) {
            std::cout << "_________________________ " << i << "________________________" << std::endl;
            optimizeAtOneParameter(eigen_parameters, i, options,0);
        }




        return true;
    }

    bool OptimizationChecker::optimizeAtOneParameter  (
            const std::vector<Eigen::VectorXd>& eigen_parameters,
            const int& ith,
            const ceres::Solver::Options& options,
            const double threshold) const{

        std::vector<Eigen::VectorXd> mutable_eigen_parameters = eigen_parameters;
        std::vector<double * > mutable_parameters(eigen_parameters.size());
        for (int i = 0; i < mutable_eigen_parameters.size(); i++) {
            mutable_parameters.at(i) = const_cast<double*>(mutable_eigen_parameters.at(i).data());
        }

        Eigen::VectorXd init_cost(function_->num_residuals());
        Eigen::VectorXd final_cost(function_->num_residuals());
        function_->Evaluate(mutable_parameters.data(), init_cost.data(), nullptr);

        // disturb ith parameter
        Eigen::VectorXd  disturb_global_parameter;
        int local_size = 0;
        int global_size = 0;
        if (ith < local_parameterizations_->size()) { // no local parameters
            local_size = local_parameterizations_->at(ith)->LocalSize();
            global_size = local_parameterizations_->at(ith)->GlobalSize();

        } else {
            local_size = function_->parameter_block_sizes().at(ith);
            global_size = local_size;

        }
        Eigen::VectorXd local_disturb(local_size);
        local_disturb.setRandom();
        local_disturb.setZero();

        local_disturb *= 0.1;
        disturb_global_parameter =  Eigen::VectorXd(global_size);
        local_parameterizations_->at(ith)->Plus(mutable_eigen_parameters.at(ith).data(),
                                                local_disturb.data(),
                                                disturb_global_parameter.data());

        mutable_parameters.at(ith) = disturb_global_parameter.data();

        ceres::Problem problem;

        int num_local_params = local_parameterizations_->size();
        for (int i = 0; i < mutable_parameters.size(); i++) {
            if (i < num_local_params) {
                problem.AddParameterBlock(mutable_parameters.at(i),
                                          function_->parameter_block_sizes().at(i),
                                          local_parameterizations_->at(i));
            } else {
                problem.AddParameterBlock(mutable_parameters.at(i),
                                          function_->parameter_block_sizes().at(i));
            }

            if (i != ith) {
                problem.SetParameterBlockConstant(mutable_parameters.at(i));
            }

        }


        // add residuals
        problem.AddResidualBlock(function_, loss_function_, mutable_parameters);

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout<<"--------------"<<std::endl;
        std::cout << summary.FullReport() << std::endl;

        function_->Evaluate(mutable_parameters.data(), final_cost.data(), nullptr);

        std::cout << "init cost: " << init_cost.transpose() << std::endl;
        std::cout << "final cost: " << final_cost.transpose() << std::endl;

        return true;

    }


}