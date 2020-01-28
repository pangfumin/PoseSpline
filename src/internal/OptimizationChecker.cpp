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
            mutable_parameters.at(i) = const_cast<double *>(eigen_parameters.at(i).data());
        }

        ceres::Problem problem;

        int num_local_params = local_parameterizations_->size();
//        std::cout << "num_local_params: "<< num_local_params<< std::endl;
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
        std::cout << summary.BriefReport() << std::endl;


        function_->Evaluate(mutable_parameters.data(), final_cost.data(), nullptr);

        std::cout << "init cost: " << init_cost.transpose() << std::endl;
        std::cout << "final cost: " << final_cost.transpose() << std::endl;



        // go
        // set all parameter constant
        for (int i = 0; i < parameters.size(); i++) {
            problem.SetParameterBlockConstant(eigen_parameters.at(i).data());
        }


        // perturb
        for (int i = 0; i < parameters.size(); i++)
        {
            bool use_local_param = false;
            if (i < num_local_params) {
                use_local_param = true;
            }
//            int i = 0;
            Eigen::VectorXd disturb_cost(function_->num_residuals());
            problem.SetParameterBlockVariable(eigen_parameters.at(i).data());
            std::cout << "before   : " << eigen_parameters.at(i).transpose() << std::endl;
            if (use_local_param) {
                perturbParameter(eigen_parameters.at(i).data(), local_parameterizations_->at(i));

            } else {
                perturbParameter(eigen_parameters.at(i).data(), function_->parameter_block_sizes().at(i));
            }
            std::cout << "after    : " << eigen_parameters.at(i).transpose() << std::endl;
            //function_->Evaluate(mutable_parameters.data(), disturb_cost.data(), nullptr);
            std::cout << "disturb cost: " << disturb_cost.transpose() << std::endl;

            ceres::Solve(options, &problem, &summary);

            std::cout<<"--------------"<<std::endl;
            std::cout << summary.BriefReport() << std::endl;
            std::cout << "final cost: " << final_cost.transpose() << std::endl;
            std::cout << "after opt: " << eigen_parameters.at(i).transpose() << std::endl;



            problem.SetParameterBlockConstant(eigen_parameters.at(i).data());


        }






        return true;
    }

    void OptimizationChecker::perturbParameter(double * param,
            ceres::LocalParameterization* localParameterization) const {
        int local_size = localParameterization->LocalSize();
        int global_size = localParameterization->GlobalSize();
        Eigen::Map<const Eigen::VectorXd> x(param, global_size);
        Eigen::Map<Eigen::VectorXd> x_plus(param, global_size);
        Eigen::VectorXd delta(local_size);
        delta.setRandom();
        delta *= 0.1;
        localParameterization->Plus(x.data(), delta.data(), x_plus.data());
    }

    void OptimizationChecker::perturbParameter(double * param,
                                               const int size) const {
        int local_size = size;
        int global_size = size;
        Eigen::Map<const Eigen::VectorXd> x(param, global_size);
        Eigen::Map<Eigen::VectorXd> x_plus(param, global_size);
        Eigen::VectorXd delta(local_size);
        delta.setRandom();
        delta *= 0.1;
        x_plus = x + delta;
    }


    bool OptimizationChecker::optimizeAtOneParameter  (
            const std::vector<Eigen::VectorXd> eigen_parameters,
            const int& ith,
            const ceres::Solver::Options& options,
            const double threshold) const{

        int num_local_params = local_parameterizations_->size();
        std::cout << "num_local_params: " << num_local_params << std::endl;

        std::vector<Eigen::VectorXd> mutable_eigen_parameters = eigen_parameters;
        std::vector<double * > mutable_parameters(eigen_parameters.size());
//        std::cout << "mutable_parameters: " << mutable_parameters.size() << std::endl;
        for (int i = 0; i < mutable_eigen_parameters.size(); i++) {
            mutable_parameters.at(i) = const_cast<double*>(mutable_eigen_parameters.at(i).data());
        }


        Eigen::VectorXd init_cost(function_->num_residuals());
        Eigen::VectorXd disturb_cost(function_->num_residuals());
        Eigen::VectorXd final_cost(function_->num_residuals());
        std::cout << "function_->num_residuals(): " << function_->num_residuals() << std::endl;
        std::cout << "mutable_parameters: " << mutable_parameters.size() << std::endl;
//        function_->Evaluate(mutable_parameters.data(), init_cost.data(), nullptr);

        // disturb ith parameter
        Eigen::VectorXd  disturb_global_parameter;
        int local_size = 0;
        int global_size = 0;
        if (ith < local_parameterizations_->size()) { // no local parameters
            local_size = local_parameterizations_->at(ith)->LocalSize();
            global_size = local_parameterizations_->at(ith)->GlobalSize();
            Eigen::VectorXd local_disturb(local_size);
            local_disturb.setRandom();
//            local_disturb.setZero();

            local_disturb *= 0.1;
            disturb_global_parameter =  Eigen::VectorXd(global_size);
            local_parameterizations_->at(ith)->Plus(mutable_eigen_parameters.at(ith).data(),
                                                    local_disturb.data(),
                                                    disturb_global_parameter.data());

        } else {
            local_size = function_->parameter_block_sizes().at(ith);
            global_size = local_size;

            Eigen::VectorXd local_disturb(local_size);
            local_disturb.setRandom();
//            local_disturb.setZero();

            local_disturb *= 0.1;

            disturb_global_parameter = mutable_eigen_parameters.at(ith) + local_disturb;
        }

//
        mutable_parameters.at(ith) = disturb_global_parameter.data();

        function_->Evaluate(mutable_parameters.data(), disturb_cost.data(), nullptr);
//
        ceres::Problem problem;

//        int num_local_params = local_parameterizations_->size();
//        std::cout << "num_local_params: " << num_local_params << std::endl;
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
//
//
//        // add residuals
//        problem.AddResidualBlock(function_, loss_function_, mutable_parameters);
//
//        ceres::Solver::Summary summary;
//        ceres::Solve(options, &problem, &summary);
//
//        std::cout<<"--------------"<<std::endl;
//        std::cout << summary.BriefReport() << std::endl;
//

        function_->Evaluate(mutable_parameters.data(), final_cost.data(), nullptr);

//
        std::cout << "init    cost: " << init_cost.transpose() << std::endl;
        std::cout << "disturb cost: " << disturb_cost.transpose() << std::endl;
        std::cout << "final   cost: " << final_cost.transpose() << std::endl;

        return true;

    }


}