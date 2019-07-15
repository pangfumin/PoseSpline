#include "pose-spline/VectorSpaceSpline.hpp"
#include "pose-spline/QuaternionSplineUtility.hpp"
#include "pose-spline/QuaternionSplineSampleError.hpp"
#include "pose-spline/VectorSplineSampleError.hpp"
#include "pose-spline/VectorSplineSampleAutoError.hpp"
//#include "okvis_util/Time.hpp"
#include <algorithm>


    VectorSpaceSpline::VectorSpaceSpline()
            : BSplineBase(1.0) {

    }

    VectorSpaceSpline::VectorSpaceSpline(double interval)
            : BSplineBase(interval){

    }

    void VectorSpaceSpline::initialSpline(std::vector<std::pair<double,Eigen::Vector3d>> Meas){

        // Build a  least-square problem
        ceres::Problem problem;

        for(auto i : Meas){
            //std::cout<<"-----------------------------------"<<std::endl;
            // add sample
            addElemenTypeSample(i.first,i.second);

            // Returns the normalized u value and the lower-bound time index.
            std::pair<double,unsigned  int> ui = computeUAndTIndex(i.first);
            //VectorX u = computeU(ui.first, ui.second, 0);
            double u = ui.first;
            int bidx = ui.second - spline_order() + 1;

            double* cp0 = getControlPoint(bidx);
            double* cp1 = getControlPoint(bidx+1);
            double* cp2 = getControlPoint(bidx+2);
            double* cp3 = getControlPoint(bidx+3);

            Eigen::Map<Eigen::Matrix<double,3,1>> CpMap0(cp0);
            Eigen::Map<Eigen::Matrix<double,3,1>> CpMap1(cp1);
            Eigen::Map<Eigen::Matrix<double,3,1>> CpMap2(cp2);
            Eigen::Map<Eigen::Matrix<double,3,1>> CpMap3(cp3);

            VectorSplineSampleError* vectorSplineSampleError
                    = new VectorSplineSampleError(u,i.second);
//
//            ceres::CostFunction* vectorSplineSampleError
//                    = new ceres::AutoDiffCostFunction<VectorSplineSampleAutoError,3,3,3,3,3>(new VectorSplineSampleAutoError(u, i.second));

            problem.AddParameterBlock(cp0,3);
            problem.AddParameterBlock(cp1,3);
            problem.AddParameterBlock(cp2,3);
            problem.AddParameterBlock(cp3,3);

            problem.AddResidualBlock(vectorSplineSampleError, NULL, cp0, cp1, cp2, cp3);

        }
        //std::cout<<"ParameterNum: "<<problem.NumParameterBlocks()<<std::endl;
        //std::cout<<"ResidualNUM: "<<problem.NumResiduals()<<std::endl;


        // Set up the only cost function (also known as residual).
        //ceres::CostFunction* cost_function = new QuadraticCostFunction;
        //problem.AddResidualBlock(cost_function, NULL, &x);
        // Run the solver!
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_solver_time_in_seconds = 30;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.parameter_tolerance = 1e-4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;


    }


    Eigen::Vector3d VectorSpaceSpline::evaluateSpline(const real_t t){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order() + 1;

        return evaluateSpline(u,
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+1)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+2)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+3)));
    }

    Eigen::Vector3d VectorSpaceSpline::evaluateDotSpline(const real_t t){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order() + 1;

        return evaluateDotSpline(u,getTimeInterval(),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+1)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+2)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+3)));
    }

    Eigen::Vector3d VectorSpaceSpline::evaluateDotSplineNumeric(const real_t t){

        double eps = 1e-5;
        return (evaluateSpline(t + eps) - evaluateSpline(t - eps))/(2*eps);
    }



    Eigen::Vector3d VectorSpaceSpline::evaluateSpline(const real_t t,
                                   const Eigen::Vector3d& v0,
                                   const Eigen::Vector3d& v1,
                                   const Eigen::Vector3d& v2,
                                   const Eigen::Vector3d& v3){

        double  Beta1 = QSUtility::beta1(t);
        double  Beta2 = QSUtility::beta2(t);
        double  Beta3 = QSUtility::beta3(t);
        Eigen::Vector3d V = v0 + Beta1*(v1 - v0) +  Beta2*(v2 - v1) + Beta3*(v3 - v2);
        return V;

    }

    Eigen::Vector3d VectorSpaceSpline::evaluateDotSpline(const real_t t,
                                                         const double timeInterval,
                                             const Eigen::Vector3d& v0,
                                             const Eigen::Vector3d& v1,
                                             const Eigen::Vector3d& v2,
                                             const Eigen::Vector3d& v3) {

        double  dotBeta1 = QSUtility::dot_beta1(timeInterval, t);
        double  dotBeta2 = QSUtility::dot_beta2(timeInterval, t);
        double  dotBeta3 = QSUtility::dot_beta3(timeInterval, t);
        Eigen::Vector3d dotV =  dotBeta1*(v1 - v0) +  dotBeta2*(v2 - v1) + dotBeta3*(v3 - v2);
        return dotV;
    }
