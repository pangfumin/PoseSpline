#include "pose-spline/PoseSpline.hpp"
#include "okvis_util/Time.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
#include "pose-spline/PoseSplineSampleError.hpp"
#include "pose-spline/PoseSplineUtility.hpp"
    PoseSpline::PoseSpline()
            : BSplineBase(1.0) {

    }

    PoseSpline::PoseSpline( double interval)
            : BSplineBase(interval) {

    }
    void PoseSpline::initialPoseSpline(std::vector<std::pair<double, Pose<double>>> Meas) {

        // Build a  least-square problem
        ceres::Problem problem;
        PoseLocalParameter *poseLocalParameter = new PoseLocalParameter;
        //std::cout<<"Meas NUM: "<<Meas.size()<<std::endl;
        for (auto i : Meas) {
            //std::cout<<"-----------------------------------"<<std::endl;
            // add sample
            addElemenTypeSample(i.first, i.second);

            // Returns the normalized u value and the lower-bound time index.
            std::pair<double, unsigned int> ui = computeUAndTIndex(i.first);
            //VectorX u = computeU(ui.first, ui.second, 0);
            double u = ui.first;
            int bidx = ui.second - spline_order() + 1;

            double *cp0 = getControlPoint(bidx);
            double *cp1 = getControlPoint(bidx + 1);
            double *cp2 = getControlPoint(bidx + 2);
            double *cp3 = getControlPoint(bidx + 3);


            PoseSplineSampleError* poseSampleFunctor = new PoseSplineSampleError(u,i.second);
/*
            std::cout<<"Q0: "<<CpMap0.transpose()<<std::endl;
            std::cout<<"Q1: "<<CpMap1.transpose()<<std::endl;
            std::cout<<"Q2: "<<CpMap2.transpose()<<std::endl;
            std::cout<<"Q3: "<<CpMap3.transpose()<<std::endl;
*/
            problem.AddParameterBlock(cp0,7,poseLocalParameter);
            problem.AddParameterBlock(cp1,7,poseLocalParameter);
            problem.AddParameterBlock(cp2,7,poseLocalParameter);
            problem.AddParameterBlock(cp3,7,poseLocalParameter);

            problem.AddResidualBlock(poseSampleFunctor, NULL, cp0, cp1, cp2, cp3);

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





    Pose<double> PoseSpline::evalPoseSpline(real_t t ){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order() + 1;
//
        Eigen::Map<Eigen::Matrix<double, 3,1>> t0(getControlPoint(bidx));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t1(getControlPoint(bidx+1));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t2(getControlPoint(bidx+2));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t3(getControlPoint(bidx+3));

        Eigen::Map<Eigen::Matrix<double, 4,1>> q0(getControlPoint(bidx) + 3);
        Eigen::Map<Eigen::Matrix<double, 4,1>> q1(getControlPoint(bidx+1) + 3);
        Eigen::Map<Eigen::Matrix<double, 4,1>> q2(getControlPoint(bidx+2) + 3);
        Eigen::Map<Eigen::Matrix<double, 4,1>> q3(getControlPoint(bidx+3) + 3);

        return PSUtility::EvaluatePS(u,
                                     Pose<double>(t0, q0), Pose<double>(t1, q1),
                                     Pose<double>(t2, q2), Pose<double>(t2,q3));
    }

    Eigen::Vector3d PoseSpline::evalLinearVelocity(real_t t ){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order() + 1;
        Eigen::Map<Eigen::Matrix<double, 3,1>> t0(getControlPoint(bidx));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t1(getControlPoint(bidx+1));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t2(getControlPoint(bidx+2));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t3(getControlPoint(bidx+3));

      
        return PSUtility::EvaluateLinearVelocity(u, getTimeInterval(),
                                                 t0, t1, t2, t3);
    }

    Eigen::Vector3d PoseSpline::evalLinearAccelerator(real_t t) {

        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order() + 1;
        Eigen::Map<Eigen::Matrix<double, 7,1>> t0(getControlPoint(bidx));
        Eigen::Map<Eigen::Matrix<double, 7,1>> t1(getControlPoint(bidx+1));
        Eigen::Map<Eigen::Matrix<double, 7,1>> t2(getControlPoint(bidx+2));
        Eigen::Map<Eigen::Matrix<double, 7,1>> t3(getControlPoint(bidx+3));
        Pose<double> Pose0, Pose1,Pose2, Pose3;
        Pose0.setParameters(t0);
        Pose1.setParameters(t1);
        Pose2.setParameters(t2);
        Pose3.setParameters(t3);


        return PSUtility::EvaluateLinearAccelerate(u, getTimeInterval(),
                                                  Pose0, Pose1, Pose2,Pose3);

    }


