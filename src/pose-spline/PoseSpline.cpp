#include "pose-spline/PoseSpline.hpp"
#include "utility/Time.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
#include "pose-spline/PoseSplineSampleError.hpp"
#include "pose-spline/PoseSplineUtility.hpp"
namespace ze {
    PoseSpline::PoseSpline(int spline_order)
            : BSpline(spline_order), mTimeInterval(0) {

    }

    PoseSpline::PoseSpline(int spline_order, double interval)
            : BSpline(spline_order), mTimeInterval(interval) {

    }

    PoseSpline::~PoseSpline() {

        for (auto i : mControlPointsParameter)
            delete[] i;

        mControlPointsParameter.clear();
    }

    void PoseSpline::setTimeInterval(double timeInterval) {
        mTimeInterval = timeInterval;

    }

    double PoseSpline::getTimeInterval() {
        return mTimeInterval;

    }

    bool PoseSpline::isTsEvaluable(double ts) {
        return ts >= t_min() && ts < t_max();

    }

    void PoseSpline::addSample(double t, Pose<double> Q) {

        if (getControlPointNum() == 0) {
            initialPoseSplineKnot(t);
            //std::cout<<"t: "<<Time(t)<<" t_max: "<<Time(t_max())<<std::endl;
        } else if (getControlPointNum() >= numCoefficientsRequired(1)) {
            //std::cout<<"add "<<Time(t - t_max())<<std::endl;

            if (t < t_min()) {
                std::cerr << "[Error] Inserted t is smaller than t_min()！" << std::endl;
                LOG(FATAL) << "Inserted " << Time(t) << " is smaller than t_min() " << Time(t_min()) << std::endl;
            } else if (t >= t_max()) {
                // add new knot and control Points
                while (t >= t_max()) {

                    //std::cout<<"t: "<<Time(t)<<" t_max: "<<Time(t_max())<<std::endl;
                    knots_.push_back(knots_.back() + mTimeInterval); // append one;
                    initialNewControlPoint();
                }

            }

        }
        // Tricky: do not add point close to t_max
        if (t_max() - t > 0.0001) {
            mSampleValues.insert(std::pair<double, Pose<double>>(t, Q));
        }
        CHECK_EQ(knots_.size() - spline_order_, getControlPointNum());

    }

    void PoseSpline::initialPoseSpline(std::vector<std::pair<double, Pose<double>>> Meas) {

        // Build a  least-square problem
        ceres::Problem problem;
        PoseLocalParameter *poseLocalParameter = new PoseLocalParameter;
        //std::cout<<"Meas NUM: "<<Meas.size()<<std::endl;
        for (auto i : Meas) {
            //std::cout<<"-----------------------------------"<<std::endl;
            // add sample
            addSample(i.first, i.second);

            // Returns the normalized u value and the lower-bound time index.
            std::pair<double, unsigned int> ui = computeUAndTIndex(i.first);
            //VectorX u = computeU(ui.first, ui.second, 0);
            double u = ui.first;
            int bidx = ui.second - spline_order_ + 1;

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


    void PoseSpline::initialPoseSplineKnot(double t) {
        // Initialize the spline so that it interpolates the two points
        // and moves between them with a constant velocity.

        // How many knots are required for one time segment?
        int K = numKnotsRequired(1);
        // How many coefficients are required for one time segment?
        int C = numCoefficientsRequired(1);

        // Initialize a uniform knot sequence
        real_t dt = mTimeInterval;
        std::vector<real_t> knots(K);
        for (int i = 0; i < K; i++) {
            knots[i] = t + (i - spline_order_ + 1) * dt;
        }

        knots_ = knots;

        for (int i = 0; i < C; i++) {
            initialNewControlPoint();
        }


    }

    void PoseSpline::initialNewControlPoint(){

        Pose<double> unit;
        double* data = new double[7];
        memcpy(data, unit.parameterPtr(),sizeof(double)*7);
        mControlPointsParameter.push_back(data);
    }


    Pose<double> PoseSpline::evalPoseSpline(real_t t ){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;
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
        unsigned int bidx = ui.second - spline_order_ + 1;
        Eigen::Map<Eigen::Matrix<double, 3,1>> t0(getControlPoint(bidx));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t1(getControlPoint(bidx+1));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t2(getControlPoint(bidx+2));
        Eigen::Map<Eigen::Matrix<double, 3,1>> t3(getControlPoint(bidx+3));

      
        return PSUtility::EvaluateLinearVelocity(u, mTimeInterval,
                                                 t0, t1, t2, t3);
    }
    


    void PoseSpline::printKnots() {
        std::cout << "knot: " << std::endl;
        for (auto i: knots_) {
            std::cout << Time(i) << std::endl;

        }
    }

}