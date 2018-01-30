#include "pose-spline/PoseSpline.hpp"
#include "utility/Time.hpp"
#include "pose-spline/PoseLocalParameter.hpp"
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

    void PoseSpline::addSample(double t, Pose Q) {

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
            mSampleValues.insert(std::pair<double, Pose>(t, Q));
        }
        CHECK_EQ(knots_.size() - spline_order_, getControlPointNum());

    }

    void PoseSpline::initialPoseSpline(std::vector<std::pair<double, Pose>> Meas) {

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

    void PoseSpline::printKnots() {
        std::cout << "knot: " << std::endl;
        for (auto i: knots_) {
            std::cout << Time(i) << std::endl;

        }
    }

}