#include "pose-spline/VectorSpaceSpline.hpp"
#include "pose-spline/QuaternionSplineUtility.hpp"
#include "pose-spline/QuaternionSplineSampleError.hpp"
#include "utility/Time.hpp"
#include <algorithm>

namespace ze {
    VectorSpaceSpline::VectorSpaceSpline(int spline_order)
            : BSpline(spline_order),mTimeInterval(0) {

    }

    VectorSpaceSpline::VectorSpaceSpline(int spline_order,double interval)
            : BSpline(spline_order),mTimeInterval(interval){

    }
    VectorSpaceSpline::~VectorSpaceSpline(){
        for (auto i : mControlPointsParameter)
            delete [] i;
        mControlPointsParameter.clear();
    }
    void VectorSpaceSpline::setTimeInterval(double timeInterval){
        mTimeInterval = timeInterval;

    }
    double VectorSpaceSpline::getTimeInterval(){
        return mTimeInterval ;

    }
    bool VectorSpaceSpline::isTsEvaluable(double ts){
        return ts >= t_min() && ts < t_max();

    }
    void VectorSpaceSpline:: addSample(double t, Eigen::Vector3d Q){

        if(getControlPointNum() == 0){
            initialSplineKnot(t);
            //std::cout<<"t: "<<Time(t)<<" t_max: "<<Time(t_max())<<std::endl;
        }else if(getControlPointNum() >= numCoefficientsRequired(1) ){
            //std::cout<<"add "<<Time(t - t_max())<<std::endl;

            if(t < t_min()){
                std::cerr<<"[Error] Inserted t is smaller than t_min()！"<<std::endl;
                LOG(FATAL) << "Inserted "<<Time(t)<<" is smaller than t_min() "<<Time(t_min())<<std::endl;
            }else if(t >= t_max()){
                // add new knot and control Points
                while(t >= t_max()){

                    //std::cout<<"t: "<<Time(t)<<" t_max: "<<Time(t_max())<<std::endl;
                    knots_.push_back(knots_.back() + mTimeInterval); // append one;
                    initialNewControlPoint();
                }

            }

        }
        // Tricky: do not add point close to t_max
        if( t_max() - t > 0.0001){
            mSampleValues.insert(std::pair<double ,Eigen::Vector3d>(t,Q));
        }
        CHECK_EQ(knots_.size() - spline_order_, getControlPointNum());

    }

    void VectorSpaceSpline::initialSpline(std::vector<std::pair<double,Eigen::Vector3d>> Meas){
        /*
        // Build a  least-square problem
        ceres::Problem problem;
        QuaternionLocalParameter* quaternionLocalParam = new QuaternionLocalParameter;
        //std::cout<<"Meas NUM: "<<Meas.size()<<std::endl;
        for(auto i : Meas){
            //std::cout<<"-----------------------------------"<<std::endl;
            // add sample
            addSample(i.first,i.second);

            // Returns the normalized u value and the lower-bound time index.
            std::pair<double,unsigned  int> ui = computeUAndTIndex(i.first);
            //VectorX u = computeU(ui.first, ui.second, 0);
            double u = ui.first;
            int bidx = ui.second - spline_order_ + 1;

            double* cp0 = getControlPoint(bidx);
            double* cp1 = getControlPoint(bidx+1);
            double* cp2 = getControlPoint(bidx+2);
            double* cp3 = getControlPoint(bidx+3);

            QuaternionMap CpMap0(cp0);
            QuaternionMap CpMap1(cp1);
            QuaternionMap CpMap2(cp2);
            QuaternionMap CpMap3(cp3);
            QuaternionSplineSampleError* quatSampleFunctor = new QuaternionSplineSampleError(u,i.second);

            problem.AddParameterBlock(cp0,4,quaternionLocalParam);
            problem.AddParameterBlock(cp1,4,quaternionLocalParam);
            problem.AddParameterBlock(cp2,4,quaternionLocalParam);
            problem.AddParameterBlock(cp3,4,quaternionLocalParam);

            problem.AddResidualBlock(quatSampleFunctor, NULL, cp0, cp1, cp2, cp3);

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
         */

    }


    void VectorSpaceSpline::initialSplineKnot(double t){
        // Initialize the spline so that it interpolates the two points
        // and moves between them with a constant velocity.

        // How many knots are required for one time segment?
        int K = numKnotsRequired(1);
        // How many coefficients are required for one time segment?
        int C = numCoefficientsRequired(1);

        // Initialize a uniform knot sequence
        real_t dt = mTimeInterval;
        std::vector<real_t> knots(K);
        for(int i = 0; i < K; i++)
        {
            knots[i] = t + (i - spline_order_ + 1) * dt;
        }

        knots_ = knots;

        for(int i = 0; i < C; i++){
            initialNewControlPoint();
        }


    }
    void VectorSpaceSpline::printKnots(){
        std::cout<<"knot: "<<std::endl;
        for(auto i: knots_){
            std::cout<<Time(i)<<std::endl;

        }
    }
    Eigen::Vector3d VectorSpaceSpline::evaluateSpline(const real_t t){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;

        return evaluateSpline(u,
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+1)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+2)),
                              Eigen::Map<Eigen::Matrix<double,3,1>>(getControlPoint(bidx+3)));
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

    void VectorSpaceSpline::initialNewControlPoint(){
        Eigen::Vector3d vec = (Eigen::Vector3d::Zero());
        double* data = new double[3];
        memcpy(data, vec.data(),sizeof(double)*3);
        mControlPointsParameter.push_back(data);
    }

}
