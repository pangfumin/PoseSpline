#include "pose-spline/QuaternionSpline.hpp"
#include "pose-spline/QuaternionSplineUtility.hpp"
#include "pose-spline/QuaternionSplineSampleError.hpp"
#include "pose-spline/QuaternionSplineSampleAutoError.hpp"
#include "utility/Time.hpp"
#include <algorithm>

namespace ze {
    QuaternionSpline::QuaternionSpline(int spline_order)
            : BSpline(spline_order),mTimeInterval(0) {

    }

    QuaternionSpline::QuaternionSpline(int spline_order,double interval)
            : BSpline(spline_order),mTimeInterval(interval){

    }
    QuaternionSpline::~QuaternionSpline(){

        for (auto i : mControlPointsParameter)
            delete [] i;

        mControlPointsParameter.clear();
    }
    void QuaternionSpline::setTimeInterval(double timeInterval){
        mTimeInterval = timeInterval;

    }
    double QuaternionSpline::getTimeInterval(){
        return mTimeInterval ;

    }
    bool QuaternionSpline::isTsEvaluable(double ts){
        return ts >= t_min() && ts < t_max();

    }
    void QuaternionSpline:: addSample(double t, Quaternion Q){

        if(getControlPointNum() == 0){
            initialQuaternionSplineKnot(t);
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
            mSampleValues.insert(std::pair<double ,Quaternion>(t,Q));
        }
        CHECK_EQ(knots_.size() - spline_order_, getControlPointNum());

    }

    void QuaternionSpline::initialQuaternionSpline(std::vector<std::pair<double,Quaternion>> Meas){
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
/*
            std::cout<<"Knot: "<<knots_.size()<<std::endl;
            std::cout<<"ContrPoint: "<<getControlPointNum()<<std::endl;
            std::cout<<"Sample: "<<mSampleValues.size()<<std::endl;

            std::cout<<"t_min(): "<<(t_min())<<std::endl;
            std::cout<<"t_max(): "<<(t_max())<<std::endl;
            std::cout<<"t_max() - t_min(): "<<t_max() - t_min()<<std::endl;
            std::cout<<"t - t_min(): "<<i.first - t_min()<<std::endl;

            std::cout<<"bidx: "<<bidx<<std::endl;
*/
            double* cp0 = getControlPoint(bidx);
            double* cp1 = getControlPoint(bidx+1);
            double* cp2 = getControlPoint(bidx+2);
            double* cp3 = getControlPoint(bidx+3);

            QuaternionMap CpMap0(cp0);
            QuaternionMap CpMap1(cp1);
            QuaternionMap CpMap2(cp2);
            QuaternionMap CpMap3(cp3);
#if 1
            QuaternionSplineSampleError* quatSampleFunctor = new QuaternionSplineSampleError(u,i.second);
#else
            ceres::CostFunction* quatSampleFunctor
                    = new ceres::AutoDiffCostFunction<QuaternionSplineSampleAutoError,3,4,4,4,4>(
                            new QuaternionSplineSampleAutoError(u, i.second));
#endif

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

    }


    void QuaternionSpline::initialQuaternionSplineKnot(double t){
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
    void QuaternionSpline::printKnots(){
        std::cout<<"knot: "<<std::endl;
        for(auto i: knots_){
            std::cout<<Time(i)<<std::endl;

        }
    }

    Quaternion QuaternionSpline::evalQuatSpline(real_t t ){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;

        return QSUtility::EvaluateQS(u,
                                   quatMap<double>(getControlPoint(bidx)),
                                   quatMap<double>(getControlPoint(bidx+1)),
                                   quatMap<double>(getControlPoint(bidx+2)),
                                   quatMap<double>(getControlPoint(bidx+3)));
    }

    Quaternion QuaternionSpline::evalDotQuatSpline(real_t t){

        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;

        return QSUtility::Evaluate_dot_QS(mTimeInterval,u,quatMap<double>(getControlPoint(bidx)),
                                          quatMap<double>(getControlPoint(bidx+1)),
                                          quatMap<double>(getControlPoint(bidx+2)),
                                          quatMap<double>(getControlPoint(bidx+3)));
    }

    Quaternion QuaternionSpline::evalDotDotQuatSpline(real_t t){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;
        return QSUtility::Evaluate_dot_dot_QS(mTimeInterval,u,quatMap<double>(getControlPoint(bidx)),
                                              quatMap<double>(getControlPoint(bidx+1)),
                                              quatMap<double>(getControlPoint(bidx+2)),
                                              quatMap<double>(getControlPoint(bidx+3)));

    }

    void  QuaternionSpline::evalQuatSplineDerivate(real_t t,
                                                   double* Quat,
                                                   double* dot_Qaut,
                                                   double* dot_dot_Quat){
        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;

        Quaternion Q0 = quatMap<double>(getControlPoint(bidx));
        Quaternion Q1 = quatMap<double>(getControlPoint(bidx + 1));
        Quaternion Q2 = quatMap<double>(getControlPoint(bidx + 2));
        Quaternion Q3 = quatMap<double>(getControlPoint(bidx + 3));

        double b1 = QSUtility::beta1(u);
        double b2 = QSUtility::beta2(u);
        double b3 = QSUtility::beta3(u);

        Eigen::Vector3d Phi1 = QSUtility::Phi(Q0,Q1);
        Eigen::Vector3d Phi2 = QSUtility::Phi(Q1,Q2);
        Eigen::Vector3d Phi3 = QSUtility::Phi(Q2,Q3);

        Quaternion r1 = QSUtility::r(b1,Phi1);
        Quaternion r2 = QSUtility::r(b2,Phi2);
        Quaternion r3 = QSUtility::r(b3,Phi3);

        Eigen::Map<Quaternion> Q_LG(Quat);

        Q_LG =  quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;

        if(dot_Qaut != NULL){
            double dot_b1 = QSUtility::dot_beta1(mTimeInterval,u);
            double dot_b2 = QSUtility::dot_beta2(mTimeInterval,u);
            double dot_b3 = QSUtility::dot_beta3(mTimeInterval,u);


            Quaternion  part1, part2,part3;
            part1 = quatLeftComp(QSUtility::dr_dt(dot_b1,b1,Phi1))*quatLeftComp(r2)*r3;
            part2 = quatLeftComp(r1)*quatLeftComp(QSUtility::dr_dt(dot_b2,b2,Phi2))*r3;
            part3 = quatLeftComp(r1)*quatLeftComp(r2)*QSUtility::dr_dt(dot_b3,b3,Phi3);

            Eigen::Map<Quaternion> dot_Q_LG(dot_Qaut);
            dot_Q_LG = quatLeftComp(Q0)*(part1 + part2 + part3);

            if(dot_dot_Quat != NULL){

                Quaternion dr1 = QSUtility::dr_dt(dot_b1,b1,Phi1);
                Quaternion dr2 = QSUtility::dr_dt(dot_b2,b2,Phi2);
                Quaternion dr3 = QSUtility::dr_dt(dot_b3,b3,Phi3);

                double dot_dot_b1 = QSUtility::dot_dot_beta1(mTimeInterval,u);
                double dot_dot_b2 = QSUtility::dot_dot_beta2(mTimeInterval,u);
                double dot_dot_b3 = QSUtility::dot_dot_beta3(mTimeInterval,u);

                Quaternion  ddr1 = QSUtility::d2r_dt2(dot_dot_b1,dot_b1,b1,Phi1);
                Quaternion  ddr2 = QSUtility::d2r_dt2(dot_dot_b2,dot_b2,b2,Phi2);
                Quaternion  ddr3 = QSUtility::d2r_dt2(dot_dot_b3,dot_b3,b3,Phi3);

                Quaternion  part11, part12, part13, part21, part22, part23, part31, part32, part33;
                part11 = quatLeftComp(ddr1)*quatLeftComp(r2)*r3;
                part12 = quatLeftComp(dr1)*quatLeftComp(dr2)*r3;
                part13 = quatLeftComp(dr1)*quatLeftComp(r2)*dr3;

                part21 = quatLeftComp(dr1)*quatLeftComp(dr2)*r3;
                part22 = quatLeftComp(r1)*quatLeftComp(ddr2)*r3;
                part23 = quatLeftComp(r1)*quatLeftComp(dr2)*dr3;

                part31 = quatLeftComp(dr1)*quatLeftComp(r2)*dr3;
                part32 = quatLeftComp(r1)*quatLeftComp(dr2)*dr3;
                part33 = quatLeftComp(r1)*quatLeftComp(r2)*ddr3;

                Eigen::Map<Quaternion> dot_dot_q(dot_dot_Quat);


                dot_dot_q = quatLeftComp(Q0)*(part11 + part12 + part13
                                              + part21 + part22 + part23
                                              + part31 + part32 + part33);
            }

        }

    }



    Eigen::Vector3d QuaternionSpline::evalOmega(real_t t){

        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;

        Quaternion Q0 = quatMap<double>(getControlPoint(bidx));
        Quaternion Q1 = quatMap<double>(getControlPoint(bidx + 1));
        Quaternion Q2 = quatMap<double>(getControlPoint(bidx + 2));
        Quaternion Q3 = quatMap<double>(getControlPoint(bidx + 3));

        double b1 = QSUtility::beta1(u);
        double b2 = QSUtility::beta2(u);
        double b3 = QSUtility::beta3(u);

        Eigen::Vector3d Phi1 = QSUtility::Phi(Q0,Q1);
        Eigen::Vector3d Phi2 = QSUtility::Phi(Q1,Q2);
        Eigen::Vector3d Phi3 = QSUtility::Phi(Q2,Q3);

        Quaternion r1 = QSUtility::r(b1,Phi1);
        Quaternion r2 = QSUtility::r(b2,Phi2);
        Quaternion r3 = QSUtility::r(b3,Phi3);

        Quaternion Q_LG =  quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;

        double dot_b1 = QSUtility::dot_beta1(mTimeInterval,u);
        double dot_b2 = QSUtility::dot_beta2(mTimeInterval,u);
        double dot_b3 = QSUtility::dot_beta3(mTimeInterval,u);


        Quaternion dot_Q_LG, part1, part2,part3;
        part1 = quatLeftComp(QSUtility::dr_dt(dot_b1,b1,Phi1))*quatLeftComp(r2)*r3;
        part2 = quatLeftComp(r1)*quatLeftComp(QSUtility::dr_dt(dot_b2,b2,Phi2))*r3;
        part3 = quatLeftComp(r1)*quatLeftComp(r2)*QSUtility::dr_dt(dot_b3,b3,Phi3);

        dot_Q_LG = quatLeftComp(Q0)*(part1 + part2 + part3);

        //std::cout<<"Q    : "<<Q_LG.transpose()<<std::endl;
        //std::cout<<"dot_Q: "<<dot_Q_LG.transpose()<<std::endl;


        return QSUtility::w<double>(Q_LG,dot_Q_LG);
    }

    Eigen::Vector3d QuaternionSpline::evalAlpha(real_t t){

        std::pair<double,unsigned  int> ui = computeUAndTIndex(t);
        double u = ui.first;
        unsigned int bidx = ui.second - spline_order_ + 1;

        Quaternion Q0 = quatMap<double>(getControlPoint(bidx));
        Quaternion Q1 = quatMap<double>(getControlPoint(bidx + 1));
        Quaternion Q2 = quatMap<double>(getControlPoint(bidx + 2));
        Quaternion Q3 = quatMap<double>(getControlPoint(bidx + 3));

        Eigen::Vector3d Phi1 = QSUtility::Phi(Q0,Q1);
        Eigen::Vector3d Phi2 = QSUtility::Phi(Q1,Q2);
        Eigen::Vector3d Phi3 = QSUtility::Phi(Q2,Q3);


        double b1 = QSUtility::beta1(u);
        double b2 = QSUtility::beta2(u);
        double b3 = QSUtility::beta3(u);

        double dot_b1 = QSUtility::dot_beta1(mTimeInterval,u);
        double dot_b2 = QSUtility::dot_beta2(mTimeInterval,u);
        double dot_b3 = QSUtility::dot_beta3(mTimeInterval,u);

        double dot_dot_b1 = QSUtility::dot_dot_beta1(mTimeInterval,u);
        double dot_dot_b2 = QSUtility::dot_dot_beta2(mTimeInterval,u);
        double dot_dot_b3 = QSUtility::dot_dot_beta3(mTimeInterval,u);

        Quaternion  ddr1 = QSUtility::d2r_dt2(dot_dot_b1,dot_b1,b1,Phi1);
        Quaternion  ddr2 = QSUtility::d2r_dt2(dot_dot_b2,dot_b2,b2,Phi2);
        Quaternion  ddr3 = QSUtility::d2r_dt2(dot_dot_b3,dot_b3,b3,Phi3);

        Quaternion dr1 = QSUtility::dr_dt(dot_b1,b1,Phi1);
        Quaternion dr2 = QSUtility::dr_dt(dot_b2,b2,Phi2);
        Quaternion dr3 = QSUtility::dr_dt(dot_b3,b3,Phi3);



        Quaternion r1 = QSUtility::r(b1,Phi1);
        Quaternion r2 = QSUtility::r(b2,Phi2);
        Quaternion r3 = QSUtility::r(b3,Phi3);

        Quaternion dot_dot_q, part11, part12, part13, part21, part22, part23, part31, part32, part33;
        part11 = quatLeftComp(ddr1)*quatLeftComp(r2)*r3;
        part12 = quatLeftComp(dr1)*quatLeftComp(dr2)*r3;
        part13 = quatLeftComp(dr1)*quatLeftComp(r2)*dr3;

        part21 = quatLeftComp(dr1)*quatLeftComp(dr2)*r3;
        part22 = quatLeftComp(r1)*quatLeftComp(ddr2)*r3;
        part23 = quatLeftComp(r1)*quatLeftComp(dr2)*dr3;

        part31 = quatLeftComp(dr1)*quatLeftComp(r2)*dr3;
        part32 = quatLeftComp(r1)*quatLeftComp(dr2)*dr3;
        part33 = quatLeftComp(r1)*quatLeftComp(r2)*ddr3;


        dot_dot_q = quatLeftComp(Q0)*(part11 + part12 + part13
                                      + part21 + part22 + part23
                                      + part31 + part32 + part33);

        Quaternion Q_LG =  quatLeftComp(Q0)*quatLeftComp(r1)*quatLeftComp(r2)*r3;

        return QSUtility::alpha<double>(Q_LG,dot_dot_q);


    }


        Eigen::Vector3d QuaternionSpline::evalNumRotOmega(real_t t){
        double eps = 1e-5;
        double t_p = t + eps;
        double t_m = t - eps;

        Quaternion q_p = evalQuatSpline(t_p);
        Quaternion q_m = evalQuatSpline(t_m);

        return getOmegaFromTwoQuaternion<double>(q_m,q_p,2.0*eps);

    }

    void QuaternionSpline::initialNewControlPoint(){

        Quaternion unit = unitQuat<double>();
        double* data = new double[4];
        memcpy(data, unit.data(),sizeof(double)*4);
        mControlPointsParameter.push_back(data);
    }


}
