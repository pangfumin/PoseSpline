#ifndef _BSPLINE_BASE_H_
#define _BSPLINE_BASE_H_

#include <vector>
#include <Eigen/Core>
#include <glog/logging.h>
#include "TypeTraits.hpp"

using real_t = double;

template <typename ElementType, int SplineOrder>
class BSplineBase {
public:
    BSplineBase(double interval):mSplineOrder(SplineOrder),
                                 mTimeInterval(interval) {};
    virtual ~BSplineBase() {
        for (auto i : mControlPointsParameter)
            delete [] i;

        mControlPointsParameter.clear();
    };

    int spline_order() const {
        return SplineOrder;
    }

    /**
     *
     * @return The degree of polynomial used by the spline.
     */
    int polynomialDegree() const {
        return SplineOrder - 1;
    }

    /**
     *
     * @return the minimum number of knots required to have at least one valid
     * time segment.
     */
    int minimumKnotsRequired() const
    {
        return numKnotsRequired(1);
    }

    int numCoefficientsRequired(int num_time_segments) const
    {
        return num_time_segments + SplineOrder - 1;
    }

    int numKnotsRequired(int num_time_segments) const
    {
        return numCoefficientsRequired(num_time_segments) + SplineOrder;
    }

    real_t t_min() const
    {
        CHECK_GE((int)knots_.size(), minimumKnotsRequired())
            << "The B-spline is not well initialized";
        return knots_[SplineOrder - 1];
    }

    real_t t_max() const
    {
        CHECK_GE((int)knots_.size(), minimumKnotsRequired())
            << "The B-spline is not well initialized";
        return knots_[knots_.size() - SplineOrder];
    }

    std::pair<real_t,int> computeTIndex(real_t t) const
    {
        CHECK_GE(t, t_min()) << "The time is out of range by " << (t - t_min());

        //// HACK - avoids numerical problems on initialisation
        if (std::abs(t_max() - t) < 1e-10)
        {
            t = t_max();
        }
        //// \HACK

        CHECK_LE(t, t_max())
            << "The time is out of range by " << (t_max() - t);
        std::vector<real_t>::const_iterator i;
        if(t == t_max())
        {
            // This is a special case to allow us to evaluate the spline at the boundary of the
            // interval. This is not stricly correct but it will be useful when we start doing
            // estimation and defining knots at our measurement times.
            i = knots_.end() - SplineOrder;
        }
        else
        {
            i = std::upper_bound(knots_.begin(), knots_.end(), t);
        }
        //CHECK_NE(i, knots_.end()) << "Something very bad has happened in computeTIndex(" << t << ")";

        // Returns the index of the knot segment this time lies on and the width of this knot segment.
        return std::make_pair(*i - *(i-1),(i - knots_.begin()) - 1);

    }

    std::pair<real_t,int> computeUAndTIndex(real_t t) const
    {
        std::pair<real_t,int> ui = computeTIndex(t);

        int index = ui.second;
        real_t denom = ui.first;

        if(denom <= 0.0)
        {
            // The case of duplicate knots.
            //std::cout << "Duplicate knots\n";
            return std::make_pair(0, index);
        }
        else
        {
            real_t u = (t - knots_[index])/denom;

            return std::make_pair(u, index);
        }
    }

    void setTimeInterval(double timeInterval){
        mTimeInterval = timeInterval;

    }
    double getTimeInterval(){
        return mTimeInterval ;

    }
    bool isTsEvaluable(double ts){
        return ts >= t_min() && ts < t_max();

    }

    void initialSplineKnot(double t){
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
            knots[i] = t + (i - SplineOrder + 1) * dt;
        }

        knots_ = knots;

        for(int i = 0; i < C; i++){
            initialNewControlPoint();
        }
    }

    void addElemenTypeSample(double t, ElementType sample){
        if(getControlPointNum() == 0){
            initialSplineKnot(t);
        }else if(getControlPointNum() >= numCoefficientsRequired(1) ){
            if(t < t_min()){
                std::cerr<<"[Error] Inserted t is smaller than t_min()！"<<std::endl;
//                LOG(FATAL) << "Inserted "<<Time(t)<<" is smaller than t_min() "<<Time(t_min())<<std::endl;
            }else if(t >= t_max()){
                // add new knot and control Points
                while(t >= t_max()){
                    knots_.push_back(knots_.back() + mTimeInterval); // append one;
                    initialNewControlPoint();
                }
            }
        }
        // Tricky: do not add point close to t_max
        if( t_max() - t > 0.0001){
            mSampleValues.insert(std::pair<double ,ElementType>(t,sample));
        }
        CHECK_EQ(knots_.size() - SplineOrder, getControlPointNum());

    }

    void addControlPointsUntil(double t){
        if(getControlPointNum() == 0){
            initialSplineKnot(t);
        }else if(getControlPointNum() >= numCoefficientsRequired(1) ){
            if(t < t_min()){
                std::cerr<<"[Error] Inserted t is smaller than t_min()！"<<std::endl;
//                LOG(FATAL) << "Inserted "<<Time(t)<<" is smaller than t_min() "<<Time(t_min())<<std::endl;
            }else if(t >= t_max()){
                // add new knot and control Points
                while(t >= t_max()){
                    knots_.push_back(knots_.back() + mTimeInterval); // append one;
                    initialNewControlPoint();
                }
            }
        }

        CHECK_EQ(knots_.size() - SplineOrder, getControlPointNum());

    }

    inline size_t getControlPointNum(){
        return mControlPointsParameter.size();
    }

    inline double* getControlPoint(unsigned int i){
        return mControlPointsParameter.at(i);
    }


private:
    void initialNewControlPoint(){
        typename TypeTraits<ElementType>::TypeT zero_ele = TypeTraits<ElementType>::zero();
        double* data = new double[TypeTraits<ElementType>::Dim];
        memcpy(data, zero_ele.data(),sizeof(double)* TypeTraits<ElementType>::Dim);
        mControlPointsParameter.push_back(data);
    }

    /// The knot sequence used by the B-spline.
    std::vector<real_t> knots_;

    std::vector<double*> mControlPointsParameter;
    std::map<double, ElementType> mSampleValues;
    int mSplineOrder;
    double mTimeInterval;
};
#endif