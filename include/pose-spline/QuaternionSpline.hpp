#ifndef QUATERNIONSPLINE_H
#define QUATERNIONSPLINE_H

#include "pose-spline/Quaternion.hpp"
#include "splines/bspline.hpp"
#include "utility/Time.hpp"


namespace ze {

    /*
     * Quaternion Spline Class .
     * Now only cubic spline is supported.
     */
    class QuaternionSpline : public BSpline {

    public:
        QuaternionSpline(int spline_order);
        QuaternionSpline(int spline_order,double interval);

        virtual ~QuaternionSpline();
        void setTimeInterval(double timeInterval);
        double getTimeInterval();
        bool isTsEvaluable(double ts);

        void addSample(double t, Quaternion Q);
        void initialQuaternionSpline(std::vector<std::pair<double,Quaternion>> Meas);

        void initialQuaternionSplineKnot(double t);

        void printKnots();
        Quaternion evalQuatSpline(real_t t);
        Quaternion evalDotQuatSpline(real_t t);
        Quaternion evalDotDotQuatSpline(real_t t);

        void evalQuatSplineDerivate(real_t t,
                                          double* Quat,
                                          double* dot_Qaut = NULL,
                                          double* dot_dot_Quat = NULL);


        Eigen::Vector3d evalOmega(real_t t);
        Eigen::Vector3d evalAlpha(real_t t);

            Eigen::Vector3d evalNumRotOmega(real_t t);

        inline size_t getControlPointNum(){
            return mControlPointsParameter.size();
        }

        inline double* getControlPoint(unsigned int i){
            return mControlPointsParameter.at(i);
        }
    private:
        void initialNewControlPoint();
        std::vector<double*> mControlPointsParameter;
        std::map<double, Quaternion> mSampleValues;
        double mTimeInterval;

    };
}

#endif