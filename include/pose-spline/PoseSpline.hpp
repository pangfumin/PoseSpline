#ifndef POSE_SPLINE_H_
#define POSE_SPLINE_H_


#include "splines/bspline.hpp"
#include "pose-spline/Pose.hpp"


namespace  ze {
    class PoseSpline : public BSpline {

    public:
        PoseSpline(int spline_order);

        PoseSpline(int spline_order, double interval);

        virtual ~PoseSpline();

        void setTimeInterval(double timeInterval);

        double getTimeInterval();

        bool isTsEvaluable(double ts);

        void addSample(double t, Pose Q);

        void initialPoseSpline(std::vector<std::pair<double, Pose>> Meas);

        void initialPoseSplineKnot(double t);

        void printKnots();

        Pose evalPoseSpline(real_t t);
        Eigen::Vector3d evalLinearVelocity(real_t t );



        void evalPoseSplineDerivate(real_t t,
                                    double *Quat,
                                    double *dot_Qaut = NULL,
                                    double *dot_dot_Quat = NULL);

        inline size_t getControlPointNum(){
            return mControlPointsParameter.size();
        }

        inline double* getControlPoint(unsigned int i){
            return mControlPointsParameter.at(i);
        }

    private:
        void initialNewControlPoint();

        std::vector<double *> mControlPointsParameter;
        std::vector<Pose> mControlPointPoses;
        std::map<double, Pose> mSampleValues;
        double mTimeInterval;
    };


}

#endif