#ifndef POSE_SPLINE_H_
#define POSE_SPLINE_H_


#include "PoseSpline/BSplineBase.hpp"
#include "PoseSpline/Pose.hpp"


class PoseSpline : public BSplineBase<Pose<double>, 4> {

public:
    PoseSpline();

    PoseSpline( double interval);

    virtual ~PoseSpline() {}

    void initialPoseSpline(std::vector<std::pair<double, Pose<double>>> Meas) ;

    Pose<double> evalPoseSpline(real_t t);
    Eigen::Vector3d evalLinearVelocity(real_t t );

    Eigen::Vector3d evalLinearAccelerator(real_t t, const Eigen::Vector3d& gravity);
    Eigen::Vector3d evalOmega(real_t t);
};
#endif