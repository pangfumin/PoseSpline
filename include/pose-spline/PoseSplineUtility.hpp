#ifndef POSESPLINEUTILITY_H
#define POSESPLINEUTILITY_H


#include "pose-spline/QuaternionSplineUtility.hpp"
#include "Pose.hpp"

class PSUtility {
public:
    static Pose EvaluateQS(double u, const Pose& P0, const Pose& P1,
                            const Pose& P2, const Pose& P3);
};

#endif

