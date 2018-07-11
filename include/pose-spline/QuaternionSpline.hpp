#ifndef QUATERNIONSPLINE_H
#define QUATERNIONSPLINE_H

#include "geometry/Quaternion.hpp"
#include "pose-spline/BSplineBase.hpp"
#include "okvis_util/Time.hpp"


class QuaternionSpline : public BSplineBase<Quaternion, 4> {

public:
    typedef BSplineBase<Quaternion, 4> base_t;
    QuaternionSpline();
    QuaternionSpline(double interval);

    virtual ~QuaternionSpline();

    void initialQuaternionSpline(std::vector<std::pair<double,Quaternion>> Meas);



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


private:


};

#endif