#ifndef VECTORSPACESPLINE_H
#define VECTORSPACESPLINE_H

#include <Eigen/Core>
#include "splines/BSplineBase.hpp"
#include "okvis_util/Time.hpp"


class VectorSpaceSpline : public BSplineBase<Eigen::Vector3d, 4> {
public:
    VectorSpaceSpline();
    VectorSpaceSpline(double interval);

    void initialSpline(std::vector<std::pair<double,Eigen::Vector3d>> Meas);
    Eigen::Vector3d evaluateSpline(const real_t t);
    Eigen::Vector3d evaluateDotSpline(const real_t t);
    Eigen::Vector3d evaluateDotSplineNumeric(const real_t t);


    static Eigen::Vector3d evaluateSpline(const real_t t,
                                          const Eigen::Vector3d& v0,
                                          const Eigen::Vector3d& v1,
                                          const Eigen::Vector3d& v2,
                                          const Eigen::Vector3d& v3);
    static Eigen::Vector3d evaluateDotSpline(const real_t t, const double timeInterval,
                                          const Eigen::Vector3d& v0,
                                          const Eigen::Vector3d& v1,
                                          const Eigen::Vector3d& v2,
                                          const Eigen::Vector3d& v3);


};


#endif