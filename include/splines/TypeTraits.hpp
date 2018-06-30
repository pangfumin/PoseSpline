#ifndef _TYPE_TRAITS_H_
#define _TYPE_TRAITS_H_

#include <Eigen/Core>
#include "pose-spline/Quaternion.hpp"
#include "pose-spline/Pose.hpp"
template <typename T>
class TypeTraits;

template <>
class TypeTraits<Quaternion> {
public:
    typename Quaternion TypeT;
    enum {Dim = 4, miniDim = 3};
    static TypeT zero() {
        return unitQuat();
    }
};


template <>
class TypeTraits<Pose> {
public:
    typename Pose TypeT;
    enum {Dim = 7, miniDim = 6};
    static TypeT zero() {
        return Pose();
    }
};


template <>
class TypeTraits<Eigen::Vector3d> {
public:
    typename Eigen::Vector3d TypeT;
    enum {Dim = 3, miniDim = 3};
    static TypeT zero() {
        return Eigen::Vector3d::Identity();
    }
};


#endif