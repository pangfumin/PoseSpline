#ifndef _TYPE_TRAITS_H_
#define _TYPE_TRAITS_H_

#include <Eigen/Core>
#include "Pose.hpp"
template <typename T>
class TypeTraits;

template <>
class TypeTraits<Quaternion> {
public:
    typedef Quaternion TypeT;
    enum {Dim = 4, miniDim = 3};
    static TypeT zero() {
        return unitQuat<double>();
    }
};


template <>
class TypeTraits<Pose<double>> {
public:
    typedef Pose<double> TypeT;
    enum {Dim = 7, miniDim = 6};
    static TypeT zero() {
        return Pose<double>();
    }
};


template <int D>
class TypeTraits<Eigen::Matrix<double, D,1>> {
public:
    typedef Eigen::Matrix<double,D,1> TypeT;
    enum {Dim = D, miniDim = D};
    static TypeT zero() {
        return Eigen::Matrix<double,D,1>::Zero();
    }
};


#endif