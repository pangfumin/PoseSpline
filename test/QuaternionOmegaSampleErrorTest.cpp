#include <iostream>
#include "pose-spline/QuaternionOmegaSampleError.hpp"
int main(){

    Quaternion Q0,Q1,Q2,Q3;
    Q0 = Quaternion(1,8,3,5);
    Q0 = Q0/Q0.norm();

    Q1 = Quaternion(1,8,3,50);
    Q1 = Q1/Q1.norm();

    Q2 = Quaternion(1,8,3,-50);
    Q2 = Q2/Q2.norm();

    Q3 = Quaternion(1,-108,3,50);
    Q3 = Q3/Q3.norm();


    return 0;
}