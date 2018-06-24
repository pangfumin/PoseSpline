#ifndef POSESPLINEUTILITY_H
#define POSESPLINEUTILITY_H


#include "pose-spline/QuaternionSplineUtility.hpp"
#include "Pose.hpp"

class PSUtility {
public:
    static Pose EvaluatePS(double u, const Pose& P0, const Pose& P1,
                            const Pose& P2, const Pose& P3);
};

class PoseSplineEvaluation {
public:
    typedef Eigen::Matrix<double, 3,3>  TranslationJacobian;
    typedef Eigen::Matrix<double, 4,3>  RotationJacobian;
    Pose operator() (double u, const Pose& P0, const Pose& P1,
                     const Pose& P2, const Pose& P3);
    template <int D>
    TranslationJacobian  getTranslationJacobian () {
        if (D == 0){
            return JacobianTrans0;
        }else if(D == 1){
            return JacobianTrans1;
        }else if(D == 2){
            return JacobianTrans2;
        }else {
            return JacobianTrans3;
        }
    }
    template <int D>
    RotationJacobian  getRotationJacobianMinimal () {
        if (D == 0){
            return JacobianRotate0;
        }else if(D == 1){
            return JacobianRotate1;
        }else if(D == 2){
            return JacobianRotate2;
        }else {
            return JacobianRotate3;
        }
    }

    inline Quaternion getRotation() {
        return Q;
    }

    inline Eigen::Vector3d getTranslation() {
        return t;
    }

private:
    Quaternion Q;
    Eigen::Vector3d t;
    TranslationJacobian JacobianTrans0,JacobianTrans1,JacobianTrans2,JacobianTrans3;
    RotationJacobian JacobianRotate0,JacobianRotate1,JacobianRotate2,JacobianRotate3;
};

#endif

