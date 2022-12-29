#include <gtest/gtest.h>
#include "PoseSpline/Pose.hpp"
#include "PoseSpline/PoseLocalParameter.hpp"
#include "PoseSpline/QuaternionLocalParameter.hpp"

// todo: test poselocalparameter
TEST (Pose, PoseInit) {
    for (int i =0; i < 1000; i++) {

        Pose<double> T_A;
        T_A.setRandom();
        Eigen::Matrix<double,7,1> parameter = T_A.parameters();
        Pose<double> T_B(parameter);
        Pose<double> T_C(parameter.data());

        GTEST_ASSERT_LT(((T_A*T_B.inverse()).Transformation() - Eigen::Matrix4d::Identity()).norm(), 1e-8);
        GTEST_ASSERT_LT(((T_A*T_C.inverse()).Transformation() - Eigen::Matrix4d::Identity()).norm(), 1e-8);
    }
}

TEST(Pose , operations){
    PoseLocalParameter* poseLocalParameter = new PoseLocalParameter();
    for (size_t i = 0; i < 100; ++i) {
        Pose<double> T_AB;
        T_AB.setRandom();
        Pose<double> T_BC;
        T_BC.setRandom();
        // Test inverse
        GTEST_ASSERT_LT(
                ((T_AB * T_AB.inverse()).Transformation() - Eigen::Matrix4d::Identity()).norm()
                ,1e-8);

        // Test composition
        GTEST_ASSERT_LT(((T_AB * T_BC).Transformation() - T_AB.Transformation() * T_BC.Transformation()).norm() , 1e-8);

        // Test construction
        Pose<double> T_AB_alternative(T_AB.Transformation());
        GTEST_ASSERT_LT((T_AB.Transformation() - T_AB_alternative.Transformation()).norm() , 1e-8);
        Pose<double> T_AB_alternative2(T_AB.r(), T_AB.q());
        GTEST_ASSERT_LT((T_AB.Transformation() - T_AB_alternative2.Transformation()).norm() , 1e-8);

        // Test =
        Pose<double> T_AB_alternative3;
        T_AB_alternative3 = T_AB;
        GTEST_ASSERT_LT((T_AB.Transformation() - T_AB_alternative3.Transformation()).norm() , 1e-8);

        // Test setters
        Pose<double> T_AB_alternative4;
        T_AB_alternative4.set(T_AB.r(), T_AB.q());
        GTEST_ASSERT_LT((T_AB.Transformation() - T_AB_alternative4.Transformation()).norm() , 1e-8);
        Pose<double> T_AB_alternative5;
        T_AB_alternative5.set(T_AB.Transformation());
        GTEST_ASSERT_LT((T_AB.Transformation() - T_AB_alternative5.Transformation()).norm() , 1e-8);

        T_AB.setRandom();

        // Test oplus
        const double dp = 1.0e-6;
        Eigen::Matrix<double, 7, 6, Eigen::RowMajor> jacobian_numDiff;
        for (size_t i = 0; i < 6; ++i) {
            Pose<double> T_AB_p = T_AB;
            Pose<double> T_AB_m = T_AB;

            Pose<double> T_AB_param = T_AB;
            Eigen::Matrix<double,7,1> T_AB_p_param;
            Eigen::Matrix<double,7,1> T_AB_m_param;


            Eigen::Matrix<double, 6, 1> dp_p;
            Eigen::Matrix<double, 6, 1> dp_m;
            dp_p.setZero();
            dp_m.setZero();
            dp_p[i] = dp;
            dp_m[i] = -dp;

            poseLocalParameter->Plus(T_AB_param.parameterPtr(),dp_p.data(),T_AB_p_param.data());
            poseLocalParameter->Plus(T_AB_param.parameterPtr(),dp_m.data(),T_AB_m_param.data());

            T_AB_p.oplus(dp_p);
            T_AB_m.oplus(dp_m);

            // check poselocalparameter plus
            GTEST_ASSERT_LT((T_AB_p.parameters() - T_AB_p_param).norm() , 1e-8);
            GTEST_ASSERT_LT((T_AB_m.parameters() - T_AB_m_param).norm() , 1e-8);

            /*jacobian_numDiff.block<7, 1>(0, i) = (T_AB_p.parameters()
                - T_AB_m.parameters()) / (2.0 * dp);*/
            jacobian_numDiff.block<3, 1>(0, i) = (T_AB_p.r() - T_AB_m.r())
                                                 / (2.0 * dp);
            jacobian_numDiff.block<4, 1>(3, i) = (T_AB_p.q()
                                                  - T_AB_m.q()) / (2.0 * dp);
        }
        Eigen::Matrix<double, 7, 6, Eigen::RowMajor> jacobian;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift_jacobian;

        T_AB.oplusJacobian(jacobian);

        // test poselcoalparameter plus-jacobian and lift-jacobian
        Eigen::Matrix<double, 7, 6, Eigen::RowMajor> local_plus_jacobian;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> local_lift_jacobian;

        poseLocalParameter->ComputeJacobian(T_AB.parameterPtr(),local_plus_jacobian.data());
        poseLocalParameter->liftJacobian(T_AB.parameterPtr(),local_lift_jacobian.data());

        GTEST_ASSERT_LT((local_plus_jacobian - jacobian).squaredNorm() , 1e-8);
        GTEST_ASSERT_LT((local_lift_jacobian*local_plus_jacobian -
              Eigen::Matrix<double,6,6>::Identity()).squaredNorm() , 1e-8);

        //std::cout << jacobian_numDiff << std::endl;
        GTEST_ASSERT_LT((jacobian - jacobian_numDiff).norm() , 1e-8);
        // also check lift Jacobian: dChi/dx*dx/dChi == 1
        T_AB.liftJacobian(lift_jacobian);
        GTEST_ASSERT_LT(
                (lift_jacobian * jacobian - Eigen::Matrix<double, 6, 6>::Identity())
                        .norm() , 1e-8);
    }

}

TEST (Pose, QuaternionPlusList) {
    for (int i =0; i < 1000; i++) {
        Quaternion q2 = randomQuat<double>();
        q2 = quatNorm(q2);

        Eigen::Matrix<double,4,3,Eigen::RowMajor> plusJacobian;
        Eigen::Matrix<double,3,4,Eigen::RowMajor> liftJacobian;

        QuaternionLocalParameter* quaternionLocalParam = new QuaternionLocalParameter;

        quaternionLocalParam->ComputeJacobian(q2.data(),plusJacobian.data());
        quaternionLocalParam->liftJacobian(q2.data(),liftJacobian.data());

//        std::cout<<"Liftjac* pluJac: "<<std::endl<<
//                 liftJacobian*plusJacobian<<std::endl;
//
//        std::cout << "liftJacobian: \n" << liftJacobian << std::endl;
//        std::cout << "plusJacobian: \n" << plusJacobian << std::endl;

        GTEST_ASSERT_LT((liftJacobian*plusJacobian - Eigen::Matrix3d::Identity()).norm(), 1e-8);
    }

    
}

TEST (Pose, PosePlusLift) {
    for (int i =0; i < 1000; i++) {
        Pose<double> T;
        T.setRandom();
        Quaternion q = T.q();

        Eigen::Matrix<double,7,6,Eigen::RowMajor> plusJacobian;
        Eigen::Matrix<double,6,7,Eigen::RowMajor> liftJacobian;

        PoseLocalParameter* poseLocalParameter = new PoseLocalParameter;

        poseLocalParameter->ComputeJacobian(T.data(),plusJacobian.data());
        poseLocalParameter->liftJacobian(T.data(),liftJacobian.data());

//        std::cout<<"Liftjac* pluJac: "<<std::endl<<
//                 liftJacobian*plusJacobian<<std::endl;
//
//        std::cout << "liftJacobian: \n" << liftJacobian << std::endl;
//        std::cout << "plusJacobian: \n" << plusJacobian << std::endl;

        GTEST_ASSERT_LT((liftJacobian*plusJacobian -
        Eigen::Matrix<double,6,6>::Identity()).norm(), 1e-8);

        Eigen::Matrix<double,4,3,Eigen::RowMajor> quatPlusJacobian;
        Eigen::Matrix<double,3,4,Eigen::RowMajor> quatLiftJacobian;

        QuaternionLocalParameter* quaternionLocalParam = new QuaternionLocalParameter;

        quaternionLocalParam->ComputeJacobian(q.data(),quatPlusJacobian.data());
        quaternionLocalParam->liftJacobian(q.data(),quatLiftJacobian.data());

        GTEST_ASSERT_LT((liftJacobian.bottomRightCorner(3,4) -
                quatLiftJacobian).norm(), 1e-8);

        GTEST_ASSERT_LT((plusJacobian.bottomRightCorner(4,3) -
                quatPlusJacobian).norm(), 1e-8);
    }


}

TEST (Pose, PoseRotation) {
    Pose<double> T;
    T.setRandom();
    Eigen::Matrix3d R = quatToRotMat<double>(T.q());

    GTEST_ASSERT_LT((R*T.C().inverse() - Eigen::Matrix3d::Identity()).norm(), 1e-8);

}

TEST (Pose, PoseTransformVector) {
    for (int i =0; i < 1000; i++) {
        Pose<double> T_AB;
        T_AB.setRandom();
        Eigen::Vector3d Bp = Eigen::Vector3d::Random();

        Eigen::Matrix3d R_AB = T_AB.C();
        Eigen::Vector3d t_AB = T_AB.r();

        Eigen::Vector3d Ap = R_AB * Bp + t_AB;
        Eigen::Vector3d Ap_ = T_AB * Bp;

        Eigen::Vector3d Bp_ = T_AB.inverse() * Ap_;
        Eigen::Vector3d Bp__ = R_AB.inverse() * (Ap_ - T_AB.r());


        GTEST_ASSERT_LT((Ap - Ap_).norm(), 1e-8);
        GTEST_ASSERT_LT((Bp - Bp_).norm(), 1e-8);
        GTEST_ASSERT_LT((Bp - Bp__).norm(), 1e-8);
    }

}