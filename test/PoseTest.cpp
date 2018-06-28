
#include "utility/catch/catch.hpp"
#include "pose-spline/Pose.hpp"
#include "pose-spline/PoseLocalParameter.hpp"

// todo: test poselocalparameter

TEST_CASE( "Pose ", "[operations]"){


    PoseLocalParameter* poseLocalParameter = new PoseLocalParameter();
    for (size_t i = 0; i < 100; ++i) {
        Pose<double> T_AB;
        T_AB.setRandom();
        Pose<double> T_BC;
        T_BC.setRandom();


        // Test inverse
        REQUIRE(
                ((T_AB * T_AB.inverse()).Transformation() - Eigen::Matrix4d::Identity()).norm()
                < 1e-8);

        // Test composition
        REQUIRE(((T_AB * T_BC).Transformation() - T_AB.Transformation() * T_BC.Transformation()).norm() < 1e-8);

        // Test construction
        Pose<double> T_AB_alternative(T_AB.Transformation());
        REQUIRE((T_AB.Transformation() - T_AB_alternative.Transformation()).norm() < 1e-8);
        Pose<double> T_AB_alternative2(T_AB.r(), T_AB.q());
        REQUIRE((T_AB.Transformation() - T_AB_alternative2.Transformation()).norm() < 1e-8);

        // Test =
        Pose<double> T_AB_alternative3;
        T_AB_alternative3 = T_AB;
        REQUIRE((T_AB.Transformation() - T_AB_alternative3.Transformation()).norm() < 1e-8);

        // Test setters
        Pose<double> T_AB_alternative4;
        T_AB_alternative4.set(T_AB.r(), T_AB.q());
        REQUIRE((T_AB.Transformation() - T_AB_alternative4.Transformation()).norm() < 1e-8);
        Pose<double> T_AB_alternative5;
        T_AB_alternative5.set(T_AB.Transformation());
        REQUIRE((T_AB.Transformation() - T_AB_alternative5.Transformation()).norm() < 1e-8);

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
            REQUIRE((T_AB_p.parameters() - T_AB_p_param).norm() < 1e-8);
            REQUIRE((T_AB_m.parameters() - T_AB_m_param).norm() < 1e-8);

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

        REQUIRE((local_plus_jacobian - jacobian).squaredNorm() < 1e-8);
        REQUIRE((local_lift_jacobian*local_plus_jacobian - Eigen::Matrix<double,6,6>::Identity()).squaredNorm() < 1e-8);

        //std::cout << jacobian_numDiff << std::endl;
        REQUIRE((jacobian - jacobian_numDiff).norm() < 1e-8);
        // also check lift Jacobian: dChi/dx*dx/dChi == 1
        T_AB.liftJacobian(lift_jacobian);
        REQUIRE(
                (lift_jacobian * jacobian - Eigen::Matrix<double, 6, 6>::Identity())
                        .norm() < 1e-8);


    }

}