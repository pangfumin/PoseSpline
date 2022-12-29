#include "extern/TransformVectorError.hpp"
#include "PoseSpline/QuaternionSplineUtility.hpp"


TransformVectorError::TransformVectorError(double t_meas,
                                       Eigen::Vector3d originalVector,
                                     Eigen::Vector3d transformedVector):
        t_meas_(t_meas),originalVector_(originalVector),
        transformedVector_Meas_(transformedVector){

};

TransformVectorError::~TransformVectorError() {
}
bool TransformVectorError::Evaluate(double const* const* parameters,
                                           double* residuals,
                                           double** jacobians) const{

    return EvaluateWithMinimalJacobians(parameters,residuals,jacobians,NULL);

}

bool TransformVectorError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                                               double* residuals,
                                                               double** jacobians,
                                                               double** jacobiansMinimal) const {
    Eigen::Map<const Eigen::Vector3d> t0(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t1(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> t2(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> t3(parameters[3]);
    Eigen::Map<const Quaternion> Q0(parameters[0] + 3);
    Eigen::Map<const Quaternion> Q1(parameters[1] + 3);
    Eigen::Map<const Quaternion> Q2(parameters[2] + 3);
    Eigen::Map<const Quaternion> Q3(parameters[3] + 3);


//    std::cout<<"Quaternion: "<<std::endl;
//    std::cout<<Q0.transpose()<<std::endl;
//    std::cout<<Q1.transpose()<<std::endl;
//    std::cout<<Q2.transpose()<<std::endl;
//    std::cout<<Q3.transpose()<<std::endl;
//
//    std::cout<<"trans: "<<std::endl;
//    std::cout<<t0.transpose()<<std::endl;
//    std::cout<<t1.transpose()<<std::endl;
//    std::cout<<t2.transpose()<<std::endl;
//    std::cout<<t3.transpose()<<std::endl;


    Eigen::Map<Eigen::Vector3d> error(residuals);


    double  Beta1 = QSUtility::beta1(t_meas_);
    double  Beta2 = QSUtility::beta2(t_meas_);
    double  Beta3 = QSUtility::beta3(t_meas_);

    Eigen::Vector3d phi1 = QSUtility::Phi<double>(Q0,Q1);
    Eigen::Vector3d phi2 = QSUtility::Phi<double>(Q1,Q2);
    Eigen::Vector3d phi3 = QSUtility::Phi<double>(Q2,Q3);

    Quaternion r_1 = QSUtility::r(Beta1,phi1);
    Quaternion r_2 = QSUtility::r(Beta2,phi2);
    Quaternion r_3 = QSUtility::r(Beta3,phi3);

    // define residual
    // For simplity, we define error  =  /hat - meas.
    Quaternion Q_hat = quatLeftComp<double>(Q0)*quatLeftComp(r_1)*quatLeftComp(r_2)*r_3;
    Eigen::Vector3d t_hat = t0 + Beta1*(t1 - t0) +  Beta2*(t2 - t1) + Beta3*(t3 - t2);

    Eigen::Vector3d rotatedVector_hat = quatToRotMat<double>(Q_hat)*originalVector_;
    Eigen::Vector3d transtformedVector_hat = rotatedVector_hat + t_hat;
    error =  transtformedVector_hat - transformedVector_Meas_;

    if(jacobians != NULL){
        Eigen::Matrix<double,3,4,Eigen::RowMajor> lift;
        QuaternionLocalParameter::liftJacobian(Q_hat.data(),lift.data());

        Eigen::Matrix<double,3,4> J_1st;
        J_1st.setZero();
        J_1st = crossMat(rotatedVector_hat) * lift;

        // std::cout<<"J_1st: "<<J_1st<<std::endl;

        Eigen::Matrix<double,4,3> Vee = QSUtility::V<double>();

        Eigen::Vector3d BetaPhi1 = Beta1*phi1;
        Eigen::Vector3d BetaPhi2 = Beta2*phi2;
        Eigen::Vector3d BetaPhi3 = Beta3*phi3;
        Eigen::Matrix3d S1 = quatS(BetaPhi1);
        Eigen::Matrix3d S2 = quatS(BetaPhi2);
        Eigen::Matrix3d S3 = quatS(BetaPhi3);


        Quaternion invQ0Q1 = quatLeftComp(quatInv<double>(Q0))*Q1;
        Quaternion invQ1Q2 = quatLeftComp(quatInv<double>(Q1))*Q2;
        Quaternion invQ2Q3 = quatLeftComp(quatInv<double>(Q2))*Q3;
        Eigen::Matrix3d L1 = quatL(invQ0Q1);
        Eigen::Matrix3d L2 = quatL(invQ1Q2);
        Eigen::Matrix3d L3 = quatL(invQ2Q3);

        Eigen::Matrix3d C0 = quatToRotMat<double>(Q0);
        Eigen::Matrix3d C1 = quatToRotMat<double>(Q1);
        Eigen::Matrix3d C2 = quatToRotMat<double>(Q2);


        Eigen::Matrix<double,4,3> temp0;
        Eigen::Matrix<double,4,3> temp01;
        Eigen::Matrix<double,4,3> temp12;
        Eigen::Matrix<double,4,3> temp23;


        temp0 = quatRightComp<double>(quatLeftComp<double>(r_1)*quatLeftComp<double>(r_2)*r_3)*quatRightComp<double>(Q0)*Vee;
        temp01 = quatLeftComp<double>(Q0)*quatRightComp<double>(quatLeftComp<double>(r_2)*r_3)*quatRightComp<double>(r_1)*Vee*S1*Beta1*L1*C0.transpose();
        temp12 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r_1)*quatRightComp<double>(r_3)*quatRightComp<double>(r_2)*Vee*S2*Beta2*L2*C1.transpose();
        temp23 = quatLeftComp<double>(Q0)*quatLeftComp<double>(r_1)*quatLeftComp<double>(r_2)*quatRightComp<double>(r_3)*Vee*S3*Beta3*L3*C2.transpose();

        if(jacobians[0] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J0(jacobians[0]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> quat_J0_minimal;
            Eigen::Matrix<double,3,3,Eigen::RowMajor> trans_J0_minimal;

            trans_J0_minimal  = (1 - Beta1)*Eigen::Matrix3d::Identity();
            //
            Eigen::Matrix<double,4,3> quat_J_spline0;

            quat_J_spline0 = temp0 - temp01;


            quat_J0_minimal  = J_1st * quat_J_spline0;


            Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
            QuaternionLocalParameter::liftJacobian(Q0.data(),J_lift.data());
            J0  << trans_J0_minimal, quat_J0_minimal * J_lift;

            //std::cout<<"J0: "<<std::endl<<J0<<std::endl;

            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J0_minimal_map(jacobiansMinimal[0]);
                J0_minimal_map << trans_J0_minimal, quat_J0_minimal;

                //std::cout<<"J0_minimal_map: "<<std::endl<<J0_minimal_map<<std::endl;

            }

        }
        if(jacobians[1] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J1(jacobians[1]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> quat_J1_minimal;
            Eigen::Matrix<double,3,3,Eigen::RowMajor> trans_J1_minimal;

            Eigen::Matrix<double,4,3> quat_J_spline1;

            quat_J_spline1 = temp01 - temp12;
            quat_J1_minimal  = J_1st * quat_J_spline1;

            Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
            QuaternionLocalParameter::liftJacobian(Q1.data(),J_lift.data());

            J1 << trans_J1_minimal, quat_J1_minimal * J_lift;

            //std::cout<<"J1: "<<std::endl<<J1<<std::endl;

            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J1_minimal_map(jacobiansMinimal[1]);
                J1_minimal_map << trans_J1_minimal, quat_J1_minimal;

            }

        }
        if(jacobians[2] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J2(jacobians[2]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> quat_J2_minimal;
            Eigen::Matrix<double,3,3,Eigen::RowMajor> trans_J2_minimal;
            //
            Eigen::Matrix<double,4,3> quat_J_spline2;


            quat_J_spline2 = temp12 - temp23;
            quat_J2_minimal  = J_1st*quat_J_spline2;

            Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
            QuaternionLocalParameter::liftJacobian(Q2.data(),J_lift.data());

            J2 << trans_J2_minimal, quat_J2_minimal*J_lift;

            //std::cout<<"J2: "<<std::endl<<J2<<std::endl;


            if(jacobiansMinimal != NULL &&  jacobiansMinimal[2] != NULL){
                Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J2_minimal_map(jacobiansMinimal[2]);
                J2_minimal_map << trans_J2_minimal, quat_J2_minimal;

            }

        }
        if(jacobians[3] != NULL){
            Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> J3(jacobians[3]);
            Eigen::Matrix<double,3,3,Eigen::RowMajor> quat_J3_minimal;
            Eigen::Matrix<double,3,3,Eigen::RowMajor> trans_J3_minimal;
            //
            Eigen::Matrix<double,4,3> quat_J_spline3;

            quat_J_spline3 = temp23 ;
            quat_J3_minimal = J_1st * quat_J_spline3;
            Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
            QuaternionLocalParameter::liftJacobian(Q3.data(),J_lift.data());

            J3 << trans_J3_minimal, quat_J3_minimal*J_lift;

            //std::cout<<"J3: "<<std::endl<<J3<<std::endl;


            if(jacobiansMinimal != NULL &&  jacobiansMinimal[3] != NULL){

                Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>> J3_minimal_map(jacobiansMinimal[3]);
                J3_minimal_map << trans_J3_minimal, quat_J3_minimal;

            }

        }
    }

    return true;
}



