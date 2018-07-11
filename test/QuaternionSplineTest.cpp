#include <iostream>
#include <fstream>

#include "common/csv_trajectory.hpp"
#include "pose-spline/QuaternionSpline.hpp"
#include <pose-spline/QuaternionSplineUtility.hpp>


using namespace ze;

std::pair<double,Quaternion>  getSample(ze::TupleVector& data, unsigned int i){
    ze::TrajectoryEle  p0 = data.at(i);
    Quaternion q = std::get<2>(p0);
    return std::make_pair(std::get<0>(p0)*1e-9,q);
};




int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    std::string dataset = "/home/pang/software/PoseSpline/data/MH_01_easy";
    ze::EurocResultSeries eurocDataReader;
    eurocDataReader.load(dataset + "/state_groundtruth_estimate0/data.csv");
    eurocDataReader.loadIMU(dataset+ "/imu0/data.csv");

    ze::TupleVector  data = eurocDataReader.getVector();
    Buffer<real_t, 7>& poseBuffer = eurocDataReader.getBuffer();
    LOG(INFO)<<"Get data size: "<<data.size(); // @200Hz
    std::vector<std::pair<int64_t ,Eigen::Vector3d>> linearVelocities = eurocDataReader.getLinearVelocities();
    std::vector<Vector3> gyroBias = eurocDataReader.getGyroBias();
    LOG(INFO)<<"Get velocities size: "<<linearVelocities.size(); // @200Hz
    LOG(INFO)<<"Get gyro_bias  size: "<<gyroBias.size(); // @200Hz

    std::vector<int64_t> imu_ts = eurocDataReader.getIMUts();
    LOG(INFO)<<"Get IMU ts  size: "<<imu_ts.size(); // @200Hz

    std::vector<Vector3> gyroMeas = eurocDataReader.getGyroMeas();
    LOG(INFO)<<"Get gyro Meas  size: "<<gyroMeas.size(); // @200Hz



    int start  = 1;
    int end = data.size() - 2;

    QuaternionSpline qspline(0.1);
    std::vector<std::pair<double,Quaternion>> samples, queryMeas;

    for(uint i = start; i <end; i++){
        std::pair<double,Quaternion> sample = getSample( data,  i);
        Eigen::Quaterniond QuatHamilton(sample.second(3),sample.second(0),sample.second(1),sample.second(2));
        Eigen::Matrix3d R = QuatHamilton.toRotationMatrix();
        Quaternion QuatJPL = rotMatToQuat(R);
        std::pair<double,Quaternion> sampleJPL = std::make_pair(sample.first, QuatJPL);
        queryMeas.push_back(sampleJPL);
        if(i % 4  == 0){
            samples.push_back(sampleJPL);

        }
    }

    qspline.initialQuaternionSpline(samples);

    /*
     *  Test: qspline.evalQuatSpline
     */

    for(auto i: queryMeas){

        if(qspline.isTsEvaluable(i.first)){
            Quaternion query = qspline.evalQuatSpline(i.first);

            Eigen::Vector3d diff = (quatLeftComp(i.second)*quatInv(query)).head(3);
            CHECK_EQ(diff.norm() < 0.01,true)<<"Qspline query is not close to the ground truth!"
                                              <<"Gt:    "<<i.second.transpose()<<std::endl
                                              <<"Query: "<<query.transpose()<<std::endl
                                              <<"diff:  "<<diff.transpose()<<std::endl<<std::endl;

//            std::cout <<"Gt:    "<<i.second.transpose()<<std::endl;
//            std::cout <<"Query: "<<query.transpose()<<std::endl;

        }

    }

    LOG(INFO)<<" - QuaternionSpline initialization passed!";

//    /*
//     * Check Qspline angular velocity
//     *
//     * Passed!
//     */
//
//
//    std::pair<double,Quaternion> checkPoint = getSample(data,(start + end)/2.0);
//
//    std::cout<<"ts: "<<checkPoint.first<<std::endl;
//    std::cout<<"Q : "<<checkPoint.second.transpose()<<std::endl;
//
//    Quaternion evalQ = qspline.evalQuatSpline(checkPoint.first);
//    std::cout<<"evalQ : "<<evalQ.transpose()<<std::endl;
//
//    Quaternion evalDotQ = qspline.evalDotQuatSpline(checkPoint.first);
//    std::cout<<"evalDotQ : "<<evalDotQ.transpose()<<std::endl;
//
//    double eps = 1e-5;
//
//    Quaternion evalQ_p = qspline.evalQuatSpline(checkPoint.first+eps);
//    Quaternion evalQ_m = qspline.evalQuatSpline(checkPoint.first-eps);
//    std::cout<<"evalQ_p : "<<evalQ_p.transpose()<<std::endl;
//    std::cout<<"evalQ_m : "<<evalQ_m.transpose()<<std::endl;
//
//    Quaternion numDotQ = (evalQ_p  - evalQ_m)/(2.0*eps);
//    std::cout<<"numDotQ : "<<numDotQ.transpose()<<std::endl;
//    CHECK_EQ((evalDotQ - numDotQ).norm()< 0.001,true)<<"EvalDotQ Not equal to num-DotQ!";
//
//
    std::ofstream ofs_debug("/home/pang/debug.txt");
//
//
    /*
     * Test qspline.evalOmega
     *
     * Passed!
     */

    std::map<int64_t,Eigen::Vector3d> gyroMap = eurocDataReader.getGyroMeasMap();
    LOG(INFO)<<"gyroMap size: "<<gyroMap.size();

    std::map<int64_t,Eigen::Vector3d>::iterator search;
    for(uint i = 1; i < data.size()-1; i++){

        ze::TrajectoryEle  p0 = data.at(i);
        int64_t ts = std::get<0>(p0);
        search = gyroMap.find(ts);

        if(search != gyroMap.end() && qspline.isTsEvaluable(ts*1e-9)){
            Eigen::Vector3d evalOmega = qspline.evalOmega(ts*1e-9);

            std::cout<<"Found!"<<std::endl;
            ofs_debug<< search->second.transpose()<<" "<< evalOmega.transpose()<<std::endl;

        }else{
            std::cout<<"Not found!"<<std::endl;
        }

    }
//
//    /*
//     *  Test getOmegaFromTwoQuaternion
//     *  Passed!
//     */
//
//
//    for(uint i = 1; i < data.size(); i++){
//        std::pair<double, Quaternion> q1 = getSample(data,i);
//        std::pair<double, Quaternion> q0 = getSample(data,i-1);
//        double dt = q1.first - q0.first;
//
//        Eigen::Vector3d omega = getOmegaFromTwoQuaternion(q0.second,q1.second,dt);
//        //ofs_debug << omega.transpose()<<std::endl;
//
//    }
//
//    /*
//     *  Test QSUtility::w_a
//     *  Passed!
//     */
//
//    for(uint i = 1; i < data.size()-1; i++){
//        std::pair<double, Quaternion> q1 = getSample(data,i);
//        std::pair<double, Quaternion> q0 = getSample(data,i-1);
//        std::pair<double, Quaternion> q2 = getSample(data,i+1);
//        double dt = q2.first - q0.first;
//
//        Quaternion dotQ0 = (q2.second - q0.second)/dt;
//
//        Eigen::Vector3d omega = QSUtility::w(q0.second,dotQ0);
//
//        //ofs_debug << omega.transpose()<<std::endl;
//
//    }
//
//
//    /*
//     * Test QuaternionSpline evalQuatSplineDerivate
//     * Passed!
//     */
//
//
//    for(uint i = 1; i < data.size()-1; i++){
//
//        std::pair<double, Quaternion> q1 = getSample(data,i);
//        if(qspline.isTsEvaluable(q1.first)){
//            std::pair<double, Quaternion> q0 = getSample(data,i-1);
//            std::pair<double, Quaternion> q2 = getSample(data,i+1);
//            double dt20 = q2.first - q0.first;
//            double dt10 = q1.first - q0.first;
//            double dt21 = q2.first - q1.first;
//
//            // num-diff 1-order
//            Quaternion num_dotQ1 = (q2.second - q0.second)/dt20;
//
//            Quaternion num_dotQ10 = (q1.second - q0.second)/dt10;
//            Quaternion num_dotQ21 = (q2.second - q1.second)/dt21;
//
//            // num-diff 2-order
//            Quaternion num_dot_dot_Q1 = (num_dotQ21 - num_dotQ10)/(dt20/2.0);
//
//
//            Quaternion Q1, dotQ1, dot_dotQ1;
//            qspline.evalQuatSplineDerivate(q1.first,Q1.data(),dotQ1.data(),dot_dotQ1.data());
//
//
//            if(qspline.isTsEvaluable(q2.first)){
//
//                Quaternion dotQ2 = qspline.evalDotQuatSpline(q2.first);
//                Quaternion another_num_dot_dot_Q1 = (dotQ2 - dotQ1)/(dt21); // a little delay
//
//
//
//                ofs_debug << q0.second.transpose()<<" "<<Q1.transpose()<<" ";
//                ofs_debug << num_dotQ1.transpose()<<" "<<dotQ1.transpose()<<" ";
//                ofs_debug << num_dot_dot_Q1.transpose()<<" "<<another_num_dot_dot_Q1.transpose()<<" "<<dot_dotQ1.transpose()<<std::endl;
//
//
////                std::cout << q0.second.transpose()<<" "<<Q1.transpose()<<" ";
////                std::cout << num_dotQ1.transpose()<<" "<<dotQ1.transpose()<<" ";
////                std::cout << num_dot_dot_Q1.transpose()<<" "<<dot_dotQ1.transpose()<<std::endl;
//
//            }
//
//
//
//
//            //std::cout << num_dotQ0.transpose()<<" "<<dotQ0.transpose()<<std::endl;
//
//        }
//    }
//
//
//
//
//
//
    ofs_debug.close();
//

    return 0;
}
