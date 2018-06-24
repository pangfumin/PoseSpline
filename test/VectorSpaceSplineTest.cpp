#include <iostream>
#include <fstream>

#include "common/csv_trajectory.hpp"
#include "splines/bspline.hpp"
#include "pose-spline/Quaternion.hpp"
#include "pose-spline/VectorSpaceSpline.hpp"
#include <pose-spline/QuaternionSplineUtility.hpp>
#include <utility/timer.h>


using namespace ze;

std::pair<double,Quaternion>  getSample(ze::TupleVector& data, unsigned int i){
    ze::TrajectoryEle  p0 = data.at(i);
    Quaternion q = std::get<2>(p0);
    return std::make_pair(std::get<0>(p0)*1e-9,q);
};


std::pair<double,Eigen::Vector3d>  getSamplePosition(ze::TupleVector& data, unsigned int i){
    ze::TrajectoryEle  p0 = data.at(i);
    Eigen::Vector3d p = std::get<1>(p0);
    return std::make_pair(std::get<0>(p0)*1e-9,p);
};



int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    std::string dataset = "/media/pang/Plus/dataset/MH_01_easy";

    ze::EurocResultSeries eurocDataReader;
    eurocDataReader.load(dataset + "/mav0/state_groundtruth_estimate0/data.csv");
    eurocDataReader.loadIMU(dataset + "/mav0/imu0/data.csv");

    ze::TupleVector  data = eurocDataReader.getVector();
    Buffer<real_t, 7>& poseBuffer = eurocDataReader.getBuffer();
    //LOG(INFO)<<"Get data size: "<<data.size(); // @200Hz
    std::vector<std::pair<int64_t ,Eigen::Vector3d>> linearVelocities = eurocDataReader.getLinearVelocities();
    std::vector<Vector3> gyroBias = eurocDataReader.getGyroBias();
    //LOG(INFO)<<"Get velocities size: "<<linearVelocities.size(); // @200Hz
    //LOG(INFO)<<"Get gyro_bias  size: "<<gyroBias.size(); // @200Hz

    std::vector<int64_t> imu_ts = eurocDataReader.getIMUts();
    //LOG(INFO)<<"Get IMU ts  size: "<<imu_ts.size(); // @200Hz

    std::vector<Vector3> gyroMeas = eurocDataReader.getGyroMeas();
    //LOG(INFO)<<"Get gyro Meas  size: "<<gyroMeas.size(); // @200Hz

    int start  = 1;
    int end = data.size() - 2;

    ze::VectorSpaceSpline vectorSpaceSpline(4,0.1);
    std::vector<std::pair<double,Eigen::Vector3d>> samples, queryMeas, queryVelocity;

    for(uint i = start; i <end; i++){

        queryMeas.push_back(getSamplePosition( data,  i));
        queryVelocity.push_back(std::make_pair(((double)linearVelocities.at(i).first)*1e-9,
                                               linearVelocities.at(i).second));
        if(i % 4  == 0){
            samples.push_back(getSamplePosition( data,  i));

        }
    }

    TimeStatistics::Timer timer;
    vectorSpaceSpline.initialSpline(samples);
    std::cout<<"intialization timer: "<< timer.stopAndGetSeconds()<<std::endl;


    /*
     *  Test:
     */

    for(auto i: queryMeas){
        if(vectorSpaceSpline.isTsEvaluable(i.first)){
            Eigen::Vector3d query = vectorSpaceSpline.evaluateSpline(i.first);
            Eigen::Vector3d diff = query - i.second;
            CHECK_EQ(diff.norm() < 0.01,true)<<" Position query is not close to the ground truth!"
                                              <<"Gt:    "<<i.second.transpose()<<std::endl
                                              <<"Query: "<<query.transpose()<<std::endl
                                              <<"diff:  "<<diff.transpose()<<std::endl<<std::endl;

        }

    }


//    int cnt  = 0;
//
//    for(auto i : queryVelocity){
//        if(cnt ++ < 100 ) continue;
//        if(vectorSpaceSpline.isTsEvaluable(i.first)){
//            Eigen::Vector3d query = vectorSpaceSpline.evaluateDotSpline(i.first);
//            Eigen::Vector3d queryNumeric = vectorSpaceSpline.evaluateDotSplineNumeric(i.first);
//            Eigen::Vector3d diff = query - i.second;
////            CHECK_EQ(diff.norm() < 0.01,true)<<" Position query is not close to the ground truth!"
////                                             <<"Gt:    "<<i.second.transpose()<<std::endl
////                                             <<"Query: "<<query.transpose()<<std::endl
////                                             <<"diff:  "<<diff.transpose()<<std::endl<<std::endl;
//            std::cout<<"meas         : "<< i.second.transpose()<<std::endl;
//            std::cout<<"queryNumeric : "<< queryNumeric.transpose()<<std::endl;
//            std::cout<<"query        : "<< query.transpose()<<std::endl;
//
//        }
//
//    }
//



    return 0;
}