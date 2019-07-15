#include <iostream>
#include <fstream>

#include "common/csv_trajectory.hpp"
#include "PoseSpline/VectorSpaceSpline.hpp"
#include <PoseSpline/QuaternionSplineUtility.hpp>
#include <okvis_util/timer.h>


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
    std::string dataset = "/home/pang/software/PoseSpline/data/MH_01_easy";
    ze::EurocResultSeries eurocDataReader;
    eurocDataReader.load(dataset + "/state_groundtruth_estimate0/data.csv");
    eurocDataReader.loadIMU(dataset+ "/imu0/data.csv");

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

    VectorSpaceSpline vectorSpaceSpline(0.1);
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
    std::ofstream ofs_debug("/home/pang/debug.txt");


    for(auto i: queryMeas){
        if(vectorSpaceSpline.isTsEvaluable(i.first)){
            Eigen::Vector3d query = vectorSpaceSpline.evaluateSpline(i.first);
            Eigen::Vector3d diff = query - i.second;
            CHECK_EQ(diff.norm() < 0.01,true)<<" Position query is not close to the ground truth!"
                                              <<"Gt:    "<<i.second.transpose()<<std::endl
                                              <<"Query: "<<query.transpose()<<std::endl
                                              <<"diff:  "<<diff.transpose()<<std::endl<<std::endl;

            ofs_debug<< query.transpose()<<" "<< i.second.transpose()<<std::endl;
        }

    }


    ofs_debug.close();

    return 0;
}