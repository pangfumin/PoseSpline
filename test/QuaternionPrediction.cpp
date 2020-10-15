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


void loadCameraPose(const std::string &strFile, std::vector<Eigen::Matrix4d> &poses)
{
    std::ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    std::string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        std::string s;
        getline(f,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double aax,aay,aaz, tx, ty,tz;

            ss >> aax >> aay >> aaz >> tx >> ty >> tz;

            double angle = Eigen::Vector3d(aax, aay, aaz).norm();


            Eigen::Quaterniond q(Eigen::AngleAxisd(angle, Eigen::Vector3d(aax, aay, aaz).normalized()));

            Eigen::Matrix4d pose;
            pose.topLeftCorner(3,3) = q.toRotationMatrix();
            pose.topRightCorner(3,1) = Eigen::Vector3d(tx, ty, tz);

            poses.push_back(pose);

        }
    }
}


int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    std::string pose_file = "/home/pang/camera_extrinsic.txt";
    std::vector<Eigen::Matrix4d> poses;

    loadCameraPose(pose_file, poses);
    std::ofstream ofs_debug("/home/pang/debug.txt");

    std::cout << "load pose: " << poses.size() << std::endl;


    QuaternionSpline qspline(0.2);
    std::vector<std::pair<double,Quaternion>> samples, queryMeas;

    for (int i = 0; i < poses.size(); i++) {
        Eigen::Matrix4d pose = poses.at(i);
        Eigen::Matrix3d R = pose.topLeftCorner(3,3);
        Eigen::Quaterniond q(R);
        Eigen::AngleAxisd aa(R);
        auto euler = R.eulerAngles(0, 1, 2);



        Quaternion QuatJPL = rotMatToQuat(R);
        std::pair<double,Quaternion> sampleJPL = std::make_pair(i * 0.033, QuatJPL);
        queryMeas.push_back(sampleJPL);

        samples.push_back(sampleJPL);



    }




    qspline.initialQuaternionSpline(samples);

    for(auto i: queryMeas){

        if(qspline.isTsEvaluable(i.first)){
            Quaternion query = qspline.evalQuatSpline(i.first);

            Eigen::Vector3d diff = (quatLeftComp(i.second)*quatInv(query)).head(3);
//            CHECK_EQ(diff.norm() < 0.01,true)<<"Qspline query is not close to the ground truth!"
//                                             <<"Gt:    "<<i.second.transpose()<<std::endl
//                                             <<"Query: "<<query.transpose()<<std::endl
//                                             <<"diff:  "<<diff.transpose()<<std::endl<<std::endl;

            std::cout <<"Gt:    "<<i.second.transpose()<<std::endl;
            std::cout <<"Query: "<<query.transpose()<<std::endl;

            Eigen::Matrix3d R0 = quatToRotMat(i.second);
            Eigen::Matrix3d R1 = quatToRotMat(query);

            Eigen::AngleAxisd aa0(R0);
            Eigen::AngleAxisd aa1(R1);

            ofs_debug << aa0.axis()[0] << " "  << aa0.axis()[1] << " " << aa0.axis()[2]<< " " << aa1.axis()[0] << " "  << aa1.axis()[1] << " " << aa1.axis()[2]<< std::endl;

        }

    }

    ofs_debug.close();



    return 0;
}
