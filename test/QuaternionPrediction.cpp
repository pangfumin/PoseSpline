#include <iostream>
#include <fstream>

#include "common/csv_trajectory.hpp"
#include "pose-spline/QuaternionSpline.hpp"
#include "pose-spline/VectorSpaceSpline.hpp"
#include <pose-spline/QuaternionSplineUtility.hpp>


using namespace ze;

std::pair<double,Quaternion>  getSample(ze::TupleVector& data, unsigned int i){
    ze::TrajectoryEle  p0 = data.at(i);
    Quaternion q = std::get<2>(p0);
    return std::make_pair(std::get<0>(p0)*1e-9,q);
};


void loadCameraPose(const std::string &strFile, std::vector<Eigen::Matrix4d> &poses, std::vector<std::string>& ids, std::vector<Eigen::Vector3d> &aas)
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
            std::string temp, index;

            ss >> temp >> index >> aax >> aay >> aaz >> tx >> ty >> tz;

            double angle = Eigen::Vector3d(aax, aay, aaz).norm();


            Eigen::AngleAxisd aa = Eigen::AngleAxisd(angle, Eigen::Vector3d(aax, aay, aaz).normalized());
            aas.push_back(Eigen::Vector3d(aax,aay,aaz));
            Eigen::Quaterniond q(aa);

            Eigen::Matrix4d pose;
            pose.topLeftCorner(3,3) = q.toRotationMatrix();
            pose.topRightCorner(3,1) = Eigen::Vector3d(tx, ty, tz);

            poses.push_back(pose);
            ids.push_back(index);

        }
    }
}


int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    std::string pose_file = "/home/pang/arc_campose.txt";
    std::vector<Eigen::Matrix4d> poses;
    std::vector<std::string> ids;
    std::vector<Eigen::Vector3d> aas;
    loadCameraPose(pose_file, poses,ids,aas);
    std::ofstream ofs_debug("/home/pang/debug.txt");

    std::cout << "load pose: " << poses.size() << " " << ids.size() << std::endl;


    double dt = 0.32;
    QuaternionSpline qspline(dt);
    std::vector<std::pair<double,Quaternion>> samples;

    VectorSpaceSpline vspline(dt);
    std::vector<std::pair<double,Eigen::Vector3d>> v_samples;


    std::vector<std::string> ids_sample;
    std::vector<Eigen::Vector3d> aas_sample;


    for (int i = 0; i < poses.size(); i++) {
        Eigen::Matrix4d pose = poses.at(i);
        Eigen::Matrix3d R = pose.topLeftCorner(3,3);
        Eigen::Vector3d t = pose.topRightCorner(3,1);
        Eigen::Quaterniond q(R);





        Quaternion QuatJPL = rotMatToQuat(R);
        std::pair<double,Quaternion> sampleJPL = std::make_pair(i * 0.033, QuatJPL);

        samples.push_back(sampleJPL);

        std::pair<double,Eigen::Vector3d> v_sample = std::make_pair(i * 0.033, t);

        v_samples.push_back(v_sample);

        ids_sample.push_back(ids[i]);

        aas_sample.push_back(aas[i]);



    }




    qspline.initialQuaternionSpline(samples);
    vspline.initialSpline(v_samples);

    for(int i = 0; i < samples.size(); i++){
        auto quat = samples[i];
        auto trans = v_samples[i];
        auto id = ids_sample[i];
        auto aa = aas_sample[i];
        if(qspline.isTsEvaluable(quat.first)){
            Quaternion q_query = qspline.evalQuatSpline(quat.first);
            Eigen::Vector3d t_query = vspline.evaluateSpline(quat.first);

//            Eigen::Vector3d diff = (quatLeftComp(quat.second)*quatInv(query)).head(3);
//            CHECK_EQ(diff.norm() < 0.01,true)<<"Qspline query is not close to the ground truth!"
//                                             <<"Gt:    "<<i.second.transpose()<<std::endl
//                                             <<"Query: "<<query.transpose()<<std::endl
//                                             <<"diff:  "<<diff.transpose()<<std::endl<<std::endl;

            std::cout <<"Gt:    "<<quat.second.transpose()<<std::endl;
            std::cout <<"Query: "<<q_query.transpose()<<std::endl;

            Eigen::Matrix3d R0 = quatToRotMat(quat.second);
            Eigen::Matrix3d R1 = quatToRotMat(q_query);

            Eigen::AngleAxisd aa0(R0);
            Eigen::AngleAxisd aa1(R1);

            Eigen::Vector3d t0 = trans.second;
            Eigen::Vector3d t1 = t_query;


            ofs_debug  << aa0.axis()[0] * aa0.angle() << " "  << aa0.axis()[1] * aa0.angle() << " " << aa0.axis()[2] * aa0.angle()
                    << " " <<  t0[0] << " " << t0[1] << " " << t0[2]
                    << " " << aa1.axis()[0] * aa0.angle() << " "  << aa1.axis()[1] * aa0.angle() << " " << aa1.axis()[2] * aa0.angle()
                    << " " <<  t1[0] << " " << t1[1] << " " << t1[2]
                    << " " <<  aa[0] << " " << aa[1] << " " << aa[2]
                    << std::endl;

        }

    }

    ofs_debug.close();



    return 0;
}
