#include "cv/FeatureTracker.hpp"
#include <iostream>
#include <vector>

#include <fstream>
/*
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}

void LoadIMUData(const string &strIMUDataFilePath, std::vector<ORB_SLAM2::IMUData>& vimuData)
{

    ifstream fIMUdata;
    fIMUdata.open(strIMUDataFilePath.c_str());



    if (!fIMUdata.good()) {
        std::cout<<"Can not read: "<<strIMUDataFilePath<<std::endl;

        return ;
    }

    std::string line;
    int cnt = 0;
    while (std::getline(fIMUdata, line))
    {
        std::stringstream  ss;
        ORB_SLAM2::IMUData Imu;
        std::stringstream stream(line);
        std::string s;
        std::getline(stream, s, ',');
        ss<<s;
        double t;
        ss>>t;

        Imu._t = t/1e9;

        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            std::stringstream  ssg;
            ssg<<s;
            double g_ele;
            ssg>>g_ele;
            Imu._g[j] = g_ele;

        }
        //std::cout<< Imu._g.transpose()<<std::endl;
        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            std::stringstream  ssa;
            ssa<<s;
            double a_ele;
            ssa>> a_ele;
            Imu._a[j] = a_ele;

        }
        //std::cout<< Imu._a.transpose()<<std::endl;
        vimuData.push_back(Imu);

    }
}
 */

