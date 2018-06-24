
#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <sys/stat.h>
#include <cstdlib> //atof

#include <deque>
#include <fstream>
#include <iostream>
#include <iomanip>      // std::setprecision
#include <limits>

#include "Eigen/Core"
#include <Eigen/Geometry>



template <typename  T>
inline T interpolate(double t0, double t1, double interp_t,T& v0, T& v1){

    T interp_v;
    double r = (interp_t - t0)/(t1-t0);
    interp_v = v0 + r*(v1 - v0);

    return interp_v;

}

inline Eigen::Matrix3d interpolateSO3 (double t0, double t1, double interp_t, Eigen::Matrix3d& v0, Eigen::Matrix3d& v1){

    double r = (interp_t - t0)/(t1-t0);

    Eigen::Quaterniond q0(v0);
    Eigen::Quaterniond q1(v1);
    Eigen::Quaterniond interp_q = q0.slerp(r,q1);
    Eigen::Matrix3d interp_R(interp_q);

    return interp_R;
}

inline Eigen::Quaterniond interpolateSO3 (double t0, double t1, double interp_t, Eigen::Quaterniond& v0, Eigen::Quaterniond& v1){

    double r = (interp_t - t0)/(t1-t0);

    Eigen::Quaterniond q0(v0);
    Eigen::Quaterniond q1(v1);
    Eigen::Quaterniond interp_q = q0.slerp(r,q1);

    return interp_q;
}

typedef   double  TimeStamp;

template <typename  T>
bool timeNear(T ta, T tb, T gap){
    return std::abs(ta - tb) < gap;
}
template<class MEASUREMENT_T>
struct Measurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TimeStamp timeStamp;      ///< Measurement timestamp
    MEASUREMENT_T measurement;  ///< Actual measurement.
    int sensorId = -1;          ///< Sensor ID. E.g. camera index in a multicamera setup

    /// \brief Default constructor.
    Measurement()
            : timeStamp(0.0) {
    }
    /**
     * @brief Constructor
     * @param timeStamp_ Measurement timestamp.
     * @param measurement_ Actual measurement.
     * @param sensorId Sensor ID (optional).
     */
    Measurement(const TimeStamp& timeStamp_, const MEASUREMENT_T& measurement_,
                int sensorId = -1)
            : timeStamp(timeStamp_),
              measurement(measurement_),
              sensorId(sensorId) {
    }

    Measurement(const Measurement& rhs):timeStamp(rhs.timeStamp), measurement(rhs.measurement), sensorId(rhs.sensorId)
    {

    }
    Measurement& operator=(const Measurement& rhs)
    {
        timeStamp = rhs.timeStamp;
        measurement = rhs.measurement;
        sensorId = rhs.sensorId;
        return *this;
    }
};

struct ImuReading{
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};
typedef Measurement<ImuReading> ImuMeasurement;
typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement> > ImuMeasurementDeque;



/// \brief Camera measurement.
struct CameraData {
    CameraData(){};
    std::string fileName;

    bool deliversKeypoints; ///< Are the keypoints delivered too?
    std::string idInSource; /// (0 based) id of the frame within the video or the image folder
};
/// \brief Camera info.
struct CameraInfo {
    std::string fileName;
};


typedef Measurement<CameraData> CameraMeasurement;
typedef std::deque<CameraMeasurement,Eigen::aligned_allocator<CameraMeasurement> > CameraMeasurementDeque;




// Groudtruth from euroc
struct GroudTruth {
    Eigen::Vector3d t_WS;
    Eigen::Quaterniond q_WS;
    Eigen::Vector3d v_WS;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
};

typedef Measurement<GroudTruth> GroudTruthMeasurement;
typedef std::deque<GroudTruthMeasurement,Eigen::aligned_allocator<GroudTruthMeasurement> >
        GroudTruthMeasurementDeque;

class EurocDatasetReader{
public:
    EurocDatasetReader(){};
    EurocDatasetReader(std::string& datasetPath):
            datasetPath_(datasetPath){
    }
    void setDataPath(std::string path){
        datasetPath_ = path;
    }

    bool readData(){

//        if(!loadFisheyeList()){
//            std::cout<<"Can NOT load Fisheye data!"<<std::endl;
//            return false;
//        }

        if(!loadIMUReadings()){
            std::cout<<"Can NOT load synIMU data!"<<std::endl;
            return false;
        }


        if(!loadGroudTruth()){
            std::cout<<"Can NOT load groudtrth data!"<<std::endl;
            return false;
        }


    }

    bool loadFisheyeList(){
        std::string fisheyeListPath = datasetPath_ + "/fisheye_timestamps.txt";

        std::ifstream ifs(fisheyeListPath);
        if(!ifs.is_open()){
            std::cerr<< "Failed to open fisheye list file: " << fisheyeListPath<<std::endl;
            return false;
        }

        double lastTimestamp = -1;
        std::string oneLine;
        CameraMeasurement camMeas;
        while(!ifs.eof()){
            std::getline(ifs, oneLine);

            std::stringstream stream(oneLine);
            std::string s;
            std::getline(stream, s, ' ');
            if(s.empty()) break;
            camMeas.measurement.fileName =datasetPath_ + "/" + s;

            std::getline(stream, s, ' ');

            double t1 = atof(s.c_str())*1e-6;

            if(std::abs(t1 - lastTimestamp) < 1e-3 || lastTimestamp > t1 ){
                continue;
            }

            lastTimestamp = t1;
            //std::cout<<t1<<std::endl;
            camMeas.timeStamp = TimeStamp(t1);

            CameraMeasDeque_.push_back(camMeas);

        }
        ifs.close();
        std::cout<<"load fisheye data: "<<CameraMeasDeque_.size()<<std::endl;
        return true;
    }

    bool loadIMUReadings() {
        std::string imuListPath = datasetPath_ + "/mav0/imu0/data.txt";

        std::ifstream ifs(imuListPath);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open imu_raw  file: " << imuListPath << std::endl;
            return false;
        }
        std::string oneLine;
        ImuMeasurement imuMeas;

        synImuMeasDeque_.clear();
        double lastTimestamp = -1.0;
        while (!ifs.eof()) {
            std::getline(ifs, oneLine);

            std::stringstream stream(oneLine);
            std::string s;
            std::getline(stream, s, ',');
            //std::cout<<"t: "<<s<<std::endl;
            if (s.empty()) break;
            double t1 = atof(s.c_str());
            t1 = t1 * 1e-9;  // [ns] to [s]

            if (synImuMeasDeque_.size() == 0) {
                lastTimestamp = t1;

            } else if (t1 <= lastTimestamp) {
                std::cerr << " IMU Unreason timestamp!"
                          << " " << t1 - lastTimestamp << std::endl;
                continue;

            }

            lastTimestamp = t1;
            imuMeas.timeStamp = TimeStamp(t1);

            Eigen::Vector3d accel;
            Eigen::Vector3d gyro;

            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                //std::cout<<s<<std::endl;
                gyro[j] = atof(s.c_str());
            }
            imuMeas.measurement.gyro = gyro;


            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                //std::cout<<s<<std::endl;
                accel[j] = atof(s.c_str());
            }
            imuMeas.measurement.accel = accel;

            synImuMeasDeque_.push_back(imuMeas);
        }

        ifs.close();

        std::cout<<"load imu data: "<<synImuMeasDeque_.size()<<std::endl;
        return true;
    }


    bool loadGroudTruth() {
        std::string groudtruth_fale = datasetPath_ + "/mav0/state_groundtruth_estimate0/data.txt";

        std::ifstream ifs(groudtruth_fale);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open groudtruth  file: " << groudtruth_fale << std::endl;
            return false;
        }
        std::string oneLine;
        GroudTruthMeasurement gtMeas;

        groudTruthMeasurementDeque_.clear();
        double lastTimestamp = -1.0;
        while (!ifs.eof()) {
            std::getline(ifs, oneLine);

            std::stringstream stream(oneLine);
            std::string s;
            std::getline(stream, s, ',');
            //std::cout<<"t: "<<s<<std::endl;
            if (s.empty()) break;
            double t1 = atof(s.c_str());
            t1 = t1 * 1e-9;  // [ns] to [s]

            if (groudTruthMeasurementDeque_.size() == 0) {
                lastTimestamp = t1;

            } else if (t1 <= lastTimestamp) {
                std::cerr << " IMU Unreason timestamp!"
                          << " " << t1 - lastTimestamp << std::endl;
                continue;

            }

            lastTimestamp = t1;
            gtMeas.timeStamp = TimeStamp(t1);

            Eigen::Vector3d t_WS;
            Eigen::Vector4d q_WS;
            Eigen::Vector3d v_WS;
            Eigen::Vector3d bg;
            Eigen::Vector3d ba;

            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');

                t_WS[j] = atof(s.c_str());
            }
            gtMeas.measurement.t_WS = t_WS;

            for (int j = 0; j < 4; ++j) {
                std::getline(stream, s, ',');
//                std::cout<<s<<" ";
                q_WS[j] = atof(s.c_str());
            }
//            std::cout<<std::endl;

            gtMeas.measurement.q_WS = Eigen::Quaterniond(q_WS(0), q_WS(1), q_WS(2), q_WS(3));


            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                //std::cout<<s<<std::endl;
                v_WS[j] = atof(s.c_str());
            }
            gtMeas.measurement.v_WS = v_WS;

            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                //std::cout<<s<<std::endl;
                bg[j] = atof(s.c_str());
            }
            gtMeas.measurement.bg = bg;

            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                //std::cout<<s<<std::endl;
                ba[j] = atof(s.c_str());
            }
            gtMeas.measurement.ba = ba;

            groudTruthMeasurementDeque_.push_back(gtMeas);
        }

        ifs.close();

        std::cout<<"load groudtrth: "<<groudTruthMeasurementDeque_.size()<<std::endl;
        return true;
    }




    // get time aligned data
    inline CameraMeasurementDeque getCameraMeasDeque(){
        return CropCameraMeasDeque_;
    }

    inline CameraMeasurementDeque getCameraMeasSubDeque(unsigned int start, unsigned int end){
        if (start < 0 || end >= CropCameraMeasDeque_.size()) {
            std::cerr<<"getCameraMeasSubDeque out of range"<<std::endl;
        }
        CameraMeasurementDeque::iterator startItr, endItr;
        startItr = endItr = CropCameraMeasDeque_.begin();
        std::advance(startItr, start);
        std::advance(endItr, end);
        CameraMeasurementDeque sub(startItr,endItr);
        return sub;
    }
    inline ImuMeasurementDeque getImuMeasDeque(){
        return  interp_synImuMeasDeque_;

    }


private:
    std::string datasetPath_;
    CameraMeasurementDeque CameraMeasDeque_, CropCameraMeasDeque_ ;
    ImuMeasurementDeque synImuMeasDeque_,interp_synImuMeasDeque_;
    GroudTruthMeasurementDeque groudTruthMeasurementDeque_;

    TimeStamp lastFisheyeTimestamp_;
    static double getMax(std::vector<double> times){

        double t = -1;
        for (auto i : times)
            if (t < i) t = i;

        return t;
    }

    static double getMin(std::vector<double> times){

        double t = std::numeric_limits<double>::max();
        for (auto i : times)
            if (t > i) t = i;

        return t;
    }

};




#endif

