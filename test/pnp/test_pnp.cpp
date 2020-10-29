#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <vector>
#include "project_error.h"
#include "pose_local_parameterization.h"

class Simulation {
public:
    Simulation(int width, int height, float focal):width_(width), height_(height), focal_(focal) {

    }

    void simulate(Eigen::Matrix4d T_WC, int count, std::vector<Eigen::Vector3d>& pt3d, std::vector<Eigen::Vector2d>& pt2d, std::vector<Eigen::Vector2d>& pt2d_bearing) {
        srand((unsigned int) time(0));
        double cx = width_ / 2;
        double cy = height_ / 2;
        for (int i = 0; i < count; i++) {

            Eigen::Vector3d rand = Eigen::Vector3d::Random();
//            std::cout << rand.transpose() <<std::endl;

            pt2d.push_back(Eigen::Vector2d(std::abs(rand.x() * width_), std::abs(rand.y() * height_)));
//            std::cout << "2d: " << pt2d.back().transpose() <<std::endl;


            // grenrate 3d

            Eigen::Vector2d ray;
            ray[0] = (pt2d.back()[0] - cx) / focal_;
            ray[1] = (pt2d.back()[1] - cy) / focal_;

            Eigen::Vector3d norm_ray(ray[0], ray[1],1.0);
            norm_ray.normalize();

            pt2d_bearing.push_back(ray);

            Eigen::Vector3d Cp = norm_ray * std::abs(rand[2]);

//            std::cout << "3d: " << Cp.transpose() <<std::endl;

            Eigen::Vector4d Wp = T_WC * (Eigen::Vector4d() << Cp,1.0 ).finished();

            pt3d.push_back(Wp.head<3>() / Wp(3));

        }

    }


    Eigen::Vector2d  project(Eigen::Vector3d& pt3d) {
        double cx = width_ / 2;
        double cy = height_ / 2;
        Eigen::Vector2d pt2d;
        pt2d << (pt3d(0)/ pt3d(2)) * focal_ + cx, (pt3d(1)/ pt3d(2)) * focal_ + cy;
        return pt2d;
    }
private:
    int width_;
    int height_;
    double focal_;

};


void pnp(Eigen::Matrix4d& T_WC, std::vector<Eigen::Vector3d>& pt3d, std::vector<Eigen::Vector2d>& pt2d_bearing) {
    ceres::Problem problem;
    PoseLocalParameterization *poseLocalParameter = new PoseLocalParameterization;

//    for (int pt3d)
}


int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    Eigen::Matrix4d T_WC = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> pt3d;
    std::vector<Eigen::Vector2d> pt2d, pt2d_bearing;

    int count = 20;
    Simulation simulate(640, 480, 200);
    simulate.simulate(T_WC, count, pt3d, pt2d, pt2d_bearing );

    std::cout << "3d - 2d: " << pt3d.size() << " " << pt2d.size()  << " " << pt2d_bearing.size() << std::endl;

//    for (int i = 0; i < count; i ++) {
//
//        Eigen::Vector2d reproject = simulate.project(pt3d[i]);
//
////        std::cout << "reproject: " << reproject.transpose() << std::endl;
////        std::cout << "gt       : " << pt2d[i].transpose() << std::endl;
//
//    }






    return 0;
}
