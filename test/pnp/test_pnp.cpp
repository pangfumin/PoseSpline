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

    void simulate(Eigen::Matrix4d T_WC, int count, std::vector<Eigen::Vector3d>& pt3d, std::vector<Eigen::Vector2d>& pt2d, std::vector<Eigen::Vector3d>& pt3d_bearing) {
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
            pt3d_bearing.push_back(norm_ray);
            norm_ray.normalize();



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

void T2double(Eigen::Matrix4d& T,double* ptr){

    Eigen::Vector3d trans = T.topRightCorner(3,1);
    Eigen::Matrix3d R = T.topLeftCorner(3,3);
    Eigen::Quaterniond q(R);

    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);
    ptr[3] = q.x();
    ptr[4] = q.y();
    ptr[5] = q.z();
    ptr[6] = q.w();
}

void applyNoise(const Eigen::Matrix4d& Tin,Eigen::Matrix4d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.25*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.10*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;
    delat_quat.normalize();

    Tout.topRightCorner(3,1) = Tin.topRightCorner(3,1) + delat_trans;
    Tout.topLeftCorner(3,3) = Tin.topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}



typedef Eigen::Matrix<double,7,1> Vec7d;

Vec7d T2param(Eigen::Matrix4d T) {
    Eigen::Matrix3d R = T.topLeftCorner(3,3);
    Eigen::Quaterniond q(R);
    Vec7d param;
    param << T.topRightCorner(3,1), q.x(), q.y(), q.z(), q.w();
    return param;
}

Eigen::Matrix4d param2T(Vec7d vec) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topRightCorner(3,1) = vec.head(3);
    Eigen::Quaterniond q(vec[6], vec[3], vec[4], vec[5]);
    T.topLeftCorner(3,3) = q.toRotationMatrix();
    return T;
}


void pnp(Eigen::Matrix4d& T_WC, std::vector<Eigen::Vector3d>& pt3d, std::vector<Eigen::Vector3d>& pt3d_bearing) {
    ceres::Problem problem;
    PoseLocalParameterization *poseLocalParameter = new PoseLocalParameterization;
    std::cout << "before OPT : \n" << T_WC << std::endl;


    Vec7d  param = T2param(T_WC);

    problem.AddParameterBlock(param.data(), 7,poseLocalParameter);
    for (int i = 0; i < pt3d.size(); i ++) {
        ceres::CostFunction* e = new ProjectError(pt3d_bearing[i], pt3d[i]);

        problem.AddResidualBlock(e,NULL, param.data());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 30;
    options.max_num_iterations = 300;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.parameter_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    T_WC = param2T(param);
    std::cout << "OPT : \n" << T_WC << std::endl;

}


int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    Eigen::Matrix4d T_WC = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> pt3d, pt3d_bearing;
    std::vector<Eigen::Vector2d> pt2d;

    int count = 20;
    Simulation simulate(640, 480, 200);
    simulate.simulate(T_WC, count, pt3d, pt2d, pt3d_bearing );

    std::cout << "3d - 2d: " << pt3d.size() << " " << pt2d.size()  << " " << pt3d_bearing.size() << std::endl;

//    for (int i = 0; i < count; i ++) {
//
//        Eigen::Vector2d reproject = simulate.project(pt3d[i]);
//
////        std::cout << "reproject: " << reproject.transpose() << std::endl;
////        std::cout << "gt       : " << pt2d[i].transpose() << std::endl;
//
//    }


    Eigen::Matrix4d noised_T_WC;
    applyNoise(T_WC, noised_T_WC);
    pnp( noised_T_WC, pt3d, pt3d_bearing);







    return 0;
}
