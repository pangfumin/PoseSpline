#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "project_error.h"
#include "pose_local_parameterization.h"

class Simulation {
public:
    Simulation(int width, int height, double focal, double min_z, double max_z):
    width_(width), height_(height), focal_(focal),min_z_(min_z), max_z_(max_z)  {

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
            norm_ray  = norm_ray *(max_z_ - min_z_) + Eigen::Vector3d(1,1,1) *min_z_;


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

    cv::Mat visualize(Eigen::Matrix4d T_WC, std::vector<Eigen::Vector3d>& pt3d, std::vector<Eigen::Vector2d>& pt2d) {
        cv::Mat image(height_, width_,  CV_8UC3);
        image.setTo(cv::Scalar(255,255,255));
        double cx = width_ / 2;
        double cy = height_ / 2;

        for (int i = 0; i < pt3d.size(); i ++) {
            cv::circle(image, cv::Point2f(pt2d[i].x(), pt2d[i].y()),3,cv::Scalar(0,255,1),3 );


            Eigen::Matrix3d R_WC = T_WC.topLeftCorner(3,3);
            Eigen::Vector3d Cp = R_WC.transpose() * (pt3d[i] - T_WC.topRightCorner(3,1));

            cv::Point2f reproject;
            reproject.x = (Cp[0]/Cp(2)) * focal_ + cx;
            reproject.y = (Cp[1]/Cp(2)) * focal_ + cy;

            cv::circle(image, reproject,3,cv::Scalar(255,0,1),3 );

            cv::line(image, cv::Point2f(pt2d[i].x(), pt2d[i].y()), reproject, cv::Scalar(0,0,255), 2);

        }
        return image;
    }


private:
    int width_;
    int height_;
    double focal_;
    double min_z_, max_z_;

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

    Eigen::Vector3d delat_trans = 0.45*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.10*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;
    delat_quat.normalize();

    Tout.topRightCorner(3,1) = Tin.topRightCorner(3,1) + delat_trans;
}



typedef Eigen::Matrix<double,3,1> Vec3d;

Vec3d T2param(Eigen::Matrix4d T) {
    Eigen::Matrix3d R = T.topLeftCorner(3,3);
    Vec3d param;
    param << T.topRightCorner(3,1);
    return param;
}

Eigen::Matrix4d param2T(Vec3d vec, Eigen::Quaterniond q) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topRightCorner(3,1) = vec;
    T.topLeftCorner(3,3) = q.toRotationMatrix();
    return T;
}


void pnp(Eigen::Matrix4d& T_WC, std::vector<Eigen::Vector3d>& pt3d, std::vector<Eigen::Vector3d>& pt3d_bearing, Eigen::Quaterniond q) {
    ceres::Problem problem;
    auto initT = T_WC;

    Vec3d  param = T2param(T_WC);

    problem.AddParameterBlock(param.data(), 3);
    for (int i = 0; i < pt3d.size(); i ++) {
        ceres::CostFunction* e = new ProjectError(pt3d_bearing[i], pt3d[i], q);

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

    T_WC = param2T(param, q);
    std::cout << "before OPT : \n" << initT << std::endl;

    std::cout << "OPT : \n" << T_WC << std::endl;

}


int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    Eigen::Matrix4d T_WC = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> pt3d, pt3d_bearing;
    std::vector<Eigen::Vector2d> pt2d;

    int count = 20;
    Simulation simulate(640, 480, 200, 0.1, 7);
    simulate.simulate(T_WC, count, pt3d, pt2d, pt3d_bearing );

    std::cout << "3d - 2d: " << pt3d.size() << " " << pt2d.size()  << " " << pt3d_bearing.size() << std::endl;

//    for (int i = 0; i < count; i ++) {
//        Eigen::Vector2d reproject = simulate.project(pt3d[i]);
//        std::cout << "reproject: " << reproject.transpose() << std::endl;
//        std::cout << "gt       : " << pt2d[i].transpose() << std::endl;
//    }


    Eigen::Matrix4d noised_T_WC;



    Eigen::Matrix3d R = T_WC.topLeftCorner(3,3);
    Eigen::Quaterniond q(R);
    applyNoise(T_WC, noised_T_WC);

    cv::Mat before_image = simulate.visualize(noised_T_WC, pt3d, pt2d);
    cv::imshow("before_image", before_image);


    // pnp
    pnp( noised_T_WC, pt3d, pt3d_bearing, q);


    cv::Mat after_image = simulate.visualize(noised_T_WC, pt3d, pt2d);
    cv::imshow("after_image", after_image);
    cv::waitKey();

    return 0;
}
