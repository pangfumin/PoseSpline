#include "project_error.h"
#include <iostream>
#include "NumbDifferentiator.hpp"
#include "pose_local_parameterization.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int width = 544;
int height = 960;
double focal = 410;


void T2double(Eigen::Isometry3d& T,double* ptr){

    Eigen::Vector3d trans = T.matrix().topRightCorner(3,1);
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Quaterniond q(R);

    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);

}

void applyNoise(const Eigen::Isometry3d& Tin,Eigen::Isometry3d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.5*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.16*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;

    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3) = Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}


void pnp(Eigen::Vector3d& t, std::vector<Eigen::Vector2d>& pt2d, std::vector<Eigen::Vector3d>& pt3d) {
    ceres::Problem problem;
    auto initT = t;

    double cx = width / 2.0;
    double cy = height / 2.0;

    problem.AddParameterBlock(t.data(), 3);
    for (int i = 0; i < pt3d.size(); i ++) {
        ceres::CostFunction* e = new ProjectError(pt2d[i], pt3d[i], cx, cy, focal);

        problem.AddResidualBlock(e,NULL, t.data());
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

    std::cout << "before OPT : \n" << initT.transpose()<< std::endl;

    std::cout << "OPT : \n" << t.transpose() << std::endl;

}


cv::Mat visualize( std::vector<Eigen::Vector3d>& vec3d,
std::vector<Eigen::Vector2d>& vec2d, Eigen::Vector3d t) {
    cv::Mat image(height, width,  CV_8UC3);
    image.setTo(cv::Scalar(255,255,255));

    double cx = width / 2.0;
    double cy = height / 2.0;

    for (int i = 0; i < vec2d.size(); i ++) {
        cv::circle(image, cv::Point2f(vec2d[i].x(), vec2d[i].y()),3,cv::Scalar(0,1, 255),3 );

        Eigen::Vector2d uv;
        auto pt3d = vec3d[i] + t;
        uv << focal * pt3d[0]/ pt3d[2] + cx, focal * pt3d[1]/ pt3d[2] + cy;
//         vec2d.push_back(uv);

        cv::circle(image, cv::Point2f(uv.x(), uv.y()),3,cv::Scalar(255,0,1),3 );

    }

    return image;
}


int main(){

    // simulate

   Eigen::Vector3d C0p(1.2,-0.3, 2);


    double cx = width / 2.0;
    double cy = height / 2.0;



    Eigen::Vector2d uv(focal  * C0p(0)/C0p(2) + cx, focal * C0p(1)/C0p(2) + cy);




    /*
     * Zero Test
     * Passed!
     */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    ProjectError* projectFactor = new ProjectError(uv, C0p, cx, cy, focal);

    double* param_T_WC0 = new double[3];



    Eigen::Vector3d param_t = {0,0,0};

    double* paramters[1] = {param_t.data()};

    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian0_min;

    double* jacobians_min[1] = {jacobian0_min.data()};


    Eigen::Matrix<double,2,3,Eigen::RowMajor> jacobian0;
    double* jacobians[1] = {jacobian0.data()};

    projectFactor->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */
//
//
    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;

    Eigen::Vector3d param_t_noised = param_t + Eigen::Vector3d(0.001, 0.02, 0.001);


    double* parameters_noised[1] = {param_t_noised.data()};

    projectFactor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(),jacobians,jacobians_min);


    std::cout<<"residual: "<<residual.transpose()<<std::endl;

    Eigen::Matrix<double,2,3,Eigen::RowMajor> num_jacobian0_min;

    NumbDifferentiator<ProjectError,1> num_differ(projectFactor);

    num_differ.df_r_xi<2,3>(parameters_noised,0,num_jacobian0_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;



    std::vector<Eigen::Vector3d> vec3d;
    std::vector<Eigen::Vector2d> vec2d;


    vec3d.push_back(Eigen::Vector3d(3.6173e-05,-2.04261e-05,1));
    vec3d.push_back(Eigen::Vector3d(-0.129064,0.00135955,1.02077));
    vec3d.push_back(Eigen::Vector3d(0.129065,-0.00136625,0.97924));
    vec3d.push_back(Eigen::Vector3d(-0.130392,-0.411304,0.956982));
    vec3d.push_back(Eigen::Vector3d(-0.0136219,-0.471249,0.904506));
    vec3d.push_back(Eigen::Vector3d(0.114658,-0.414131,0.905035));
    vec3d.push_back(Eigen::Vector3d(-0.0059986,-0.531102,0.821417));
    vec3d.push_back(Eigen::Vector3d(-0.20701,-0.181552,1.07393));
    vec3d.push_back(Eigen::Vector3d(0.207106,-0.179638,0.983115));
    vec3d.push_back(Eigen::Vector3d(-0.228879,0.0652985,1.10983));
    vec3d.push_back(Eigen::Vector3d(0.189251,0.0424184,0.977305));

//     for (int i = 0; i < vec3d.size(); i++) {
//         Eigen::Vector2d uv;
//         auto pt3d = vec3d[i];
//         uv << focal * pt3d[0]/ pt3d[2] + cx, focal * pt3d[1]/ pt3d[2] + cy;
//         vec2d.push_back(uv);
//     }

    vec2d.push_back(Eigen::Vector2d(272,480));
    vec2d.push_back(Eigen::Vector2d(240.041,480));
    vec2d.push_back(Eigen::Vector2d(303.959,480));
    vec2d.push_back(Eigen::Vector2d(223.523,329.587));
    vec2d.push_back(Eigen::Vector2d(269.845,327.428));
    vec2d.push_back(Eigen::Vector2d(316.168,325.269));
    vec2d.push_back(Eigen::Vector2d(272.359,269.853));
    vec2d.push_back(Eigen::Vector2d(208.441,399.396));
    vec2d.push_back(Eigen::Vector2d(333.404,399.396));
    vec2d.push_back(Eigen::Vector2d(207.005,473.523));
    vec2d.push_back(Eigen::Vector2d(332.686,472.084));





    srand((unsigned)time(NULL));

    for (int  i = 0; i < 50; i++) {
        Eigen::Vector3d rand;
        rand.setRandom();

        rand *= 0.125;
//        rand.head<2>().setZero();
        Eigen::Vector3d est_t(0,0,0);

        cv::Mat before_image  = visualize(vec3d, vec2d, est_t);
        cv::imshow("before_image", before_image);
//    cv::waitKey();



        for (int j = 0; j < vec3d.size(); j++) {
            vec3d[j] = vec3d[j] + rand;
        }
        pnp(est_t, vec2d, vec3d);

        std::cout << "noised: " << rand.transpose() << std::endl;


        cv::Mat after_image  = visualize(vec3d, vec2d, est_t);
        cv::imshow("after_image", after_image);
        cv::waitKey(300);

    }



    return 0;
}
