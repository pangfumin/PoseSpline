#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "project_error.h"
#include "pose_local_parameterization.h"

#include <algorithm>    // std::max



int width = 544;
int height = 960;
double focal = 516;

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
            pt2d.push_back(Eigen::Vector2d(std::abs(rand.x() * width_), std::abs(rand.y() * height_)));

            // grenrate 3d
            Eigen::Vector2d ray;
            ray[0] = (pt2d.back()[0] - cx) / focal_;
            ray[1] = (pt2d.back()[1] - cy) / focal_;

            Eigen::Vector3d norm_ray(ray[0], ray[1],1.0);
            pt3d_bearing.push_back(norm_ray);
            norm_ray.normalize();
            norm_ray  = norm_ray *(max_z_ - min_z_) + Eigen::Vector3d(1,1,1) *min_z_;


            Eigen::Vector3d Cp = norm_ray * std::abs(rand[2]);
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

Eigen::Matrix4d pnp(Eigen::Matrix4d T_WC,
        std::vector<Eigen::Vector2d>& pt2ds, std::vector<Eigen::Vector3d>& pt3ds) {
    std::vector<Eigen::Vector3d> used_pt3d_bearing, used_pt3d;



    double cx = width / 2;
    double cy = height / 2;

    ceres::Problem problem;
    auto initT = T_WC;
    Vec3d  param = T2param(T_WC);
    problem.AddParameterBlock(param.data(), 3);

    Eigen::Quaterniond q(1.0,0,0,0);
    for (int i = 0; i < pt3ds.size(); i++) {
        auto pt2d = pt2ds[i];
        Eigen::Vector3d bearing((pt2d.x() - cx)/ focal, (pt2d.y() - cy)/ focal, 1.0);
        auto pt3d = pt3ds[i];
        ceres::CostFunction* e = new ProjectError(bearing, pt3d, q);
        problem.AddResidualBlock(e,NULL, param.data());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 3;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.parameter_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;


    Eigen::Matrix4d res;
    res = param2T(param, q);
    std::cout << "before OPT : \n" << initT << std::endl;

    std::cout << "after  OPT : \n" << res << std::endl;

    return res;

}

std::vector<std::string> StrSplit(const std::string &str, char delimiter) {
    std::vector<std::string> splits;
    size_t curr = 0;
    size_t next = str.find(delimiter, curr);
    while (next != std::string::npos) {
        splits.push_back(str.substr(curr, next - curr));
        curr = next + 1;
        next = str.find(delimiter, curr);
    }
    splits.push_back(str.substr(curr));
    return std::move(splits);
}

void loadData(const std::string data_file, std::vector<std::vector<Eigen::Vector2d>>& pt2ds, std::vector<std::vector<Eigen::Vector3d>>& pt3ds) {
    std::ifstream ifs(data_file);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open data list file: " << data_file
                  << std::endl;
        return ;
    }

    std::cout << "list_file: " << data_file << std::endl;

    bool first_msg = true;

    std::string one_line;
    int imu_seq = 0;

    int CNT_2D = 19;
    int CNT_3D = 21;

    std::string line;
    int SKIP = 2;
//    ; // get rid of the header
//    std::cout << line << std::endl;
    while (std::getline(ifs, line)) {
//        std::cout << line << std::endl;
        if (line.front() != '#') {
            std::vector<std::string> content = StrSplit(line, ' ');
//            std::cout << "content: " << content.size()  << " " << content[0] << std::endl;
            std::vector<Eigen::Vector3d> pt3d;
            std::vector<Eigen::Vector2d> pt2d;

            for (int i = 0; i < CNT_2D ; i++) {

                double x = std::stod(content[SKIP + i*2]);
                double y = std::stod(content[SKIP + i*2 + 1]);
                pt2d.push_back(Eigen::Vector2d(x,y));

            }

            for (int i = 0; i < CNT_3D ; i++) {

                double x = std::stod(content[SKIP + CNT_2D*2 + i*3]);
                double y = std::stod(content[SKIP + CNT_2D*2 + i*3 + 1]);
                double z = std::stod(content[SKIP + CNT_2D*2 + i*3 + 2]);
                pt3d.push_back(Eigen::Vector3d(x,y,z));

            }

            pt2ds.push_back(pt2d);
            pt3ds.push_back(pt3d);

        }
    }
}

void visualize(cv::Mat& image, std::vector<Eigen::Vector2d> pt2ds, std::vector<Eigen::Vector3d> pt3ds, Eigen::Matrix4d T_WC,std::vector<int> index_2d, std::map<int, int> corresponding_pair) {

    for (int i = 0; i < corresponding_pair.size(); i++) {
        cv::Point2f pt(  pt2ds.at(index_2d[i]).x(), pt2ds.at(index_2d[i]).y());

        std::string text = std::to_string(index_2d[i]);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.51;
        int thickness = 2;
        cv::putText(image, text, pt, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);
        cv::circle(image, pt,3,cv::Scalar(255,0,1),3 );

    }

    int start = 0;
    int end = 1;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);


    start = 18;
    end = 1;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);

    start = 8;
    end = 18;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);
    start = 11;
    end = 18;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);

    start = 1;
    end = 2;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);
    start = 1;
    end = 5;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);


    start = 2;
    end = 3;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);
    start = 3;
    end = 4;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);


    start = 5;
    end = 6;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);
    start = 6;
    end = 7;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);



    Eigen::Vector3d t_WC = T_WC.topRightCorner(3,1);

    double cx = width / 2;
    double cy = height / 2;
    for (int i = 0; i < index_2d.size(); i++) {
        int id1 = corresponding_pair[index_2d[i]];
        auto pt3d = pt3ds[id1];
        //pt3d -= t_WC;

        Eigen::Vector2d pt2d;
        pt2d << (pt3d(0)/ pt3d(2)) * focal + cx, (pt3d(1)/ pt3d(2)) * focal + cy;
        cv::Point2f pt( pt2d.x(), pt2d.y());
        cv::circle(image, pt,3,cv::Scalar(10,255,1),3 );

        std::string text = std::to_string(id1);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.51;
        int thickness = 2;
        cv::putText(image, text, pt, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);

    }


}

std::vector<Eigen::Vector3d> normalize3d( std::vector<Eigen::Vector3d>& pt3ds) {

    double min_x = 10000, max_x = -10000;
    double min_y = 10000, max_y = -10000;
    double min_z = 10000, max_z = -10000;

    std::vector<Eigen::Vector3d> rescaled_shift_pt3ds;
    for (int i = 0; i < pt3ds.size(); i++) {


        auto pt3d = pt3ds[i];
        if (min_x > pt3d.x()) {
            min_x = pt3d.x();
        }
        if (max_x < pt3d.x()) {
            max_x = pt3d.x();
        }

        if (min_y > pt3d.y()) {
            min_y = pt3d.y();
        }
        if (max_y < pt3d.y()) {
            max_y = pt3d.y();
        }

        if (min_z > pt3d.z()) {
            min_z = pt3d.z();
        }
        if (max_z < pt3d.z()) {
            max_z = pt3d.z();
        }

        rescaled_shift_pt3ds.push_back(pt3d);


    }

    double scale = std::max(max_x - min_x, std::max(max_y - min_y, max_z - min_z));

//    std::cout << "max_x: " << max_x << " " << min_x << " " << max_y << " " << min_y << " " << max_z << " " << min_z  << " " << scale<< std::endl;

    for (int i = 0; i < rescaled_shift_pt3ds.size(); i++) {
        rescaled_shift_pt3ds[i] = rescaled_shift_pt3ds[i] * 1.0/ scale + Eigen::Vector3d(0,0,1);

    }

    return rescaled_shift_pt3ds;



}







int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    Eigen::Matrix4d T_WC = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> pt3d, pt3d_bearing;
    std::vector<Eigen::Vector2d> pt2d;
    std::vector<std::vector<Eigen::Vector2d>> pt2ds;
    std::vector<std::vector<Eigen::Vector3d>> pt3ds;

    std::string data_file = "/home/pang/Downloads/handHold_2d3d_keypoints.txt";
    loadData(data_file, pt2ds, pt3ds);

    std::cout << "data: " << pt2ds.size() << " " << pt3ds.size() << std::endl;

    std::map<int, int> corresponding_pair;
    std::vector<int> index_2d;
    //腰
    corresponding_pair.insert({18,0});
    corresponding_pair.insert({8,5});
    corresponding_pair.insert({11,1});
    index_2d.push_back(18);
    index_2d.push_back(8);
    index_2d.push_back(11);

    // 肩膀
    corresponding_pair.insert({2,13});
    corresponding_pair.insert({1,10});
    corresponding_pair.insert({5,17});
    index_2d.push_back(2);
    index_2d.push_back(1);
    index_2d.push_back(5);


    // 头
    corresponding_pair.insert({0,11});
    index_2d.push_back(0);




    // 胳膊
    corresponding_pair.insert({3,14});
    corresponding_pair.insert({6,18});
    corresponding_pair.insert({4,15});
    corresponding_pair.insert({7,19});

    index_2d.push_back(3);
    index_2d.push_back(6);
    index_2d.push_back(4);
    index_2d.push_back(7);




    for (int i = 0; i < pt2ds.size(); i++) {

        Eigen::Matrix4d T_WC = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d res_TWC = Eigen::Matrix4d::Identity();
//

        std::vector<Eigen::Vector3d> normalize_pt3d = normalize3d(pt3ds[i]);

        std::vector<Eigen::Vector2d> used_pt2d;
        std::vector<Eigen::Vector3d> used_pt3d;
        for (int j = 0; j < index_2d.size();j++) {
            used_pt2d.push_back(pt2ds[i][index_2d[j]]);
            int id1 = corresponding_pair[index_2d[j]];
            used_pt3d.push_back(pt3ds[i][id1]);
        }

        T_WC(2,3) = -0.3;
        res_TWC = pnp(T_WC,used_pt2d, used_pt3d);



        cv::Mat image(960, 544,  CV_8UC3);
        image.setTo(cv::Scalar(255,255,255));

        visualize(image, pt2ds[i], normalize_pt3d, res_TWC,index_2d, corresponding_pair);
        cv::imshow("image", image);
        cv::waitKey(2);
    }






    return 0;
}
