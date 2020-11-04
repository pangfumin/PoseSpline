#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/core/eigen.hpp>
#include "tiny_solver_multiple_function.h"

#include <algorithm>    // std::max

int width = 544;
int height = 960;
double focal = 516;


class ProjectEdge {
public:
    typedef double Scalar;
    enum {
        // Can also be Eigen::Dynamic.
        NUM_RESIDUALS = 2,
        NUM_PARAMETERS = 3,
    };
    ProjectEdge( const Eigen::Vector2d& uv, const Eigen::Vector3d& pt3d,
                 const double cx, const double cy, double focal ):
            uv_(uv), pt3d_(pt3d), cx_(cx), cy_(cy), focal_(focal){

    }

    bool EvaluateResidualsAndJacobians2(const double* parameters,
                                        double* residuals,
                                        double* jacobian) const  {
        double a = parameters[0];
        double b = parameters[1];
        double c = parameters[2];




        Eigen::Map<const Eigen::Matrix<double,3,1>> t(parameters);
        // T_WC
        cache_Cp_ =  pt3d_ + t;
        cache_inv_z_ = 1/cache_Cp_(2);
        Eigen::Vector2d hat_C0uv(focal_ * cache_Cp_(0)*cache_inv_z_ + cx_, focal_ * cache_Cp_(1)*cache_inv_z_ + cy_);

        Eigen::Map<Eigen::Matrix<double, 2,1>> residual(residuals);
        residual = hat_C0uv - uv_;



        if (jacobian) {
            Eigen::Map<Eigen::Matrix<double, 2, 3>> jaco_abc(jacobian);  // 误差为1维，状态量 3 个，所以是 1x3 的雅克比矩阵
            Eigen::Matrix<double,2,3> H;
            H << 1, 0, -cache_Cp_(0)*cache_inv_z_,
                    0, 1, -cache_Cp_(1)*cache_inv_z_;
            H *= focal_ * cache_inv_z_;


            jaco_abc = H ;


        }
        return true;
    }


    bool operator()(const double* parameters,
                    double* residuals,
                    double* jacobian) const {
        return EvaluateResidualsAndJacobians2(parameters, residuals, jacobian);
    }
    Eigen::Vector2d uv_;
    Eigen::Vector3d pt3d_;
    double cx_, cy_, focal_;

    mutable Eigen::Vector3d cache_Cp_;
    mutable  double cache_inv_z_;
};



Eigen::Vector3d pnp(Eigen::Vector3d t,
        std::vector<Eigen::Vector2d>& pt2ds, std::vector<Eigen::Vector3d>& pt3ds) {
    std::vector<Eigen::Vector3d> used_pt3d_bearing, used_pt3d;


    double cx = width / 2;
    double cy = height / 2;
//

    solver::TinySolverMultipleFunction<ProjectEdge> solver_multiple;
    std::vector<ProjectEdge> f_vec;
    for (int i = 0; i < pt3ds.size(); i++) {
        auto pt2d = pt2ds[i];
        auto pt3d = pt3ds[i];

        ProjectEdge factor(pt2d, pt3d, cx, cy, focal);
        f_vec.push_back(factor);

    }



    Eigen::Vector3d other_x(0,0,0);
    solver_multiple.Solve(f_vec, &other_x);

//    std::cout << "ceres: " << t.transpose() << std::endl;



    return other_x;

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

void loadData( const std::string data_file, std::vector<int>& data_id, std::vector<int>& image_id, std::vector<std::vector<Eigen::Vector2d>>& pt2ds, std::vector<std::vector<Eigen::Vector3d>>& pt3ds) {
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

            data_id.push_back(std::stod(content[0]));
            image_id.push_back(std::stod(content[1]));

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

void visualize(cv::Mat& image, std::vector<Eigen::Vector2d> pt2ds, std::vector<Eigen::Vector3d> pt3ds,
        Eigen::Vector3d t,std::vector<int> index_2d, std::map<int, int> corresponding_pair) {


//    std::cout << "T_WC: \n " << T_WC << std::endl;
//
//    Eigen::Vector3d t_WC = T_WC.topRightCorner(3,1);
//    Eigen::Matrix3d R_WC = T_WC.topLeftCorner(3,3);



    double cx = width / 2;
    double cy = height / 2;
//    for (int i = 0; i < index_2d.size(); i++) {
//
//    }


    for (int i = 0; i < corresponding_pair.size(); i++) {

        int id1 = corresponding_pair[index_2d[i]];
        auto pt3d = pt3ds[id1] + t;
//        pt3d  = R_WC.transpose()*(pt3d - t_WC);

        // raw 3d
        Eigen::Vector2d pt2d;
        pt2d << (pt3d(0)/ pt3d(2)) * focal + cx, (pt3d(1)/ pt3d(2)) * focal + cy;
        cv::Point2f pt0( pt2d.x(), pt2d.y());
        cv::circle(image, pt0,3,cv::Scalar(10,255,1),3 );

        std::string text = std::to_string(id1);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.51;
        int thickness = 2;
        cv::putText(image, text, pt0, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);



        // raw 2d
        cv::Point2f pt1(  pt2ds.at(index_2d[i]).x(), pt2ds.at(index_2d[i]).y());

         text = std::to_string(index_2d[i]);
         font_face = cv::FONT_HERSHEY_COMPLEX;
         font_scale = 0.51;
         thickness = 2;
        cv::putText(image, text, pt1, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);
        cv::circle(image, pt1,3,cv::Scalar(255,0,1),3 );

//        cv::line(image, pt0, pt1, cv::Scalar(15,200,100), 2);


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

    start = 5;
    end = 6;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);


    start = 8;
    end = 9;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);
    start = 11;
    end = 12;
    cv::line(image, cv::Point2f(  pt2ds.at(start).x(), pt2ds.at(start).y()), cv::Point2f(  pt2ds.at(end).x(), pt2ds.at(end).y()), cv::Scalar(255,0,0), 2);




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

//    double scale = std::max(max_x - min_x, std::max(max_y - min_y, max_z - min_z));
    double scale = 1.0;

//    std::cout << "max_x: " << max_x << " " << min_x << " " << max_y << " " << min_y << " " << max_z << " " << min_z  << " " << scale<< std::endl;

    for (int i = 0; i < rescaled_shift_pt3ds.size(); i++) {
        rescaled_shift_pt3ds[i] = rescaled_shift_pt3ds[i] + Eigen::Vector3d(0,0,1);

    }

    return rescaled_shift_pt3ds;
}



std::vector<Eigen::Vector2d> normalize2d( std::vector<Eigen::Vector2d>& pt2ds)  {

    double cx = width / 2;
    double cy = height / 2;

    Eigen::Vector2d  diff = Eigen::Vector2d(cx, cy) - pt2ds[18];
    std::vector<Eigen::Vector2d> normalized;

    for (int i =0; i < pt2ds.size(); i++) {
        normalized.push_back(pt2ds[i] + diff);
    }
    return normalized;
}


void evaluate(cv::Mat& image, std::vector<Eigen::Vector2d> pt2ds, std::vector<Eigen::Vector3d> pt3ds,
               Eigen::Vector3d t, int* index_2d, int* index_3d, int cnt, std::vector<Eigen::Vector2d>& gt2d, std::vector<Eigen::Vector2d>& rep2d, double& mean_error) {


//    std::cout << "T_WC: \n " << T_WC << std::endl;
//
//    Eigen::Vector3d t_WC = T_WC.topRightCorner(3,1);
//    Eigen::Matrix3d R_WC = T_WC.topLeftCorner(3,3);



    double cx = width / 2;
    double cy = height / 2;
//    for (int i = 0; i < index_2d.size(); i++) {
//
//    }

    double error = 0;
    for (int i = 0; i <cnt; i++) {

        int id1 = index_3d[i];
        auto pt3d = pt3ds[id1] + t;
//        pt3d  = R_WC.transpose()*(pt3d - t_WC);

        // raw 3d
        Eigen::Vector2d pt2d;
        pt2d << (pt3d(0)/ pt3d(2)) * focal + cx, (pt3d(1)/ pt3d(2)) * focal + cy;
        cv::Point2f pt0( pt2d.x(), pt2d.y());
        cv::circle(image, pt0,3,cv::Scalar(10,255,1),3 );

        std::string text = std::to_string(id1);
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.51;
        int thickness = 2;
        cv::putText(image, text, pt0, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);



        // raw 2d
        cv::Point2f pt1(  pt2ds.at(index_2d[i]).x(), pt2ds.at(index_2d[i]).y());

        text = std::to_string(index_2d[i]);
        font_face = cv::FONT_HERSHEY_COMPLEX;
        font_scale = 0.51;
        thickness = 2;
        cv::putText(image, text, pt1, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);
        cv::circle(image, pt1,3,cv::Scalar(255,0,1),3 );

//        cv::line(image, pt0, pt1, cv::Scalar(15,200,100), 2);

        gt2d.push_back(pt2ds.at(index_2d[i]));
        rep2d.push_back(pt2d);


        error += (gt2d.back() - rep2d.back()).norm();
    }

    mean_error = error / 13;




}

int main(int argc, char** argv){
    //google::InitGoogleLogging(argv[0]);

    Eigen::Matrix4d T_WC = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> pt3d, pt3d_bearing;
    std::vector<Eigen::Vector2d> pt2d;
    std::vector<std::vector<Eigen::Vector2d>> pt2ds;
    std::vector<std::vector<Eigen::Vector3d>> pt3ds;

    std::string data_file = "/home/pang/Downloads/handHold_2d3d_keypoints.txt";
    std::vector<int> data_id;
    std::vector<int> image_id;
    loadData(data_file, data_id, image_id, pt2ds, pt3ds);


    std::ofstream ofs_point_data("/home/pang/Downloads/handHold_2d3d_keypoints_RESULT.txt");
    std::ofstream ofs_error_data("/home/pang/Downloads/handHold_2d3d_keypoints_MEAN_ERROR_RESULT.txt");

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

    // 膝盖
    corresponding_pair.insert({9,6});
    corresponding_pair.insert({12,2});



    index_2d.push_back(3);
    index_2d.push_back(6);
    index_2d.push_back(9);
    index_2d.push_back(12);


    int eval_2d_index[13] = {11, 8, 5,  2, 10, 9, 13, 12, 4,  3,  7,  6, 0};
    int eval_3d_index[13] = {1,  5,17, 13, 7, 6,  3,  2, 15, 14, 19, 18, 11};

    int cur_data = data_id[0];
    double mean_error_per_data = 0;
    int cur_data_cnt = 0;


    for (int i = 0; i < pt2ds.size(); i++) {

        std::vector<Eigen::Vector3d> normalize_pt3d = normalize3d(pt3ds[i]);
//        std::vector<Eigen::Vector2d> normalize_pt2d = normalize2d(pt2ds[i]);

        std::vector<Eigen::Vector2d> used_pt2d;
        std::vector<Eigen::Vector3d> used_pt3d;
        for (int j = 0; j < index_2d.size();j++) {
            used_pt2d.push_back(pt2ds[i][index_2d[j]]);
            int id1 = corresponding_pair[index_2d[j]];
            used_pt3d.push_back(normalize_pt3d[id1]);
        }

        Eigen::Vector3d est_t(0,0,0);
        est_t = pnp(est_t,used_pt2d, used_pt3d);



        cv::Mat image(960, 544,  CV_8UC3);
        image.setTo(cv::Scalar(255,255,255));

        visualize(image, pt2ds[i], normalize_pt3d, est_t,index_2d, corresponding_pair);




        // evaluate
        cv::Mat eval_image(960, 544,  CV_8UC3);
        eval_image.setTo(cv::Scalar(255,255,255));


        std::vector<Eigen::Vector2d> gt2d, rep2d;
        double mean_error = 0;
        evaluate(eval_image, pt2ds[i], normalize_pt3d, est_t, eval_2d_index, eval_3d_index, 13,gt2d, rep2d, mean_error);

        std::cout << "mean_error: " << data_id[i] << " " << image_id[i] << " " << mean_error  << std::endl;
        ofs_point_data << data_id[i] << " " << image_id[i] << " " << mean_error << " ";
        for (int k  = 0 ; k < gt2d.size(); k++) {
            ofs_point_data << gt2d[k].x() << " " << gt2d[k].y() << " ";
        }

        for (int k  = 0 ; k < rep2d.size(); k++) {
            ofs_point_data << rep2d[k].x() << " " << rep2d[k].y() << " ";
        }

        ofs_point_data << std::endl;

        if (data_id[i] != cur_data) {


            std::cout << "------------------------------"  << mean_error_per_data / cur_data_cnt<< std::endl;
            ofs_error_data << cur_data << " " << mean_error_per_data / cur_data_cnt << std::endl;
            cur_data = data_id[i];
            mean_error_per_data = 0;
            cur_data_cnt = 0;
        } else {
            mean_error_per_data += mean_error;
            cur_data_cnt++;
        }



        cv::imshow("image", image);
        cv::imshow("eval_image", eval_image);
        cv::waitKey(20);
    }

    // todo
    ofs_error_data << cur_data << " " << mean_error_per_data / cur_data_cnt << std::endl;


    ofs_point_data.close();
    ofs_error_data.close();






    return 0;
}
