#include <string>
#include <iostream>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main (){

    std::string image_file = "/home/pang/Pictures/Selection_022.png";
    cv::Mat im = cv::imread(image_file,CV_LOAD_IMAGE_GRAYSCALE);
    cv::Size win_size = cv::Size(21, 21);
    int max_level = 5;
    std::vector<cv::Mat> pyr;
    int level = cv::buildOpticalFlowPyramid(im, pyr, win_size, max_level);
    std::cout << "pyr: " << pyr.size()  << " " << level<< std::endl;


    for (auto image : pyr) {

        std::cout << "image: " << image.cols << " " << image.rows << std::endl;
    }

    cv::imshow("image", pyr[6]);
    cv::waitKey();


    return 0;
}