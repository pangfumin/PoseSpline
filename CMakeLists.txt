cmake_minimum_required(VERSION 3.0.0)
project(PoseSpline)


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(cmake_build_type Debug)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

include_directories(include
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        )
include_directories( ${GLOG_INCLUDE_DIRS})

add_library(Spline
        src/spline/bspline.cpp
        )
target_link_libraries(Spline
        ${CERES_LIBRARIES}
        ${OPENCV_LIBRARIES}
        glog::glog)


add_library( PoseSpline
        src/utility/Duration.cpp
        src/utility/Time.cpp
        src/pose-spline/QuaternionSplineUtility.cpp
        src/pose-spline/QuaternionSpline.cpp
        src/pose-spline/QuaternionLocalParameter.cpp
        src/pose-spline/QuaternionSplineSampleError.cpp
        src/pose-spline/OmegaExtrinsicTemperalError.cpp
        src/pose-spline/PoseSpline.cpp
        src/pose-spline/PoseLocalParameter.cpp
        src/pose-spline/VectorSpaceSpline.cpp
        src/pose-spline/csv_trajectory.cpp)


target_link_libraries(PoseSpline
        Spline
        ${CERES_LIBRARIES}
        ${OPENCV_LIBRARIES}
        glog::glog)


## Test

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR include/utility/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_subdirectory(test)



