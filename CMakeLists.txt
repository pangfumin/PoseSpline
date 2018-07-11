cmake_minimum_required(VERSION 3.0.0)
project(PoseSpline)


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(cmake_build_type Release)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

include_directories(include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        )



add_library(cv
        src/okvis_cv/CameraBase.cpp
        src/okvis_cv/NCameraSystem.cpp)


target_link_libraries(cv
        ${OPENCV_LIBRARIES}
        ${GLOG_LIBRARIES})

add_library( PoseSpline
        src/utility/Duration.cpp
        src/utility/Time.cpp
        src/extern/spline_projection_error.cpp
        src/extern/pinhole_project_error.cpp
        src/extern/project_error.cpp
        src/extern/RotateVectorError.cpp
        src/pose-spline/QuaternionSplineUtility.cpp
        src/pose-spline/QuaternionSpline.cpp
        src/pose-spline/QuaternionLocalParameter.cpp
        src/pose-spline/QuaternionSplineSampleError.cpp
        src/pose-spline/PoseSplineSampleError.cpp
        src/pose-spline/PoseSplineUtility.cpp
        src/pose-spline/OmegaExtrinsicTemperalError.cpp
        src/pose-spline/QuaternionOmegaSampleError.cpp
        src/pose-spline/PoseSpline.cpp
        src/pose-spline/PoseLocalParameter.cpp
        src/pose-spline/VectorSpaceSpline.cpp
        src/pose-spline/VectorSplineSampleError.cpp
        src/pose-spline/VectorSplineSampleAutoError.cpp
        src/pose-spline/VectorSplineSampleVelocityError.cpp
        src/pose-spline/LinearAccelerateSampleError.cpp
        src/pose-spline/AngularVelocitySampleError.cpp
        src/pose-spline/csv_trajectory.cpp)


target_link_libraries(PoseSpline
        ${CERES_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES})


## Test

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR include/utility/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_subdirectory(test)
