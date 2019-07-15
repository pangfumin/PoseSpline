cmake_minimum_required(VERSION 3.0.0)
project(PoseSpline)

set(cmake_build_type Release)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")




cs_add_library(${PROJECT_NAME}
#        src/extern/spline_projection_error.cpp
#        src/extern/pinhole_project_error.cpp
#        src/extern/project_error.cpp
#        src/extern/RotateVectorError.cpp
        src/QuaternionSplineUtility.cpp
        src/QuaternionSpline.cpp
        src/QuaternionSplineSampleError.cpp
        src/PoseSplineSampleError.cpp
        src/PoseSplineUtility.cpp
        src/OmegaExtrinsicTemperalError.cpp
        #src/PoseSpline/QuaternionOmegaSampleError.cpp
        src/PoseSpline.cpp
        src/VectorSpaceSpline.cpp
        src/VectorSplineSampleError.cpp
        src/VectorSplineSampleAutoError.cpp
        src/VectorSplineSampleVelocityError.cpp
        src/Time.cpp
        src/Duration.cpp
#        src/PoseSpline/LinearAccelerateSampleError.cpp
#        src/PoseSpline/AngularVelocitySampleError.cpp
        )




## Test

if(CATKIN_ENABLE_TESTING)
    catkin_add_gtest(geometry_tests
            test/test_main.cpp
            test/PoseTest.cpp
            test/UtilityTest.cpp
            )
    if(TARGET geometry_tests)
        target_link_libraries(geometry_tests ${PROJECT_NAME})
    endif()

    catkin_add_gtest(ceres_tests
            test/test_main.cpp
            test/QuaternionCeresTest.cpp
            )
    if(TARGET ceres_tests)
        target_link_libraries(ceres_tests ${PROJECT_NAME})
    endif()

    catkin_add_gtest(spline_tests
            test/test_main.cpp
            test/PoseSplineTest.cpp
            )
    if(TARGET spline_tests)
        target_link_libraries(spline_tests ${PROJECT_NAME})
    endif()


endif()


cs_install()
cs_export()
