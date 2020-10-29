# Make test executable


add_executable(QuaternionCeresTest QuaternionCeresTest.cpp)
target_link_libraries(QuaternionCeresTest
        PoseSpline)


add_executable(QuaternionSplineCeresTest QuaternionSplineCeresTest.cpp)
target_link_libraries(QuaternionSplineCeresTest
        PoseSpline)

add_executable(QuaternionOmegaCeresTest QuaternionOmegaCeresTest.cpp)
target_link_libraries(QuaternionOmegaCeresTest
        PoseSpline)

add_executable(VectorSplineVelocityCeresTest VectorSplineVelocityCeresTest.cpp)
target_link_libraries(VectorSplineVelocityCeresTest
        PoseSpline)

add_executable(VectorSplineCeresTest VectorSplineCeresTest.cpp)
target_link_libraries(VectorSplineCeresTest
        PoseSpline)

add_executable(testCamera testCamera.cpp)
target_link_libraries(testCamera
        PoseSpline)

add_executable(AngularVelocitySampleTest AngularVelocitySampleTest.cpp)
target_link_libraries(AngularVelocitySampleTest
        PoseSpline)


add_executable(QuaternionSplineTest QuaternionSplineTest.cpp)
target_link_libraries(QuaternionSplineTest
        PoseSpline)

add_executable(QuaternionPrediction QuaternionPrediction.cpp)
target_link_libraries(QuaternionPrediction
        PoseSpline)

add_executable(VectorSpaceSplineTest VectorSpaceSplineTest.cpp)
target_link_libraries(VectorSpaceSplineTest
        PoseSpline)


add_executable(testPinholeProjectFactor testPinholeProjectFactor.cpp)
target_link_libraries(testPinholeProjectFactor
        PoseSpline)



add_executable(testPoseSplineSampleError testPoseSplineSampleError.cpp)
target_link_libraries(testPoseSplineSampleError
        PoseSpline)

add_executable(testLinearAccelerateSampleError testLinearAccelerateSampleError.cpp)
target_link_libraries(testLinearAccelerateSampleError
        PoseSpline)


add_executable(testSplineRotateError testSplineRotateError.cpp)
target_link_libraries(testSplineRotateError
        PoseSpline)


# gtest
enable_testing()
set(FRAME_TEST_SOURCES
  frame/TestFrame.cpp
  frame/TestMultiFrame.cpp
  frame/TestNCameraSystem.cpp  
  frame/TestPinholeCamera.cpp)

add_executable(frame_test ${FRAME_TEST_SOURCES} ./gtest/gtest-all.cc)
target_link_libraries(frame_test pthread  cv util  ${OpenCV_LIBRARIES})
set_target_properties(frame_test PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")

add_executable(pose_test PoseTest.cpp ./gtest/gtest-all.cc)
target_link_libraries(pose_test  pthread  util PoseSpline  )
set_target_properties(pose_test PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")

add_executable(UtilityTest UtilityTest.cpp ./gtest/gtest-all.cc)
target_link_libraries(UtilityTest pthread PoseSpline)
set_target_properties(pose_test PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")


add_executable(PoseSplineTest PoseSplineTest.cpp ./gtest/gtest-all.cc)
target_link_libraries(PoseSplineTest pthread PoseSpline)
set_target_properties(PoseSplineTest PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")

add_executable(testSimulator testSimulator.cpp ./gtest/gtest-all.cc)
target_link_libraries(testSimulator pthread util simulate PoseSpline estimator ceres_ext frontend)
set_target_properties(testSimulator PROPERTIES COMPILE_FLAGS "-std=c++11 -Wno-deprecated-register")


add_test(NAME frame_test COMMAND frame_test --gtest_color=yes)


add_executable(test_buildOpticalFlowPyramid test_buildOpticalFlowPyramid.cpp)
target_link_libraries(test_buildOpticalFlowPyramid
        PoseSpline)


add_executable(test_pnp pnp/test_pnp.cpp
        pnp/pose_local_parameterization.cpp
        pnp/utility.cpp
        pnp/project_error.cpp)
target_link_libraries(test_pnp
        PoseSpline)

add_executable(testProjectError pnp/testProjectError.cpp
        pnp/pose_local_parameterization.cpp
        pnp/utility.cpp
        pnp/project_error.cpp)
target_link_libraries(testProjectError
        PoseSpline)


add_executable(human_2d_3d_alignment pnp/human_2d_3d_alignment.cpp
        pnp/pose_local_parameterization.cpp
        pnp/utility.cpp
        pnp/project_error.cpp)
target_link_libraries(human_2d_3d_alignment
        PoseSpline)

add_executable(test_opencvPnp pnp/test_opencvPnp.cpp
        pnp/pose_local_parameterization.cpp
        pnp/utility.cpp
        pnp/project_error.cpp)
target_link_libraries(test_opencvPnp
        PoseSpline)





# # Copy test data.
# file(COPY ./data DESTINATION ${CMAKE_BINARY_DIR})