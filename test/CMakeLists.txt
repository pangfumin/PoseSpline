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




add_executable(AngularVelocitySampleTest AngularVelocitySampleTest.cpp)
target_link_libraries(AngularVelocitySampleTest
        PoseSpline)


add_executable(QuaternionSplineTest QuaternionSplineTest.cpp)
target_link_libraries(QuaternionSplineTest
        PoseSpline)

add_executable(PoseSplineTest PoseSplineTest.cpp)
target_link_libraries(PoseSplineTest
        PoseSpline)



add_executable(VectorSpaceSplineTest VectorSpaceSplineTest.cpp)
target_link_libraries(VectorSpaceSplineTest
        PoseSpline)

add_executable(UtilityTest UtilityTest.cpp)
target_link_libraries(UtilityTest
        PoseSpline)



add_executable(testPinholeProjectFactor testPinholeProjectFactor.cpp)
target_link_libraries(testPinholeProjectFactor
        PoseSpline)

add_executable(testProjectError testProjectError.cpp)
target_link_libraries(testProjectError
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



#
#add_executable(testDataReader testDataReader.cpp)
#target_link_libraries(testDataReader
#        PoseSpline)
#


add_executable(test_main
        test_main.cpp PoseTest.cpp)
target_link_libraries(test_main PoseSpline Catch)
