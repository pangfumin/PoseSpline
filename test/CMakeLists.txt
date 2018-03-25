# Make test executable


add_executable(QuaternionCeresTest QuaternionCeresTest.cpp)
target_link_libraries(QuaternionCeresTest
        PoseSpline)


add_executable(QuaternionSplineCeresTest QuaternionSplineCeresTest.cpp)
target_link_libraries(QuaternionSplineCeresTest
        PoseSpline)

add_executable(VectorSplineVelocityCeresTest VectorSplineVelocityCeresTest.cpp)
target_link_libraries(VectorSplineVelocityCeresTest
        PoseSpline)

add_executable(VectorSplineCeresTest VectorSplineCeresTest.cpp)
target_link_libraries(VectorSplineCeresTest
        PoseSpline)



add_executable(SplineTest SplineTest.cpp)
target_link_libraries(SplineTest
        PoseSpline)


add_executable(VectorSpaceSplineTest VectorSpaceSplineTest.cpp)
target_link_libraries(VectorSpaceSplineTest
        PoseSpline)

add_executable(UtilityTest UtilityTest.cpp)
target_link_libraries(UtilityTest
        PoseSpline)

add_executable(QuaternionOmegaSampleErrorTest QuaternionOmegaSampleErrorTest.cpp)
target_link_libraries(QuaternionOmegaSampleErrorTest
        PoseSpline)


add_executable(testDataReader testDataReader.cpp)
target_link_libraries(testDataReader
        PoseSpline)



add_executable(test_main
        test_main.cpp PoseTest.cpp)
target_link_libraries(test_main PoseSpline Catch)
