# Make test executable


add_executable(QuaternionCeresTest QuaternionCeresTest.cpp)
target_link_libraries(QuaternionCeresTest
        PoseSpline)


add_executable(QuaternionSplineCeresTest QuaternionSplineCeresTest.cpp)
target_link_libraries(QuaternionSplineCeresTest
        PoseSpline)

add_executable(VectorSplineCeresTest VectorSplineCeresTest.cpp)
target_link_libraries(VectorSplineCeresTest
        PoseSpline)


add_executable(SplineTest SplineTest.cpp)
target_link_libraries(SplineTest
        PoseSpline)


add_executable(UtilityTest UtilityTest.cpp)
target_link_libraries(UtilityTest
        PoseSpline)


add_executable(test_main
        test_main.cpp PoseTest.cpp)
target_link_libraries(test_main PoseSpline Catch)
