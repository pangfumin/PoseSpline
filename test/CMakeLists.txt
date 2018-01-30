# Make test executable

add_executable(test_main
        test_main.cpp PoseTest.cpp)
target_link_libraries(test_main PoseSpline Catch)
