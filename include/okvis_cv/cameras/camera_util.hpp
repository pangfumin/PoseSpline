#pragma once
#include "okvis_util/size.hpp"
#include "okvis_util/types.hpp"
#include "okvis_cv/cameras/NCameraSystem.hpp"

Keypoints generateRandomKeypoints(
        const Size2u size,
        const uint32_t margin,
        const uint32_t num_keypoints);

Keypoints generateUniformKeypoints(
        const Size2u size,
        const uint32_t margin,
        const uint32_t num_cols);

std::tuple<Keypoints, Bearings, Positions> generateRandomVisible3dPoints(
        const okvis::cameras::NCameraSystem& cam,
        const int cam_id,
        const uint32_t num_points,
        const uint32_t margin,
        const real_t min_depth,
        const real_t max_depth);