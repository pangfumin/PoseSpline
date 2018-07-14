#include "okvis_cv/cameras/camera_util.hpp"
#include "okvis_util/random.hpp"
// -----------------------------------------------------------------------------
Keypoints generateRandomKeypoints(
        const Size2u size,
        const uint32_t margin,
        const uint32_t num_keypoints)
{
//    DEBUG_CHECK_GT(size.width(), margin + 1u);
//    DEBUG_CHECK_GT(size.height(), margin + 1u);

    Keypoints kp(2, num_keypoints);
    for(uint32_t i = 0u; i < num_keypoints; ++i)
    {
        kp(0,i) = sampleUniformRealDistribution<real_t>(false, margin, size.width() - 1 - margin);
        kp(1,i) = sampleUniformRealDistribution<real_t>(false, margin, size.height() - 1 - margin);
    }
    return kp;
}

// -----------------------------------------------------------------------------
Keypoints generateUniformKeypoints(
        const Size2u size,
        const uint32_t margin,
        const uint32_t num_cols)
{
//    DEBUG_CHECK_GT(size.width(), margin + 1u);
//    DEBUG_CHECK_GT(size.height(), margin + 1u);
    const uint32_t num_rows = num_cols * size.height() / size.width();

    // Compute width and height of a cell:
    real_t w = (static_cast<real_t>(size.width() - 0.01)  - 2.0 * margin) / (num_cols - 1);
    real_t h = (static_cast<real_t>(size.height() - 0.01) - 2.0 * margin) / (num_rows - 1);

    // Sample keypoints:
    Keypoints kp(2, num_rows * num_cols);
    for (uint32_t y = 0u; y < num_rows; ++y)
    {
        for (uint32_t x = 0u; x < num_cols; ++x)
        {
            uint32_t i = y * num_cols + x;
            kp(0,i) = margin + x * w;
            kp(1,i) = margin + y * h;
        }
    }
    return kp;
}

// -----------------------------------------------------------------------------
std::tuple<Keypoints, Bearings, Positions> generateRandomVisible3dPoints(
        const okvis::cameras::NCameraSystem& cam,
        const int cam_id,
        const uint32_t num_points,
        const uint32_t margin,
        const real_t min_depth,
        const real_t max_depth)
{
    Size2u size(cam.cameraGeometry(cam_id)->imageWidth(), cam.cameraGeometry(cam_id)->imageHeight());
    Keypoints px = generateRandomKeypoints(size, margin, num_points);

    Bearings  f(3, num_points);
    cam.cameraGeometry(cam_id)->backProjectBatch(px,&f,NULL);
    Positions pos  = f;
    for(uint32_t i = 0u; i < num_points; ++i)
    {
        pos.col(i) *= sampleUniformRealDistribution<real_t>(false, min_depth, max_depth);
    }
    return std::make_tuple(px, f, pos);
}
