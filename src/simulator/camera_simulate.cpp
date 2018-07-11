#include "simulator/camera_simulate.hpp"
#include "okvis_util/size.hpp"
CameraSimulator::CameraSimulator(const std::shared_ptr<Trajectory>& trajectory,
                                 const okvis::cameras::NCameraSystem& nCameraSystem,
                const CameraSimulatorOptions cameraSimulatorOptions ):
trajectory_(trajectory), nCameraSystem_(nCameraSystem), options_(cameraSimulatorOptions) {

}

void CameraSimulator::initializeMap() {
    int64_t num_frames = trajectory_->size();
    int  num_camera = nCameraSystem_.numCameras();
    for (size_t i = 0 ; i < num_frames; i++) {
        for (size_t j= 0 ; j < num_camera; j ++) {

        }
    }

}

// -----------------------------------------------------------------------------
CameraMeasurements CameraSimulator::visibleLandmarks(
        const uint32_t cam_idx,
        const Pose<double>& T_W_B,
        const uint32_t lm_min_idx,
        const uint32_t lm_max_idx)
{


    const uint32_t num_landmarks = lm_max_idx - lm_min_idx;
    if (num_landmarks == 0)
    {
        return CameraMeasurements();
    }

    const Size2u image_size ( nCameraSystem_.cameraGeometry(cam_idx)->imageWidth(),
                              nCameraSystem_.cameraGeometry(cam_idx)->imageHeight());
    const auto lm_W = landmarks_W_.middleCols(lm_min_idx, num_landmarks);
    const auto T_C_W = (T_W_B * (*nCameraSystem_.T_SC(cam_idx))).inverse();

    const auto lm_C =
    Keypoints px = rig_->at(cam_idx).projectVectorized(lm_C);
    std::vector<uint32_t> visible_indices;
    for (uint32_t i = 0u; i < num_landmarks; ++i)
    {
        if (lm_C(2,i) < options_.min_depth_m ||
            lm_C(2,i) > options_.max_depth_m)
        {
            // Landmark is either behind or too far from the camera.
            continue;
        }

        if (isVisible(image_size, px.col(i)))
        {
            visible_indices.push_back(i);
        }
    }

    // Copy visible indices into Camera Measurements struct:
    CameraMeasurements m;
    m.keypoints_.resize(Eigen::NoChange, visible_indices.size());
    m.global_landmark_ids_.resize(visible_indices.size());
    for (size_t i = 0; i < visible_indices.size(); ++i)
    {
        m.keypoints_.col(i) = px.col(visible_indices[i]);
        m.global_landmark_ids_[i] = visible_indices[i];
    }

    return m;
}