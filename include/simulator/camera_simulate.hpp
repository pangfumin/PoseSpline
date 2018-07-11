#ifndef  _CAMERA_SIMULATE_H_
#define  _CAMERA_SIMULATE_H_

#include <vector>
#include <memory>
#include <okvis_cv/MultiFrame.hpp>
#include <okvis_cv/cameras/NCameraSystem.hpp>
#include <okvis_util/types.hpp>

// -----------------------------------------------------------------------------
struct CameraSimulatorOptions
{
    uint32_t min_num_keypoints_per_frame { 50  };
    real_t keypoint_noise_sigma { 1.0 };
    uint32_t max_num_landmarks_ { 10000 };
    real_t min_depth_m { 2.0 };
    real_t max_depth_m { 7.0 };
};


struct CameraMeasurements
{
    //! Each column is a keypoint observation.
    Keypoints keypoints_;

    //! Global landmark index of each observed feature. The size of the vector is
    //! the same as the number of columns in the keypoints block.
    std::vector<int32_t> global_landmark_ids_;

    //! Temporary track index of a landmark. If the landmark is
    //! re-observed after a loop, it will be assigned a different id.
    std::vector<int32_t> local_track_ids_;
};
using CameraMeasurementsVector = std::vector<CameraMeasurements>;


using Trajectory =  std::vector<std::pair<Time, Pose<double>> >;
class CameraSimulator {
public:
    CameraSimulator(const std::shared_ptr<Trajectory>& trajectory,
                    const okvis::cameras::NCameraSystem& nCameraSystem,
                    const CameraSimulatorOptions cameraSimulatorOptions );
private:

    void initializeMap();
    CameraMeasurements visibleLandmarks(
            const uint32_t cam_idx,
            const Pose<double>& T_W_B,
            const uint32_t lm_min_idx,
            const uint32_t lm_max_idx);
    okvis::cameras::NCameraSystem nCameraSystem_;
    std::vector<std::shared_ptr<okvis::MultiFrame>> frame_;
    std::shared_ptr<Trajectory> trajectory_;
    CameraSimulatorOptions options_;

    Positions landmarks_W_;
};

#endif