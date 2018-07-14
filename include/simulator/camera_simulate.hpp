#ifndef  _CAMERA_SIMULATE_H_
#define  _CAMERA_SIMULATE_H_

#include <vector>
#include <memory>
#include <okvis_cv/MultiFrame.hpp>
#include <okvis_cv/cameras/NCameraSystem.hpp>
#include <okvis_util/types.hpp>
#include <okvis_util/size.hpp>

// -----------------------------------------------------------------------------
struct CameraSimulatorOptions
{
    uint32_t min_num_keypoints_per_frame { 50  };
    real_t keypoint_noise_sigma { 1.0 };
    uint32_t max_num_landmarks_ { 10000 };
    real_t min_depth_m { 1.0 };
    real_t max_depth_m { 15.0 };
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

/**
 * todo(pang) : initial pose with posespline, for simulation temperal offset measurement
 */

class CameraSimulator {
public:
    CameraSimulator(const std::shared_ptr<Trajectory>& trajectory,
                    const okvis::cameras::NCameraSystem& nCameraSystem,
                    const CameraSimulatorOptions cameraSimulatorOptions );

    uint32_t getLandmarksNum() {return num_landmarks_;};
    Position getLandmark(const uint64_t& id) {return landmarks_W_.col(id);};

    bool hasNextMeasurement();
    okvis::MultiFramePtr  getNextMeasurement();
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

    Trajectory::const_iterator trajectory_itr_;

    uint32_t num_landmarks_;
    Positions landmarks_W_;

    const uint32_t landmark_preAlooc_num_;
};

#endif