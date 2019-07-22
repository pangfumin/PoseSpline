/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Sep 3, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file ImuError.hpp
 * @brief Header file for the ImuError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_IMUERROR_HPP_
#define INCLUDE_OKVIS_CERES_IMUERROR_HPP_

#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <PoseSpline/Time.hpp>
#include "Transformation.hpp"
#include <queue>

/// \brief okvis Main namespace of this package.
namespace okvis {
    template<class MEASUREMENT_T>
    struct Measurement {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Time timeStamp;      ///< Measurement timestamp
        MEASUREMENT_T measurement;  ///< Actual measurement.
        int sensorId = -1;          ///< Sensor ID. E.g. camera index in a multicamera setup

        /// \brief Default constructor.
        Measurement()
                : timeStamp(0.0) {
        }
        /**
         * @brief Constructor
         * @param timeStamp_ Measurement timestamp.
         * @param measurement_ Actual measurement.
         * @param sensorId Sensor ID (optional).
         */
        Measurement(const Time& timeStamp_, const MEASUREMENT_T& measurement_,
                    int sensorId = -1)
                : timeStamp(timeStamp_),
                  measurement(measurement_),
                  sensorId(sensorId) {
        }
    };

/// \brief IMU measurements. For now assume they are synchronized:
    struct ImuSensorReadings {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        /// \brief Default constructor.
        ImuSensorReadings()
                : gyroscopes(),
                  accelerometers() {
        }
        /**
         * @brief Constructor.
         * @param gyroscopes_ Gyroscope measurement.
         * @param accelerometers_ Accelerometer measurement.
         */
        ImuSensorReadings(Eigen::Vector3d gyroscopes_,
                          Eigen::Vector3d accelerometers_)
                : gyroscopes(gyroscopes_),
                  accelerometers(accelerometers_) {
        }
        Eigen::Vector3d gyroscopes;     ///< Gyroscope measurement.
        Eigen::Vector3d accelerometers; ///< Accelerometer measurement.
    };

    typedef Measurement<ImuSensorReadings> ImuMeasurement;
    typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement> > ImuMeasurementDeque;

    struct ImuParameters{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double a_max;  ///< Accelerometer saturation. [m/s^2]
        double g_max;  ///< Gyroscope saturation. [rad/s]
        double sigma_g_c;  ///< Gyroscope noise density.
        double sigma_bg;  ///< Initial gyroscope bias.
        double sigma_a_c;  ///< Accelerometer noise density.
        double sigma_ba;  ///< Initial accelerometer bias
        double sigma_gw_c; ///< Gyroscope drift noise density.
        double sigma_aw_c; ///< Accelerometer drift noise density.
        double tau;  ///< Reversion time constant of accerometer bias. [s]
        double g;  ///< Earth acceleration.
        Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
        int rate;  ///< IMU rate in Hz.
    };

    typedef Eigen::Matrix<double, 9,1> SpeedAndBias;

/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
    namespace ceres {

        namespace ode {

// to make things a bit faster than using angle-axis conversion:
            __inline__ double sinc(double x) {
                if (fabs(x) > 1e-6) {
                    return sin(x) / x;
                } else {
                    static const double c_2 = 1.0 / 6.0;
                    static const double c_4 = 1.0 / 120.0;
                    static const double c_6 = 1.0 / 5040.0;
                    const double x_2 = x * x;
                    const double x_4 = x_2 * x_2;
                    const double x_6 = x_2 * x_2 * x_2;
                    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
                }
            }
        }
            /// \brief Implements a nonlinear IMU factor.
        class ImuError :
                public ::ceres::SizedCostFunction<15 /* number of residuals */,
                        7 /* size of first parameter (PoseParameterBlock k) */,
                        9 /* size of second parameter (SpeedAndBiasParameterBlock k) */,
                        7 /* size of third parameter (PoseParameterBlock k+1) */,
                        9 /* size of fourth parameter (SpeedAndBiasParameterBlock k+1) */> {
        public:

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            /// \brief The base in ceres we derive from
            typedef ::ceres::SizedCostFunction<15, 7, 9, 7, 9> base_t;

            /// \brief The number of residuals
            static const int kNumResiduals = 15;

            /// \brief The type of the covariance.
            typedef Eigen::Matrix<double, 15, 15> covariance_t;

            /// \brief The type of the information (same matrix dimension as covariance).
            typedef covariance_t information_t;

            /// \brief The type of hte overall Jacobian.
            typedef Eigen::Matrix<double, 15, 15> jacobian_t;

            /// \brief The type of the Jacobian w.r.t. poses --
            /// \warning This is w.r.t. minimal tangential space coordinates...
            typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

            /// \brief The type of Jacobian w.r.t. Speed and biases
            typedef Eigen::Matrix<double, 15, 9> jacobian1_t;

            /// \brief Default constructor -- assumes information recomputation.
            ImuError() {
            }

            /// \brief Trivial destructor.
            virtual ~ImuError() {
            }

            /// \brief Construct with measurements and parameters.
            /// \@param[in] imuMeasurements All the IMU measurements.
            /// \@param[in] imuParameters The parameters to be used.
            /// \@param[in] t_0 Start time.
            /// \@param[in] t_1 End time.
            ImuError(const okvis::ImuMeasurementDeque & imuMeasurements,
                     const okvis::ImuParameters & imuParameters, const Time& t_0,
                     const Time& t_1);

            /**
             * @brief Propagates pose, speeds and biases with given IMU measurements.
             * @remark This can be used externally to perform propagation
             * @param[in] imuMeasurements All the IMU measurements.
             * @param[in] imuParams The parameters to be used.
             * @param[inout] T_WS Start pose.
             * @param[inout] speedAndBiases Start speed and biases.
             * @param[in] t_start Start time.
             * @param[in] t_end End time.
             * @param[out] covariance Covariance for GIVEN start states.
             * @param[out] jacobian Jacobian w.r.t. start states.
             * @return Number of integration steps.
             */
            static int propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                                   const okvis::ImuParameters & imuParams,
                                   okvis::kinematics::Transformation& T_WS,
                                   okvis::SpeedAndBias & speedAndBiases,
                                   const Time& t_start, const Time& t_end,
                                   covariance_t* covariance = 0,
                                   jacobian_t* jacobian = 0);

            /**
             * @brief Propagates pose, speeds and biases with given IMU measurements.
             * @warning This is not actually const, since the re-propagation must somehow be stored...
             * @param[in] T_WS Start pose.
             * @param[in] speedAndBiases Start speed and biases.
             * @return Number of integration steps.
             */
            int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                                   const okvis::SpeedAndBias & speedAndBiases) const;

            // setters

            /// \brief (Re)set the parameters.
            /// \@param[in] imuParameters The parameters to be used.
            void setImuParameters(const okvis::ImuParameters& imuParameters) {
                imuParameters_ = imuParameters;
            }

            /// \brief (Re)set the measurements
            /// \@param[in] imuMeasurements All the IMU measurements.
            void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
                imuMeasurements_ = imuMeasurements;
            }

            /// \brief (Re)set the start time.
            /// \@param[in] t_0 Start time.
            void setT0(const Time& t_0) {
                t0_ = t_0;
            }

            /// \brief (Re)set the start time.
            /// \@param[in] t_1 End time.
            void setT1(const Time& t_1) {
                t1_ = t_1;
            }

            // getters

            /// \brief Get the IMU Parameters.
            /// \return the IMU parameters.
            const okvis::ImuParameters& imuParameters() const {
                return imuParameters_;
            }

            /// \brief Get the IMU measurements.
            const okvis::ImuMeasurementDeque& imuMeasurements() const {
                return imuMeasurements_;
            }

            /// \brief Get the start time.
            Time t0() const {
                return t0_;
            }

            /// \brief Get the end time.
            Time t1() const {
                return t1_;
            }

            // error term and Jacobian implementation
            /**
             * @brief This evaluates the error term and additionally computes the Jacobians.
             * @param parameters Pointer to the parameters (see ceres)
             * @param residuals Pointer to the residual vector (see ceres)
             * @param jacobians Pointer to the Jacobians (see ceres)
             * @return success of th evaluation.
             */
            virtual bool Evaluate(double const* const * parameters, double* residuals,
                                  double** jacobians) const;

            /**
             * @brief This evaluates the error term and additionally computes
             *        the Jacobians in the minimal internal representation.
             * @param parameters Pointer to the parameters (see ceres)
             * @param residuals Pointer to the residual vector (see ceres)
             * @param jacobians Pointer to the Jacobians (see ceres)
             * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
             * @return Success of the evaluation.
             */
            bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                              double* residuals, double** jacobians,
                                              double** jacobiansMinimal) const;

            // sizes
            /// \brief Residual dimension.
            size_t residualDim() const {
                return kNumResiduals;
            }

            /// \brief Number of parameter blocks.
            virtual size_t parameterBlocks() const {
                return parameter_block_sizes().size();
            }

            /// \brief Dimension of an individual parameter block.
            /// @param[in] parameterBlockId ID of the parameter block of interest.
            /// \return The dimension.
            size_t parameterBlockDim(size_t parameterBlockId) const {
                return base_t::parameter_block_sizes().at(parameterBlockId);
            }

            /// @brief Return parameter block type as string
            virtual std::string typeInfo() const {
                return "ImuError";
            }

        protected:
            // parameters
            okvis::ImuParameters imuParameters_; ///< The IMU parameters.

            // measurements
            okvis::ImuMeasurementDeque imuMeasurements_; ///< The IMU measurements used. Must be spanning t0_ - t1_.

            // times
            Time t0_; ///< The start time (i.e. time of the first set of states).
            Time t1_; ///< The end time (i.e. time of the sedond set of states).

            // preintegration stuff. the mutable is a TERRIBLE HACK, but what can I do.
            mutable std::mutex preintegrationMutex_; //< Protect access of intermediate results.
            // increments (initialise with identity)
            mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
            mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
            mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
            mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
            mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

            // cross matrix accumulatrion
            mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

            // sub-Jacobians
            mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
            mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
            mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

            /// \brief The Jacobian of the increment (w/o biases).
            mutable Eigen::Matrix<double,15,15> P_delta_ = Eigen::Matrix<double,15,15>::Zero();

            /// \brief Reference biases that are updated when called redoPreintegration.
            mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

            mutable bool redo_ = true; ///< Keeps track of whether or not this redoPreintegration() needs to be called.
            mutable int redoCounter_ = 0; ///< Counts the number of preintegrations for statistics.

            // information matrix and its square root
            mutable information_t information_; ///< The information matrix for this error term.
            mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.

        };

    }  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_IMUERROR_HPP_ */