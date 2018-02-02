// Copyright (c) 2015-2016, ETH Zurich, Wyss Zurich, Zurich Eye
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the ETH Zurich, Wyss Zurich, Zurich Eye nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL ETH Zurich, Wyss Zurich, Zurich Eye BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "common/buffer.hpp"
#include "common/file_utils.hpp"
#include "common/macros.hpp"
#include "common/types.hpp"
#include "common/transformation.hpp"

namespace ze {

//! Reading of various csv trajectory file formats (e.g. swe, euroc, pose).
//! Reads the result in a buffer that allows accessing the pose via the
//! timestamps.
using TrajectoryEle =     std::tuple<int64_t,Vector3,Vector4>;
using TupleVector = std::vector<std::tuple<int64_t,Vector3,Vector4>>;
class CSVTrajectory
{
public:
  ZE_POINTER_TYPEDEFS(CSVTrajectory);

  virtual void load(const std::string& in_file_path) = 0;
  virtual int64_t getTimeStamp(const std::string& ts_str) const;

protected:
  CSVTrajectory() = default;

  void readHeader(const std::string& in_file_path);
  Vector3 readTranslation(const std::vector<std::string>& items);
  Vector4 readOrientation(const std::vector<std::string>& items);
  Vector7 readPose(const std::vector<std::string>& items);

  std::ifstream in_str_;
  std::map<std::string, int> order_;
  std::string header_;
  const char delimiter_{','};
  size_t num_tokens_in_line_;
};

class PositionSeries : public CSVTrajectory
{
public:
  ZE_POINTER_TYPEDEFS(PositionSeries);

  PositionSeries();
  virtual void load(const std::string& in_file_path) override;
  const Buffer<real_t, 3>& getBuffer() const;
  Buffer<real_t, 3>& getBuffer();

protected:
  Buffer<real_t, 3> position_buf_;
};

class PoseSeries : public CSVTrajectory
{
public:
  ZE_POINTER_TYPEDEFS(PoseSeries);

  PoseSeries();

  virtual void load(const std::string& in_file_path) override;
  virtual const Buffer<real_t, 7>& getBuffer() const;
  virtual  TupleVector getVector() ;
  virtual Buffer<real_t, 7>& getBuffer();
  virtual StampedTransformationVector getStampedTransformationVector();

  static Transformation getTransformationFromVec7(const Vector7& data);

protected:
  Buffer<real_t, 7> pose_buf_;
    TupleVector  vector_;
};

class SWEResultSeries : public PoseSeries
{
public:
  SWEResultSeries();
};

class SWEGlobalSeries : public PoseSeries
{
public:
  SWEGlobalSeries();
};

class EurocResultSeries : public PoseSeries
{
public:
  EurocResultSeries();
    virtual void load(const std::string& in_file_path);
    void loadIMU(const std::string& in_file_path);
    std::vector<std::pair<int64_t,Vector3>> getLinearVelocities();
    std::vector<Vector3> getGyroBias();
    std::vector<Vector3> getAccelBias();



    std::vector<int64_t> getIMUts();
    std::vector<Vector3> getGyroMeas();
    std::vector<Vector3> getAccelMeas();

    std::map<int64_t,Vector3> getGyroMeasMap();
    std::map<int64_t,Vector3> getAccelMeasMap();

private:
    Eigen::Vector3d readLinearVelocitis(const std::vector<std::string>& items);
    Eigen::Vector3d readGyroBias(const std::vector<std::string>& items);
    Eigen::Vector3d readAccelBias(const std::vector<std::string>& items);

    Eigen::Vector3d readGyroMeas(const std::vector<std::string>& items);
    Eigen::Vector3d readAccelMeas(const std::vector<std::string>& items);

    void mapGyro(int64_t ts,const std::vector<std::string>& items);
    void mapAccel(int64_t ts,const std::vector<std::string>& items);



    std::vector<std::pair<int64_t ,Eigen::Vector3d>> linearVelocities_;
    std::vector<Eigen::Vector3d> gyroBias_;
    std::vector<Eigen::Vector3d> accelBias_;

    std::vector<int64_t > imu_ts_;
    std::vector<Eigen::Vector3d> gyroMeas_;
    std::vector<Eigen::Vector3d> accelMeas_;

    std::map<int64_t,Eigen::Vector3d> gyroMeasMap_;
    std::map<int64_t,Eigen::Vector3d> accelMeasMap_;


    std::ifstream imu_in_str_;
    std::map<std::string, int> imu_order_;
    std::string imu_header_;
    size_t imu_num_tokens_in_line_;

};

} // ze namespace
