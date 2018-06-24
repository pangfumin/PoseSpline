#ifndef NINEBOT_SLAM_DEMO_TIMER_H_
#define NINEBOT_SLAM_DEMO_TIMER_H_

#include <chrono>
namespace TimeStatistics {
typedef double real_t;
//! Seconds to nanoseconds.
inline constexpr int64_t secToNanosec(real_t seconds) {
  return static_cast<int64_t>(seconds * 1e9);
}

//! Milliseconds to nanoseconds.
inline constexpr int64_t millisecToNanosec(real_t milliseconds) {
  return static_cast<int64_t>(milliseconds * 1e6);
}

//! Nanoseconds to seconds.
//! WARNING: Don't pass very large or small numbers to this
//! function as the
//!          representability of the float value does not
//! capture nanoseconds
//!          resolution. The resulting accuracy will
//! be in the order of
//!          hundreds of nanoseconds.
inline constexpr real_t nanosecToSecTrunc(int64_t nanoseconds) {
  return static_cast<real_t>(nanoseconds) / 1e9;
}

//! Nanoseconds to milliseconds.
//! WARNING: Don't pass very large or very small numbers
//! to this function as the
//!          representability of the float value does not
//! capture nanoseconds
//!          resolution.
inline constexpr real_t nanosecToMillisecTrunc(int64_t nanoseconds) {
  return static_cast<real_t>(nanoseconds) / 1e6;
}

//! Return total nanoseconds from seconds and nanoseconds pair.
inline constexpr int64_t nanosecFromSecAndNanosec(int32_t sec, int32_t nsec) {
  return static_cast<int64_t>(sec) * 1000000000ll + static_cast<int64_t>(nsec);
}

//! Simple timing utilty.
class Timer {
 public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using ns = std::chrono::nanoseconds;
  using ms = std::chrono::milliseconds;

  //! The constructor directly starts the timer.
  Timer() : start_time_(Clock::now()) {}

  inline void start() {
    start_time_ = Clock::now();
  }

  inline int64_t stopAndGetNanoseconds() {
    const TimePoint end_time(Clock::now());
    ns duration = std::chrono::duration_cast<ns>(end_time - start_time_);
    return duration.count();
  }

  inline real_t stopAndGetMilliseconds() {
    return nanosecToMillisecTrunc(stopAndGetNanoseconds());
  }

  inline real_t stopAndGetSeconds() {
    return nanosecToSecTrunc(stopAndGetNanoseconds());
  }

 private:
  TimePoint start_time_;
};

}  // namespace TimeStatistics
#endif  // NINEBOT_SLAM_DEMO_TIMER_H_
