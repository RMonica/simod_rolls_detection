#ifndef FILTERS_H
#define FILTERS_H

#include "simod_rolls_detection/utils_rot.h"
#include <vector>

class LowPassFilter
{
 public:
  /**
   * @brief LowPassFilter constructor.
   * @param[in] cutoff_freq [Hz] Cutoff frequency of low-pass filter.
   * @param[in] sampling_time_sec Sampling time in seconds.
   */
  LowPassFilter(const double& cutoff_freq, const double& sampling_time_sec);
  /**
   * @brief LowPassFilter full constructor.
   * @param[in] cutoff_freq [Hz] Cutoff frequency of low-pass filter.
   * @param[in] sampling_time_sec Sampling time in seconds.
   * @param[in] init_value Initial value of the signal to be filtered.
   */
  LowPassFilter(const double& cutoff_freq, const double& sampling_time_sec,
                const Vector6d& init_value);
  virtual ~LowPassFilter() {}

  /**
   * @brief Set LP filter cutoff frequency.
   * @param[in] cutoff_freq [Hz] Cutoff frequency of low-pass filter.
   * @param[in] reset If _True_ reinitialize the filter, i.e. clear its history.
   */
  void SetCutoffFreq(const double& cutoff_freq, bool reset = false);
  /**
   * @brief Set LP filter sampling time.
   * @param[in] sampling_time_sec Sampling time in seconds.
   * @param[in] reset If _True_ reinitialize the filter, i.e. clear its history.
   */
  void SetSamplingTime(const double& sampling_time_sec, bool reset = false);
  /**
   * @brief Set LP filter time constant.
   * @param[in] time_const_sec Time constant in seconds.
   * @param[in] reset If _True_ reinitialize the filter, i.e. clear its history.
   */
  void SetTimeConstant(const double& time_const_sec, bool reset = false);

  /**
   * @brief Get LP filter cutoff frequency.
   * @return [Hz] Cutoff frequency of low-pass filter.
   */
  double GetCutoffFreq() const { return cutoff_freq_; }
  /**
   * @brief Get LP filter sampling time.
   * @return Sampling time in seconds.
   */
  double GetSamplingTime() const { return Ts_; }
  /**
   * @brief Get LP filter time constant.
   * @return LP filter time constant in seconds.
   */
  double GetTimeConstant() const { return Tf_; }
  /**
   * @brief Get latest filtered value.
   * @return Latest filtered value.h
   */
  Vector6d GetLatestFilteredValue() const { return filtered_value_; }

  /**
   * @brief Filter a new value of a signal.
   * @param[in] raw_value New signal value.
   * @return Filtered value.
   */
  virtual Vector6d Filter(const Vector6d& raw_value);
  /**
   * @brief Reset filter, clearing its history.
   */
  void Reset() { initialized_ = false; }

 protected:
  double cutoff_freq_; /**< Cut-off frequency in [Hz]. */
  double Ts_;          /**< Sampling time in [sec]. */
  double
    Tf_; /**< Filter time constant in [sec]. This is equal to @f$1/(2/pi f_{cutoff})@f$.*/
  double alpha_ = 0.0; /**< Time constants dependend filter factor. */

  bool initialized_;      /**< Boolean flag to check if filter is already initialized. */
  Vector6d filtered_value_; /**< Latest filtered value. */

  /**
   * @brief Update filter constants
   * @param[in] reset If _True_ reinitialize the filter, i.e. clear its history.
   */
  virtual void UpdateConstant(bool reset = false);
};

#endif // FILTERS_H
