#include "simod_rolls_detection/filters.h"


LowPassFilter::LowPassFilter(const double& cutoff_freq, const double& sampling_time_sec)
  : cutoff_freq_(cutoff_freq), Ts_(sampling_time_sec), initialized_(false)
{
  Tf_ = 1. / (2. * M_PI * cutoff_freq_); // time constant
  UpdateConstant(true);
}

LowPassFilter::LowPassFilter(const double& cutoff_freq, const double& sampling_time_sec,
                             const Vector6d& init_value)
  : cutoff_freq_(cutoff_freq), Ts_(sampling_time_sec), initialized_(true),
    filtered_value_(init_value)
{
  Tf_ = 1. / (2. * M_PI * cutoff_freq_); // time constant
  UpdateConstant(false);
}

//--------- Public functions ---------------------------------------------------------//

void LowPassFilter::SetCutoffFreq(const double& cutoff_freq, bool reset /*=false*/)
{
  cutoff_freq_ = cutoff_freq;
  Tf_          = 1. / (2. * M_PI * cutoff_freq_); // time constant
  UpdateConstant(reset);
}

void LowPassFilter::SetSamplingTime(const double& sampling_time_sec,
                                    bool reset /*=false*/)
{
  Ts_ = sampling_time_sec;
  UpdateConstant(reset);
}

void LowPassFilter::SetTimeConstant(const double& time_const_sec, bool reset /*=false*/)
{
  Tf_          = time_const_sec;
  cutoff_freq_ = 1. / (2. * M_PI * Tf_);
  UpdateConstant(reset);
}

Vector6d LowPassFilter::Filter(const Vector6d& raw_value)
{
  if (initialized_)
    // y_k = (1 - a) * y_k-1 + a * x_k
    filtered_value_ += alpha_ * (raw_value - filtered_value_);
  else
  {
    filtered_value_ = raw_value;
    initialized_    = true;
  }
  return filtered_value_;
}

//--------- Private functions --------------------------------------------------------//

void LowPassFilter::UpdateConstant(bool reset /*= false*/)
{
  alpha_ = Ts_ / (Tf_ + Ts_);
  if (reset)
    Reset();
}
