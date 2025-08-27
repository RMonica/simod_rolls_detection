#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <functional>
#include <random>

#include <Eigen/Dense>
#include <Eigen/StdVector>

class CloseLineDetection
{
  public:
  typedef std::function<void(const int level, const std::string & message)> LogFunction;
  typedef std::function<void(const std::string & image_name, const cv::Mat & image, const std::string & encoding)> PublishImageFunction;

  typedef std::vector<int> IntVector;
  typedef std::vector<bool> BoolVector;
  typedef std::vector<float> FloatVector;
  typedef std::vector<std::string> StringVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> Vector3fVector;
  typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> Vector4fVector;

  typedef uint8_t uint8;

  struct Config
  {
    float roll_diameter = 0.11f;

    float min_correlation_threshold = 0.1f;
    float correlation_lines_weight = 0.1f;

    int non_maximum_suppression_window = 50;
    int median_filter_window = 100;

    int gaussian_blur_half_window_size = 20;

    float initial_guess_window_size_y = 0.2f; // meters
    float initial_guess_window_size_x = 1.0f; // meters

    float line_max_angle = M_PI / 12.0f;

    float min_virtual_camera_distance = 0.4f;

    int random_seed = 52; // fair d100 roll

    bool mode_basic = false;
    int basic_gaussian_blur_half_window_size = 10;
    int basic_canny_threshold = 10;
    int basic_hough_threshold = 100;
    float basic_max_ransac_distance = 0.03f;
  };

  struct ExtractedLine
  {
    std::vector<cv::Point> main_points;
    std::vector<cv::Point> points;
  };

  struct CorrelationMask
  {
    cv::Mat mask;
    cv::Point origin;
    float scale;

    CorrelationMask(const cv::Mat mask, const cv::Point origin, const float scale):
        mask(mask.clone()), origin(origin), scale(scale) {}
    CorrelationMask(const CorrelationMask & other):
        mask(other.mask.clone()), origin(other.origin), scale(other.scale) {}
    CorrelationMask & operator=(const CorrelationMask & other)
    {
      mask = other.mask.clone();
      origin = other.origin;
      scale = other.scale;
      return *this;
    }
  };

  struct Intrinsics
  {
    float cx;
    float cy;
    float fx;
    float fy;
  };

  CloseLineDetection(const Config & config,
                     LogFunction log_function,
                     PublishImageFunction publish_image_function);

  static StringVector GetImagePublisherNames();

  cv::Mat DrawCircles(const cv::Mat &img, const std::vector<cv::Vec3f> & circles, const cv::Scalar color);

  // returns origin ox, oy, and direction dx, dy
  ExtractedLine DetectLinesBasic(const cv::Mat & image,
                                 const float distance,
                                 const float focal_length,
                                 const Eigen::Vector2f & initial_guess_pixel_camera,
                                 const cv::Mat & reprojection_mask,
                                 Vector4fVector & result_lines);

  cv::Mat DetectLines(const cv::Mat & image);

  CorrelationMask ShiftCorrelationMask(const CorrelationMask & mask, const float fraction);

  cv::Mat ComputeCorrelation(const cv::Mat & grayscale, const CorrelationMask & correlation_mask);

  cv::Mat DoCorrelation(const cv::Mat & image, const float distance, const float focal_length);

  cv::Mat DoCircles(const cv::Mat & image);

  cv::Mat FloatImageToGrayscale(const cv::Mat fi);

  cv::Mat NormalizeImage(const cv::Mat fi);

  float Gauss(const float x, const float mean, const float stddev);

  IntVector NonMaximumSuppression(const IntVector & column_maxes_in, const FloatVector & column_maxes_v);

  cv::Mat GetInitialGuessMask(const int width, const int height, const Eigen::Vector2f & initial_guess_pixel_camera,
                              const float focal_length, const float distance);

  ExtractedLine ExtractLineFromImage(const cv::Mat & image,
                                     const float distance,
                                     const float focal_length,
                                     const Eigen::Vector2f & initial_guess_pixel_camera,
                                     const cv::Mat & reprojection_mask);

  void DoReprojection(const cv::Mat & image, const Eigen::Affine3f & camera_pose, const Intrinsics & intrinsics,
                      const float initial_guess_y, const float layer_height,
                      Eigen::Affine3f & reproj_camera_pose, cv::Mat & reprojected_image,
                      cv::Mat & reprojected_mask);

  Vector3fVector Run(const cv::Mat & image,
                     const Intrinsics & intrinsics,
                     const Eigen::Affine3f & camera_pose,
                     const float layer_height,
                     const float initial_guess_x,
                     const float initial_guess_y
                     );

  private:
  Config m_config;
  LogFunction m_log;
  PublishImageFunction m_publish_image;
};
