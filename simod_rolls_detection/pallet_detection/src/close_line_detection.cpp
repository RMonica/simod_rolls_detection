#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <functional>
#include <random>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "close_line_detection.h"

template <typename T>
inline T SQR(const T & t) {return t * t; }

CloseLineDetection::CloseLineDetection(const Config & config,
                     LogFunction log_function,
                     PublishImageFunction publish_image_function):
      m_config(config), m_log{log_function}, m_publish_image{publish_image_function}
{

}

CloseLineDetection::StringVector CloseLineDetection::GetImagePublisherNames()
{
  StringVector result = {
    "canny_basic",
    "lines_basic",
    "best_line_basic",
    "blurredGray",
    "canny",
    "linesp",
    "correlation_mask",
    "correlation",
    "correlation_overlay",
    "blurredGray",
    "circle_canny",
    "circles",
    "blurredLines",
    "initial_guess_mask",
    "correlation_total",
    "max_line",
    "input",
    "reprojected",
    "blurredGray_basic"};

  return result;
}

cv::Mat CloseLineDetection::DrawCircles(const cv::Mat &img, const std::vector<cv::Vec3f> & circles, const cv::Scalar color)
{
  cv::Mat result = img.clone();
  for (size_t i = 0; i < circles.size(); i++)
  {
    cv::Vec3f c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    cv::circle(result, center, c[2], color, 3, cv::LINE_AA);
  }
  return result;
}

// returns origin ox, oy, and direction dx, dy
CloseLineDetection::ExtractedLine CloseLineDetection::DetectLinesBasic(const cv::Mat & image,
                               const float distance,
                               const float focal_length,
                               const Eigen::Vector2f & initial_guess_pixel_camera,
                               const cv::Mat & reprojection_mask,
                               Vector4fVector & result_lines)
{
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat blurredGray = gray;

  const int gaussian_blur_window_size = m_config.basic_gaussian_blur_half_window_size * 2 + 1;
  cv::GaussianBlur(gray, blurredGray, cv::Size(gaussian_blur_window_size, gaussian_blur_window_size), 0, 0);

  m_publish_image("blurredGray_basic", blurredGray, "mono8");

  const int lowThreshold = m_config.basic_canny_threshold;
  const int ratio = 2;
  const int kernel_size = 3;
  cv::Mat canny;
  cv::Canny(blurredGray, canny, lowThreshold, lowThreshold*ratio, kernel_size);

  cv::Mat initial_guess_mask = GetInitialGuessMask(image.cols, image.rows, initial_guess_pixel_camera,
                                                   focal_length, distance);
  initial_guess_mask = reprojection_mask.mul(initial_guess_mask);
  for (int y = 0; y < image.rows; y++)
    for (int x = 0; x < image.cols; x++)
    {
      const int W = 2;
      // search a small window around
      // to clear discontinuities due to masks
      for (int dy = -W; dy <= W; dy++)
        for (int dx = -W; dx <= W; dx++)
        {
          int nx = x + dx;
          int ny = y + dy;
          if (nx < 0 || ny < 0)
            continue;
          if (nx >= image.cols || ny >= image.rows)
            continue;

          if (initial_guess_mask.at<float>(ny, nx) == 0.0f)
            canny.at<uint8>(y, x) = 0;
        }
    }

  m_publish_image("canny_basic", canny, "mono8");

  std::vector<cv::Vec2f> lines;
  const float min_angle = CV_PI/2.0f - m_config.line_max_angle;
  const float max_angle = CV_PI/2.0f + m_config.line_max_angle;
  cv::HoughLines(canny, lines, 1, CV_PI/360, m_config.basic_hough_threshold, 0, 0, min_angle, max_angle);

  m_log(1, "HoughLines: Found " + std::to_string(lines.size()) + " lines.");

  result_lines.clear();
  for(size_t i = 0; i < lines.size(); i++)
  {
    float rho = lines[i][0], theta = lines[i][1];
    double a = std::cos(theta), b = std::sin(theta);
    double x0 = a * rho, y0 = b * rho;

    Eigen::Vector4f l;
    l.x() = b; // swap because vector [a, b] is perpendicular to the line
    l.y() = -a;
    l.z() = x0;
    l.w() = y0;
    result_lines.push_back(l);

    // [x, y]^T = [l.x l.y]^T t + [l.z l.w]^T
  }

  cv::Mat lines_img = image.clone();
  for(size_t i = 0; i < result_lines.size(); i++)
  {
    const Eigen::Vector4f & l = result_lines[i];
    cv::Point pt1, pt2;
    float a = l[0];
    float b = l[1];
    float x0 = l[2];
    float y0 = l[3];
    pt1.x = cvRound(x0 + 2000 * (a));
    pt1.y = cvRound(y0 + 2000 * (b));
    pt2.x = cvRound(x0 - 2000 * (a));
    pt2.y = cvRound(y0 - 2000 * (b));

    cv::line(lines_img, pt1, pt2, cv::Scalar(0,0,255), 1, cv::LINE_AA);
  }

  m_publish_image("lines_basic", lines_img, "bgr8");

  if (result_lines.empty())
  {
    m_log(3, "DetectLinesBasic: No lines found, exiting.");
    return ExtractedLine();
  }

  std::mt19937 random_generator(m_config.random_seed);
  std::uniform_int_distribution<int> random_distribution(0, result_lines.size() - 1);

  const float max_ransac_distance = m_config.basic_max_ransac_distance * focal_length / distance;

  // RANSAC
  const int RANSAC_ITERATIONS = 10;
  IntVector best_consensus;
  Eigen::Vector4f best_line = Eigen::Vector4f::Zero();
  for (int iter = 0; iter < RANSAC_ITERATIONS; iter++)
  {
    const int selected = random_distribution(random_generator);

    const Eigen::Vector4f sel_l = result_lines[selected];
    const float sel_a = sel_l[0];
    const float sel_b = sel_l[1];
    const float sel_x0 = sel_l[2];
    const float sel_y0 = sel_l[3];

    IntVector consensus;

    for (int i = 0; i < int(result_lines.size()); i++)
    {
      const Eigen::Vector4f l = result_lines[i];
      const float a = l[0];
      const float b = l[1];
      const float x0 = l[2];
      const float y0 = l[3];

      bool found_above_th = false;
      for (int x = 0; x < image.cols; x++)
      {
        const float sel_y = (x - sel_x0) / sel_a * sel_b + sel_y0;
        const float y = (x - x0) / a * b + y0;

        if (std::abs(sel_y - y) > max_ransac_distance)
          found_above_th = true;
      }

      if (!found_above_th)
      {
        consensus.push_back(i);
      }

      if (consensus.size() > best_consensus.size())
      {
        best_consensus = consensus;
        best_line = l;
      }
    }
  }

  m_log(1, "DetectLinesBasic: Best consensus is " + std::to_string(best_consensus.size()));

  Eigen::Vector4f avg_line = Eigen::Vector4f::Zero();
  for (int i : best_consensus)
    avg_line += result_lines[i];
  if (best_consensus.size())
    avg_line /= float(best_consensus.size());

  cv::Mat best_line_img = image.clone();
  ExtractedLine result;
  {
    float a = avg_line[0];
    float b = avg_line[1];
    float x0 = avg_line[2];
    float y0 = avg_line[3];

    cv::Point pt1(0, (0 - x0) / a * b + y0);
    cv::Point pt2(image.cols - 1, (image.cols - 1 - x0) / a * b + y0);
    cv::line(best_line_img, pt1, pt2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

    result.main_points.push_back(pt1);
    result.main_points.push_back(pt2);

    result.points.push_back(pt1);
    result.points.push_back(pt2);
  }

  m_publish_image("best_line_basic", best_line_img, "bgr8");

  return result;
}

cv::Mat CloseLineDetection::DetectLines(const cv::Mat & image)
{
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat blurredGray = gray;

  //cv::medianBlur(gray, blurredGray, 11);
  const int gaussian_blur_window_size = m_config.gaussian_blur_half_window_size * 2 + 1;
  cv::GaussianBlur(gray, blurredGray, cv::Size(gaussian_blur_window_size, gaussian_blur_window_size), 0, 0);

  m_publish_image("blurredGray", blurredGray, "mono8");

  const int lowThreshold = 10;
  const int ratio = 2;
  const int kernel_size = 3;
  cv::Mat canny;
  cv::Canny(blurredGray, canny, lowThreshold, lowThreshold*ratio, kernel_size);

  m_publish_image("canny", canny, "mono8");

  cv::Mat linesp_img = image.clone();
  cv::Mat linesp_grayscale_img(image.rows, image.cols, CV_32FC1);
  linesp_grayscale_img = 0.0f;
  {
    const int threshold = 75;
    const int min_line_length = 100;
    const int max_line_gap = 50;
    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    cv::HoughLinesP(canny, linesP, 1, CV_PI/180, threshold, min_line_length, max_line_gap); // runs the actual detection

    m_log(1, "HoughLinesP: Found " + std::to_string(linesP.size()) + " segments.");

    for(size_t i = 0; i < linesP.size(); i++ )
    {
      cv::Vec4i l = linesP[i];

      float angle = std::atan2(l[3] - l[1], l[2] - l[0]);
      if (angle > CV_PI / 2.0f)
        angle -= CV_PI;
      if (angle < -CV_PI / 2.0f)
        angle += CV_PI;
      if (std::abs(angle) > m_config.line_max_angle)
        continue;

      cv::line(linesp_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,255,0), 3, cv::LINE_AA);
      cv::line(linesp_grayscale_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(1.0f), 1, cv::LINE_AA);
    }
  }

  m_publish_image("linesp", linesp_img, "bgr8");

  return linesp_grayscale_img;
}

CloseLineDetection::CorrelationMask CloseLineDetection::ShiftCorrelationMask(const CorrelationMask & mask, const float fraction)
{
  CorrelationMask new_mask(mask);

  const int height = mask.mask.rows;
  const int width = mask.mask.cols;
  const int pixels = std::round(mask.mask.cols * fraction);

  mask.mask(cv::Rect(0, 0, pixels, height)).copyTo(new_mask.mask(cv::Rect(width - pixels, 0, pixels, height)));
  mask.mask(cv::Rect(pixels, 0, width - pixels, height)).copyTo(new_mask.mask(cv::Rect(0, 0, width - pixels, height)));
  // new_mask.origin.x += pixels;
  // if (new_mask.origin.x > width)
  //   new_mask.origin.x -= width;

  return new_mask;
}

cv::Mat CloseLineDetection::ComputeCorrelation(const cv::Mat & grayscale, const CorrelationMask & correlation_mask)
{
  cv::Mat rescaled_gray;
  cv::resize(grayscale, rescaled_gray, cv::Size(), correlation_mask.scale, correlation_mask.scale, cv::INTER_LINEAR);

  // rescaled image is too small for this correlation mask
  // at least 2 correlation masks must fit
  if (rescaled_gray.cols < correlation_mask.mask.cols*2 || rescaled_gray.rows < correlation_mask.mask.rows*2)
  {
    cv::Mat ecorrelation(grayscale.rows, grayscale.cols, CV_32FC1);
    ecorrelation.setTo(0.0f);
    return ecorrelation;
  }

  cv::Mat correlation;
  cv::matchTemplate(rescaled_gray, correlation_mask.mask, correlation, cv::TM_CCOEFF_NORMED);
  //cv::matchTemplate(rescaled_gray, correlation_mask, correlation, cv::TM_CCORR_NORMED);
  //cv::matchTemplate(rescaled_gray, correlation_mask, correlation, cv::TM_SQDIFF_NORMED);
  //cv::matchTemplate(rescaled_gray, correlation_mask, correlation, cv::TM_SQDIFF);

  cv::Mat inv_rescaled_gray = 255 - rescaled_gray;
  cv::Mat inv_correlation_mask = 255 - correlation_mask.mask;
  cv::Mat inv_correlation;
  cv::matchTemplate(inv_rescaled_gray, inv_correlation_mask, inv_correlation, cv::TM_CCOEFF_NORMED);
  correlation = (correlation + inv_correlation) / 2.0f;

  cv::Mat ecorrelation(rescaled_gray.rows, rescaled_gray.cols, CV_32FC1);
  ecorrelation.setTo(0.0f);
  correlation.copyTo(ecorrelation(cv::Rect(correlation_mask.origin.x, correlation_mask.origin.y, correlation.cols, correlation.rows)));
  cv::resize(ecorrelation, ecorrelation, cv::Size(grayscale.cols, grayscale.rows), 0, 0, cv::INTER_LINEAR);

  return ecorrelation;
}

cv::Mat CloseLineDetection::DoCorrelation(const cv::Mat & image, const float distance, const float focal_length)
{
  const float roll_diameter = m_config.roll_diameter;

  m_log(1, "DoCorrelation start.");

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  const cv::Scalar WHITE = cv::Scalar(255);

  const int correlation_mask_size = 50;

  const float expected_pixels = roll_diameter * focal_length / distance;
  const float SCALING = correlation_mask_size / expected_pixels;
  m_log(1, "Computed scaling: " + std::to_string(SCALING));

  std::vector<CorrelationMask> correlation_masks;

  {
    cv::Mat correlation_mask3(correlation_mask_size * 1 / 2, correlation_mask_size, CV_8UC1);
    correlation_mask3 = 0;
    cv::circle(correlation_mask3, cv::Point(correlation_mask_size / 2 - 1, correlation_mask_size / 2 - 1), correlation_mask_size / 2,
               WHITE, cv::FILLED, cv::LINE_8, 0);
    // cv::circle(correlation_mask3, cv::Point(correlation_mask_size * 1 / 3 - 1, correlation_mask_size * 1 / 3),
    //            correlation_mask_size * 1 / 3,
    //            WHITE, cv::FILLED, cv::LINE_8, 0);
    // cv::circle(correlation_mask3, cv::Point(correlation_mask_size * 2 / 3 + 1, correlation_mask_size * 1 / 3),
    //            correlation_mask_size * 1 / 3,
    //            WHITE, cv::FILLED, cv::LINE_8, 0);
    // cv::rectangle(correlation_mask3, cv::Point(correlation_mask_size * 1 / 3 - 1, 0),
    //                                  cv::Point(correlation_mask_size * 2 / 3 + 1, correlation_mask_size * 1 / 3),
    //                                  WHITE, cv::FILLED);
    cv::Point correlation_mask_origin3(correlation_mask_size / 2 - 1, 0);

    CorrelationMask mstruct = CorrelationMask(correlation_mask3, correlation_mask_origin3, SCALING);
    correlation_masks.push_back(mstruct);
    correlation_masks.push_back(ShiftCorrelationMask(mstruct, 0.25f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct, 0.5f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct, 0.75f));

    m_publish_image("correlation_mask", mstruct.mask, "mono8");

    cv::Mat correlation_mask4;
    cv::flip(correlation_mask3, correlation_mask4, 0);
    cv::Point correlation_mask_origin4(correlation_mask_size / 2 - 1, correlation_mask_size / 2 - 1);

    CorrelationMask mstruct2 = CorrelationMask(correlation_mask4, correlation_mask_origin4, SCALING);

    correlation_masks.push_back(mstruct2);
    correlation_masks.push_back(ShiftCorrelationMask(mstruct2, 0.25f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct2, 0.5f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct2, 0.75f));

  }

  {
    cv::Mat correlation_mask3(correlation_mask_size, correlation_mask_size, CV_8UC1);
    correlation_mask3 = 0;
    cv::circle(correlation_mask3, cv::Point(correlation_mask_size / 2 - 1, correlation_mask_size / 2 - 1),
               correlation_mask_size / 2 + 0.5,
               WHITE, cv::FILLED, cv::LINE_8, 0);
    cv::circle(correlation_mask3, cv::Point(correlation_mask_size / 2 - 1, correlation_mask_size / 2 - 1),
               correlation_mask_size / 6 + 0.5,
               cv::Scalar(127), cv::FILLED, cv::LINE_8, 0);
    cv::Point correlation_mask_origin3(correlation_mask_size / 2 - 1, 0);

    CorrelationMask mstruct = CorrelationMask(correlation_mask3, correlation_mask_origin3, SCALING);
    correlation_masks.push_back(mstruct);
    correlation_masks.push_back(ShiftCorrelationMask(mstruct, 0.25f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct, 0.5f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct, 0.75f));

    cv::Point correlation_mask_origin4(correlation_mask_size / 2 - 1, correlation_mask_size - 1);

    CorrelationMask mstruct2 = CorrelationMask(correlation_mask3, correlation_mask_origin4, SCALING);
    correlation_masks.push_back(mstruct2);
    correlation_masks.push_back(ShiftCorrelationMask(mstruct2, 0.25f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct2, 0.5f));
    correlation_masks.push_back(ShiftCorrelationMask(mstruct2, 0.75f));
  }

  cv::Mat correlation;

  for (const CorrelationMask & cm : correlation_masks)
  {
    cv::Mat ecorrelation1 = ComputeCorrelation(gray, cm);
    if (correlation.empty())
      correlation = ecorrelation1;
    else
      correlation += ecorrelation1;
  }
  correlation = correlation / correlation_masks.size();

  cv::Mat grayscale_correlation = FloatImageToGrayscale(correlation);

  cv::Mat image_correlation_overlay;
  std::vector<cv::Mat> splitted(3);
  cv::split(image, splitted);
  splitted[1] += grayscale_correlation - 127;
  cv::merge(splitted, image_correlation_overlay);

  m_publish_image("correlation", grayscale_correlation, "mono8");

  m_publish_image("correlation_overlay", image_correlation_overlay, "bgr8");

  return correlation;
}

cv::Mat CloseLineDetection::DoCircles(const cv::Mat & image)
{
  m_log(1, "DoCircles start.");

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat blurredGray;
  cv::GaussianBlur(gray, blurredGray, cv::Size(41, 41), 0, 0);

  m_publish_image("blurredGray", blurredGray, "mono8");

  const int lowThreshold = 10;
  const int ratio = 2;
  const int kernel_size = 3;
  cv::Mat canny;
  cv::Canny(blurredGray, canny, lowThreshold, lowThreshold*ratio, kernel_size);

  m_publish_image("circle_canny", canny, "mono8");

  std::vector<cv::Vec3f> circles;

  const float minRadius = 250.0 / 2.0;
  const float maxRadius = 400.0 / 2.0;
  const float minDist = 200.0;
  cv::HoughCircles(blurredGray, circles, cv::HOUGH_GRADIENT, 1,
                   minDist, // minimum distance between the circles
                   30, 20, // p1 & p2 Canny threshold and circle perfectness
                   minRadius, maxRadius // minimum radius and maximum radius
                   );

  m_log(1, "Found " + std::to_string(circles.size()) + " circles.");

  cv::Mat image_with_circles = DrawCircles(image, circles, cv::Scalar(255, 0, 0));

  m_publish_image("circles", image_with_circles, "bgr8");

  return cv::Mat();
}

cv::Mat CloseLineDetection::FloatImageToGrayscale(const cv::Mat fi)
{
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::minMaxLoc(fi, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

  cv::Mat grayscale_correlation;
  fi.convertTo(grayscale_correlation, CV_8UC1, 255.0 / maxVal);

  return grayscale_correlation;
}

cv::Mat CloseLineDetection::NormalizeImage(const cv::Mat fi)
{
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::minMaxLoc(fi, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

  return fi / maxVal;
}

float CloseLineDetection::Gauss(const float x, const float mean, const float stddev)
{
  return std::exp(-SQR(x - mean)/SQR(stddev));
}

CloseLineDetection::IntVector CloseLineDetection::NonMaximumSuppression(const IntVector & column_maxes_in, const FloatVector & column_maxes_v)
{
  IntVector column_maxes = column_maxes_in;
  BoolVector valid(column_maxes.size(), true);

  while (std::find(valid.begin(), valid.end(), true) != valid.end())
  {
    float max = -1.0f;
    int max_x = -1;
    for (int x = 0; x < int(column_maxes.size()); x++)
    {
      if (valid[x] && column_maxes[x] != -1 && column_maxes_v[x] > max)
      {
        max_x = x;
        max = column_maxes_v[x];
      }
    }

    {
      const int x = max_x;
      if (x == -1)
        break; // should never happen

      const int WINDOW = m_config.non_maximum_suppression_window;
      for (int dx = -WINDOW; dx <= WINDOW; dx++)
      {
        int nx = x + dx;
        if (nx < 0 || nx >= int(column_maxes.size()))
          continue;

        valid[nx] = false;

        if (dx != 0)
          column_maxes[nx] = -1;
      }
    }
  }

  return column_maxes;
}

cv::Mat CloseLineDetection::GetInitialGuessMask(const int width, const int height, const Eigen::Vector2f & initial_guess_pixel_camera,
                            const float focal_length, const float distance)
{
  cv::Mat initial_guess_mask(height, width, CV_32FC1);
  initial_guess_mask.setTo(0.0);

  const float initial_guess_width = m_config.initial_guess_window_size_x * focal_length / distance;
  const float initial_guess_height = m_config.initial_guess_window_size_y * focal_length / distance;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
    {
      const int dy = y - initial_guess_pixel_camera.y();
      const int dx = x - initial_guess_pixel_camera.x();
      if (std::abs(dx) <= initial_guess_width && std::abs(dy) <= initial_guess_height)
        initial_guess_mask.at<float>(y, x) = 1.0f;
      // Gauss(y, initial_guess_pixel_camera, initial_guess_width * 2.0f);
    }

  return initial_guess_mask;
}

CloseLineDetection::ExtractedLine CloseLineDetection::ExtractLineFromImage(const cv::Mat & image,
                                   const float distance,
                                   const float focal_length,
                                   const Eigen::Vector2f & initial_guess_pixel_camera,
                                   const cv::Mat & reprojection_mask)
{
  cv::Mat lines_img = DetectLines(image);
  cv::Mat blurredLines;
  cv::GaussianBlur(lines_img, blurredLines, cv::Size(51, 51), 0, 0);
  //blurredLines = NormalizeImage(blurredLines);

  m_publish_image("blurredLines", FloatImageToGrayscale(blurredLines), "mono8");

  cv::Mat correlation = DoCorrelation(image, distance, focal_length);
  correlation = NormalizeImage(correlation);

  cv::Mat initial_guess_mask = GetInitialGuessMask(image.cols, image.rows, initial_guess_pixel_camera,
                                                  focal_length, distance);
  initial_guess_mask = reprojection_mask.mul(initial_guess_mask);
  m_publish_image("initial_guess_mask", FloatImageToGrayscale(initial_guess_mask), "mono8");

  cv::Mat total = (blurredLines * m_config.correlation_lines_weight + correlation).mul(initial_guess_mask);

  for (int y = 0; y < total.rows; y++)
    for (int x = 0; x < total.cols; x++)
      if (total.at<float>(y, x) < m_config.min_correlation_threshold)
        total.at<float>(y, x) = 0.0f;

  cv::Mat grayscale_total = FloatImageToGrayscale(total);

  m_publish_image("correlation_total", grayscale_total, "mono8");

  std::vector<int> column_maxes(image.cols);
  std::vector<float> column_maxes_v(image.cols);

  cv::Mat max_img = image.clone();
  for (int x = 0; x < total.cols; x++)
  {
    int max_y = -1;
    float max_v = 0.0f;
    for (int y = 0; y < total.rows; y++)
    {
      if (total.at<float>(y, x) > max_v)
      {
        max_v = total.at<float>(y, x);
        max_y = y;
      }
    }

    column_maxes[x] = max_y;
    column_maxes_v[x] = max_v;
  }

  {
    std::vector<int> f_column_maxes = column_maxes;

    // remove outliers using median filter
    for (int i = 0; i < int(f_column_maxes.size()); i++)
    {
      if (f_column_maxes[i] == -1)
        continue;

      const int WINDOW = m_config.median_filter_window;
      std::vector<int> arr(2*WINDOW+1);
      for (int di = -WINDOW; di <= WINDOW; di++)
      {
        int ni = i + di;
        if (ni < 0)
          ni = 0;
        if (ni >= int(f_column_maxes.size()))
          ni = f_column_maxes.size() - 1;

        arr[di + WINDOW] = column_maxes[ni];
      }
      std::sort(arr.begin(), arr.end());
      f_column_maxes[i] = arr[WINDOW];
    }

    // filter using Gaussian
    column_maxes = f_column_maxes;
    for (int i = 0; i < int(f_column_maxes.size()); i++)
    {
      if (f_column_maxes[i] == -1)
        continue;

      const int STDDEV = 5;
      float counter = 0.0f;
      float sum = 0.0f;
      for (int di = -STDDEV * 2; di <= STDDEV * 2; di++)
      {
        int ni = i + di;
        if (ni < 0)
          ni = 0;
        if (ni >= int(f_column_maxes.size()))
          ni = f_column_maxes.size() - 1;
        if (f_column_maxes[ni] == -1)
          continue;

        counter += Gauss(ni, i, STDDEV);
        sum += Gauss(ni, i, STDDEV) * f_column_maxes[ni];
      }
      if (counter > 0.0001f)
        f_column_maxes[i] = sum / counter;
    }

    column_maxes = f_column_maxes;
  }

  const IntVector suppressed_column_maxes = NonMaximumSuppression(column_maxes, column_maxes_v);

  std::vector<cv::Point> points;
  for (int x = 0; x < int(suppressed_column_maxes.size()); x++)
  {
    if (suppressed_column_maxes[x] == -1)
      continue;

    max_img.at<cv::Vec3b>(suppressed_column_maxes[x], x)[2] = 255;
    points.push_back(cv::Point(x, suppressed_column_maxes[x]));
  }

  m_log(1, "Line strip has " + std::to_string(points.size()) + " points.");

  for (size_t i = 1; i < points.size(); i++)
  {
    const cv::Point this_pt = points[i];
    const cv::Point prev_pt = points[i - 1];
    cv::line(max_img, prev_pt, this_pt, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
  }

  m_publish_image("max_line", max_img, "bgr8");

  m_log(1, "Done.");

  ExtractedLine result;
  result.main_points = points;

  for (size_t x = 0; x < column_maxes.size(); x++)
    if (column_maxes[x] != -1)
      result.points.push_back(cv::Point(x, column_maxes[x]));

  return result;
}

void CloseLineDetection::DoReprojection(const cv::Mat & image, const Eigen::Affine3f & camera_pose, const Intrinsics & intrinsics,
                    const float initial_guess_y, const float layer_height,
                    Eigen::Affine3f & reproj_camera_pose, cv::Mat & reprojected_image,
                    cv::Mat & reprojected_mask)
{
  reproj_camera_pose.linear() = (Eigen::AngleAxisf(M_PI, Eigen::Vector3f(1.0f, 0.0f, 0.0f)) *
                                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f(0.0f, 0.0f, 1.0f))).matrix();
  reproj_camera_pose.translation() = camera_pose.translation();
  reproj_camera_pose.translation().y() = initial_guess_y;

  if (reproj_camera_pose.translation().z() < layer_height + m_config.min_virtual_camera_distance)
    reproj_camera_pose.translation().z() = layer_height + m_config.min_virtual_camera_distance;

  reprojected_image = cv::Mat(image.rows, image.cols, CV_8UC3);
  reprojected_image.setTo(cv::Scalar(0, 0, 0));
  reprojected_mask = cv::Mat(image.rows, image.cols, CV_32FC1);
  reprojected_mask.setTo(0.0f);

  for (int y = 0; y < reprojected_image.rows; y++)
    for (int x = 0; x < reprojected_image.cols; x++)
    {
      const Eigen::Vector3f d((x - intrinsics.cx) / intrinsics.fx,
                              (y - intrinsics.cy) / intrinsics.fy,
                              1.0f); // ray direction in camera coords
      const Eigen::Vector3f dw = reproj_camera_pose.linear() * d; // ray direction in world coords
      const Eigen::Vector3f o = reproj_camera_pose.translation(); // ray origin in world coords
      // intersect with plane z = layer_height
      const float t = (layer_height - o.z()) / dw.z();
      if (std::isnan(t) || t < 0.001f)
        continue; // no intersection with plane

      const Eigen::Vector3f ptw = o + dw * t;
      const Eigen::Vector3f ptn = camera_pose.inverse() * ptw;

      const int xn = std::round(ptn.x() / ptn.z() * intrinsics.fx + intrinsics.cx);
      const int yn = std::round(ptn.y() / ptn.z() * intrinsics.fy + intrinsics.cy);
      if (xn < 0 || yn < 0)
        continue;
      if (xn >= image.cols || yn >= image.rows)
        continue;

      reprojected_image.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(yn, xn);
      reprojected_mask.at<float>(y, x) = 1.0f;
    }
}

CloseLineDetection::Vector3fVector CloseLineDetection::Run(const cv::Mat & image,
         const Intrinsics & intrinsics,
         const Eigen::Affine3f & camera_pose,
         const float layer_height,
         const float initial_guess_x,
         const float initial_guess_y
         )
{
  m_publish_image("input", image, "bgr8");

  Eigen::Affine3f reproj_camera_pose;

  cv::Mat reprojected_image;
  cv::Mat reprojected_mask;

  DoReprojection(image, camera_pose, intrinsics, initial_guess_y, layer_height,
                 reproj_camera_pose, reprojected_image, reprojected_mask);

  m_publish_image("reprojected", reprojected_image, "bgr8");

  const float focal_length = (intrinsics.fx + intrinsics.fy) / 2.0f;

  float distance;
  distance = reproj_camera_pose.translation().z() - layer_height;
  m_log(1, "computed camera distance: " + std::to_string(distance));
  if (distance < 0.0f)
  {
    m_log(3, "camera distance is " + std::to_string(distance) + " < 0: camera cannot be below pallet!");
    return Vector3fVector();
  }

  const Eigen::Vector3f initial_guess_camera = reproj_camera_pose.inverse() * Eigen::Vector3f(initial_guess_x, initial_guess_y, 0.0f);
  const float initial_guess_y_camera = initial_guess_camera.y();
  const float initial_guess_x_camera = initial_guess_camera.x();
  Eigen::Vector2f initial_guess_pixel_camera;
  initial_guess_pixel_camera.y() = initial_guess_y_camera * intrinsics.fy / distance + intrinsics.cy;
  initial_guess_pixel_camera.x() = initial_guess_x_camera * intrinsics.fx / distance + intrinsics.cx;

  ExtractedLine line;

  if (!m_config.mode_basic)
  {
    line = ExtractLineFromImage(reprojected_image, distance, focal_length, initial_guess_pixel_camera,
                                reprojected_mask);
  }
  else
  {
    Vector4fVector all_basic_lines;
    line = DetectLinesBasic(reprojected_image, distance, focal_length, initial_guess_pixel_camera,
                            reprojected_mask, all_basic_lines);
  }

  Vector3fVector points3d;
  for (const cv::Point & pt : line.main_points)
  {
    const Eigen::Vector3f lpt((pt.x - intrinsics.cx) * distance / intrinsics.fx,
                              (pt.y - intrinsics.cy) * distance / intrinsics.fy,
                              distance);
    const Eigen::Vector3f gpt = reproj_camera_pose * lpt;
    points3d.push_back(gpt);
  }

  return points3d;
}
