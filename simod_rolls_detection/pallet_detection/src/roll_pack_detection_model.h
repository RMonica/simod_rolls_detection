#ifndef ROLL_PACK_DETECTION_MODEL_H
#define ROLL_PACK_DETECTION_MODEL_H

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <cmath>
#include <array>
#include <string>
#include <memory>
#include <vector>

// Eigen
#include <Eigen/Dense>

template <int COUNT, typename T>
struct RollPackDetectionModel
{
  static constexpr int count = COUNT;
  static constexpr int HOMOGRAPHY_SIZE = 8;

  typedef std::array<double, HOMOGRAPHY_SIZE> HomographyArray;

  std::array<T, HOMOGRAPHY_SIZE> homography;
  std::array<T, COUNT - 1> translations_x;
  std::array<T, COUNT - 1> translations_y;

  static constexpr int GetNumParams() {return HOMOGRAPHY_SIZE + (COUNT - 1) * 2; }

  typedef std::array<Eigen::Vector2d, COUNT> Vector2dArray;
  typedef std::array<double, COUNT> DoubleArray;

  RollPackDetectionModel()
  {
    for (int i = 0; i < HOMOGRAPHY_SIZE; i++)
      homography[i] = 0.0;
    homography[0] = 1.0;
    homography[4] = 1.0;

    for (int i = 0; i < COUNT - 1; i++)
      translations_x[i] = 0;
    for (int i = 0; i < COUNT - 1; i++)
      translations_y[i] = 0;
  }

  explicit RollPackDetectionModel(const HomographyArray & h)
  {
    for (int i = 0; i < HOMOGRAPHY_SIZE; i++)
      homography[i] = h[i];

    for (int i = 0; i < COUNT - 1; i++)
      translations_x[i] = 0;
    for (int i = 0; i < COUNT - 1; i++)
      translations_y[i] = 0;
  }

  // from serialized array
  RollPackDetectionModel(const T * const x, const int offset)
  {
    for (int i = 0; i < HOMOGRAPHY_SIZE; i++)
      homography[i] = x[offset + i];

    for (int i = 0; i < COUNT - 1; i++)
      translations_x[i] = x[offset + HOMOGRAPHY_SIZE + i];
    for (int i = 0; i < COUNT - 1; i++)
      translations_y[i] = x[offset + HOMOGRAPHY_SIZE + (COUNT - 1) + i];
  }

  void Serialize(T * const x, const int offset) const
  {
    for (int i = 0; i < HOMOGRAPHY_SIZE; i++)
      x[offset + i] = homography[i];

    for (int i = 0; i < COUNT - 1; i++)
      x[offset + HOMOGRAPHY_SIZE + i] = translations_x[i];
    for (int i = 0; i < COUNT - 1; i++)
      x[offset + HOMOGRAPHY_SIZE + (COUNT - 1) + i] = translations_y[i];
  }

  void ApplyDeformToPoint(const int element_n, const T & ix, const T & iy, T & ox, T & oy) const
  {
    T tx(0.0); // total translation x for this element
    T ty(0.0);
    for (int i = 1; i <= element_n; i++)
    {
      tx += translations_x[i + (COUNT - 1) / 2 - 1];
    }

    for (int i = element_n; i < 0; i++)
    {
      tx += translations_x[i + COUNT / 2];
    }

    if (element_n > 0)
      ty = translations_y[element_n + (COUNT - 1) / 2 - 1];
    if (element_n < 0)
      ty = translations_y[element_n + COUNT / 2];

    ox = ix + tx;
    oy = iy + ty;
  }

  void ApplyToPoint(const int element_n, const T & ix, const T & iy, T & ox, T & oy) const
  {
    T tx(0.0); // total translation x for this element
    T ty(0.0);
    for (int i = 1; i <= element_n; i++)
    {
      tx += translations_x[i + (COUNT - 1) / 2 - 1];
    }

    for (int i = element_n; i < 0; i++)
    {
      tx += translations_x[i + COUNT / 2];
    }

    if (element_n > 0)
      ty = translations_y[element_n + (COUNT - 1) / 2 - 1];
    if (element_n < 0)
      ty = translations_y[element_n + COUNT / 2];

    ox = (homography[0] * (ix + tx) + homography[1] * (iy + ty) + homography[2]) / (homography[6] * (ix + tx) + homography[7] * (iy + ty) + T(1.0));
    oy = (homography[3] * (ix + tx) + homography[4] * (iy + ty) + homography[5]) / (homography[6] * (ix + tx) + homography[7] * (iy + ty) + T(1.0));
  }

  static int PointToElementN(const double image_width, const double image_height,
                             const double x, const double y)
  {
    const double rx = x / image_width;
    const int n = int(std::floor(rx * COUNT));
    return std::max(std::min(n, COUNT - 1), 0) - ((COUNT - 1) / 2);
  }

  // may return NAN
  void ApplyToPointInv(const double image_width, const double image_height,
                       const T ix, const T iy, T & ox, T & oy) const
  {
    // try all possible element_n until one is ok
    for (int i = -(COUNT - 1) / 2; i <= COUNT / 2; i++)
    {
      ApplyToPointInv(i, ix, iy, ox, oy);
      if (PointToElementN(image_width, image_height, ox, oy) == i)
        return; // found!
    }

    ox = oy = T(std::numeric_limits<double>::quiet_NaN());
  }

  void ApplyToPointInv(const int element_n, const T & ix, const T & iy, T & ox, T & oy) const
  {
    T tx(0.0); // total translation x for this element
    T ty(0.0);
    for (int i = 1; i <= element_n; i++)
    {
      tx += translations_x[i + (COUNT - 1) / 2 - 1];
    }

    for (int i = element_n; i < 0; i++)
    {
      tx += translations_x[i + COUNT / 2];
    }

    if (element_n > 0)
      ty = translations_y[element_n + (COUNT - 1) / 2 - 1];
    if (element_n < 0)
      ty = translations_y[element_n + COUNT / 2];

    const std::array<T, HOMOGRAPHY_SIZE> & h = homography;

    // (%i7) solve([x1 = (h0*x0 + h1*y0 + h2) / (h6*x0 + h7*y0 + 1), y1 = (h3*x0 + h4*y0 + h5) / (h6*x0 + h7*y0 + 1)], [x0, y0]);
    //                  h1 (y1 - h5) - h2 h7 y1 + (h5 h7 - h4) x1 + h2 h4
    //   (%o7) [[x0 = - ------------------------------------------------------------,
    //                  (- h0 h7 y1) + h1 h6 y1 + (h3 h7 - h4 h6) x1 + h0 h4 - h1 h3
    //                    h0 (y1 - h5) - h2 h6 y1 + (h5 h6 - h3) x1 + h2 h3
    //               y0 = ------------------------------------------------------------]]
    //                    (- h0 h7 y1) + h1 h6 y1 + (h3 h7 - h4 h6) x1 + h0 h4 - h1 h3

    ox = (-(h[1] * (iy - h[5]) - h[2] * h[7] * iy + (h[5] * h[7] - h[4]) * ix + h[2] * h[4])
           /
          (- h[0] * h[7] * iy + h[1] * h[6] * iy + (h[3] * h[7] - h[4] * h[6]) * ix + h[0] * h[4] - h[1] * h[3])) - tx;
    oy = ((h[0] * (iy - h[5]) - h[2] * h[6] * iy + (h[5] * h[6] - h[3]) * ix + h[2] * h[3])
          /
          (- h[0] * h[7] * iy + h[1] * h[6] * iy + (h[3] * h[7] - h[4] * h[6]) * ix + h[0] * h[4] - h[1] * h[3])) - ty;
  }

  bool IsValid() const
  {
    return true;
  }

  static void SetParameterBounds(/* ceres::Problem */ void * problem,
                                 const double template_image_width,
                                 const double template_image_height,
                                 double *x, const int offset);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <int COUNT>
class RollPackDetectionEstimator
{
  public:
  typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
  typedef RollPackDetectionModel<COUNT, double> MyRollPackDetectionModel;
  typedef std::shared_ptr<MyRollPackDetectionModel> MyRollPackDetectionModelPtr;
  typedef typename MyRollPackDetectionModel::HomographyArray HomographyArray;

  RollPackDetectionEstimator(const double template_size_x, const double template_size_y,
                             const double max_error_for_huber_loss, const Eigen::Vector2d & translation_px_weight):
      m_template_size_x(template_size_x), m_template_size_y(template_size_y),
      m_max_error_for_huber_loss(max_error_for_huber_loss),
      m_translation_px_weight_x(translation_px_weight.x()), m_translation_px_weight_y(translation_px_weight.y()) {}

  int PointToElementN(const Eigen::Vector2d & pt) const;
  int PointToElementN(const double x, const double y) const;

  // NULL if error?
  MyRollPackDetectionModelPtr Estimate(const HomographyArray & initial_homography,
                                       const Vector2dVector & reference_points,
                                       const Vector2dVector & observed_points,
                                       double & final_cost) const;

  MyRollPackDetectionModelPtr Estimate(const MyRollPackDetectionModel & initial_model,
                                       const Vector2dVector & reference_points,
                                       const Vector2dVector & observed_points,
                                       double & final_cost) const;

  private:
  double m_template_size_x, m_template_size_y;
  double m_max_error_for_huber_loss;
  double m_translation_px_weight_x; // error for one pixel of translation
  double m_translation_px_weight_y; // error for one pixel of translation
  bool m_quiet = true;
};

#endif // ROLL_PACK_DETECTION_MODEL_H
