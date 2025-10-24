#ifndef BOXES_TO_PALLET_DESCRIPTION_H
#define BOXES_TO_PALLET_DESCRIPTION_H

#include <Eigen/Dense>
#include <Eigen/Geometry>

// STL
#include <fstream>
#include <stdint.h>
#include <vector>
#include <set>
#include <cmath>
#include <map>

#include "box.h"
#include "expected_pallet.h"

class BoxesToPalletDescription
{
  public:
  typedef std::vector<int> IntVector;
  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > Vector4fVector;
  typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > Vector4dVector;
  typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
  typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
  typedef std::set<int> IntSet;
  typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> Vector3fPair;
  typedef std::vector<Vector3fPair> Vector3fPairVector;

  typedef uint64_t uint64;
  typedef uint32_t uint32;
  typedef int32_t int32;
  typedef int64_t int64;
  typedef uint8_t uint8;
  typedef uint16_t uint16;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::pair<uint64, uint64> Uint64Pair;
  typedef std::vector<Uint64Pair> Uint64PairVector;
  typedef std::set<Uint64Pair> Uint64PairSet;

  using ExpectedPallet = pallet_detection::ExpectedPallet;
  using ExpectedElement = pallet_detection::ExpectedElement;
  using ExpectedElementVector = pallet_detection::ExpectedElementVector;
  using ExpectedElementType = pallet_detection::ExpectedElementType;

  template <typename T> static T SQR(const T & t) {return t * t; }

  BoxesToPalletDescription() {}

  bool is_pillar_in_elements(ExpectedElementVector vector, ExpectedElement element);

  std::set<float> get_all_z_levels(const ExpectedPallet & epal);

  template <typename Type>
  bool is_in_vector(const std::vector<Type>& vector, const Type& element)
  {
    return std::find(vector.begin(), vector.end(), element) != vector.end();
  }

  Box get_box_by_pillar(const Eigen::Vector4f& pillar_position);

  Eigen::Vector4f get_plane_values(const Eigen::Vector4f& P1, const Eigen::Vector4f& P2);

  float get_segment_orientation(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& target_position,
                                const Eigen::Vector4f& P1, const Eigen::Vector4f& P2);

  int get_n_similar_pillar(const Eigen::Vector4f& pillar_position);

  Eigen::Vector4f get_other_point(const Eigen::Vector4f& pillar_position, const std::string& selected_point);

  bool are_collinear_segments(const Segment& s1, const Segment& s2);

  bool are_adjacent_segments(const Segment& s1, const Segment& s2);

  Eigen::Vector2f segment_intersection(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& target_position,
                                       const Eigen::Vector4f& P1_position, const Eigen::Vector4f& P2_position);

  bool is_element_visible(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& target_position);

  void generate_projection_points(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& target_position, int index_box);
  void generate_box_segments(const Eigen::Vector4f& cam_position, Box& box);

  void create_cam_segment(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& pillar_final_position, 
                          const Eigen::Vector4f& segment_color);

  void create_box_segment(const Eigen::Vector4f& pillar_1_position, const Eigen::Vector4f& pillar_2_position,
                          const Box& box, bool is_visible, bool show);

  Segment get_merged_segment(const std::vector<Segment>& adjacent_segments, const Eigen::Vector4f& cam_position);

  void append_adjacent_segment(std::vector<Segment>& adjacent_segments, const Segment& segment);

  void merge_segments(const Eigen::Vector4f& cam_position);

  void create_new_plane(const Eigen::Vector4f& pillar_1_position, const Eigen::Vector4f& pillar_2_position,
                        double box_z_scale, int plane_number);

  void create_new_plane_pillar(const Eigen::Vector4f& pillar_position, double box_z_scale);

  void fill_elements_array();

  void generate_segments(const Box& box, const Eigen::Vector4f& cam_position);

  ExpectedPallet Run(const ExpectedPallet & epal, const Eigen::Vector3f & camera_position, std::vector<Eigen::Vector4f> & cam_positions, std::vector<ColoredSegment> & visible_cam_segments);

  private:
  std::vector<Box> boxes;
  std::vector<Segment> box_segment_array;
  std::vector<Segment> box_visible_segment_array;
  std::vector<ColoredSegment> cam_segment_array;
  std::vector<Eigen::Vector4f> projection_points;
  ExpectedElementVector elements;
  ExpectedElementVector visible_planes;
  ExpectedElementVector visible_plane_pillars;
  std::set<float> z_levels;

  int n_boxes;
  int pl_number;
  float error_margin;
  float minimum_segment_length;
};


#endif // BOXES_TO_PALLET_DESCRIPTION_H
