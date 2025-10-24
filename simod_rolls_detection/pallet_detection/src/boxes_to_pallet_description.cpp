#include "boxes_to_pallet_description.h"

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

bool BoxesToPalletDescription::is_pillar_in_elements(ExpectedElementVector vector, ExpectedElement element)
{
  bool P1eqP2;

  for(auto e : vector)
  {
    P1eqP2 = (element.pillar.x() <= e.pillar.x() + 0.01f && element.pillar.x() >= e.pillar.x() - 0.01f
              && element.pillar.y() <= e.pillar.y() + 0.01f && element.pillar.y() >= e.pillar.y() - 0.01f
              && element.pillar.z() == e.pillar.z()
              && element.pillar.w() == e.pillar.w());

    if((int)e.type == (int)ExpectedElementType::PILLAR && P1eqP2)
      return true;
  }

  return false;
}

std::set<float> BoxesToPalletDescription::get_all_z_levels(const ExpectedPallet & epal)
{
  std::set<float> ret;

  for (const ExpectedElement & e : epal)
  {
    if (e.type == ExpectedElementType::BOX)
      ret.insert(e.box.z());
  }

  return ret;
}

Box BoxesToPalletDescription::get_box_by_pillar(const Eigen::Vector4f& pillar_position)
{
  for(auto box : boxes)
  {
    for(auto pillar : box.getVisiblePillars())
    {
      if(pillar == pillar_position)
        return box;
    }
  }

  return Box(-1);
}

Eigen::Vector4f BoxesToPalletDescription::get_plane_values(const Eigen::Vector4f& P1, const Eigen::Vector4f& P2)
{
  Eigen::Vector4f res;
  Eigen::Vector3f A1 = P2.head<3>() - P1.head<3>();
  Eigen::Vector3f A2(0.0f, 0.0f, 1.0f);

  Eigen::Vector3f N = (A1.cross(A2)).normalized();

  float d = - N.x() * P1.x() - N.y() * P1.y();

  res = Eigen::Vector4f(N.x(), N.y(), N.z(), d);

  return res;
}

float BoxesToPalletDescription::get_segment_orientation(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& target_position,
                              const Eigen::Vector4f& P1, const Eigen::Vector4f& P2)
{
  Eigen::Vector3f diff = target_position.head<3>() - cam_position.head<3>();
  Eigen::Vector3f plane_normal_values = get_plane_values(P1, P2).head<3>();
  float res = (diff.normalized()).dot(plane_normal_values);

  return res;
}

int BoxesToPalletDescription::get_n_similar_pillar(const Eigen::Vector4f& pillar_position)
{
  int count = 0;

  for(auto box : boxes)
  {
    for(auto other_pillar_position : box.getPillarsPositions())
    {
      if(pillar_position.x() <= (other_pillar_position.x() + error_margin) &&
          pillar_position.x() >= (other_pillar_position.x() - error_margin) &&
          pillar_position.y() <= (other_pillar_position.y() + error_margin) &&
          pillar_position.y() >= (other_pillar_position.y() - error_margin) &&
          pillar_position.z() <= (other_pillar_position.z() + error_margin) &&
          pillar_position.z() >= (other_pillar_position.z() - error_margin))
        count++;
    }
  }

  return count;
}

Eigen::Vector4f BoxesToPalletDescription::get_other_point(const Eigen::Vector4f& pillar_position, const std::string& selected_point)
{
  for(auto segment : box_visible_segment_array)
  {
    if(pillar_position.x() <= (segment.P1.x() + error_margin) && pillar_position.x() >= (segment.P1.x() - error_margin)
        && pillar_position.y() <= (segment.P1.y() + error_margin) && pillar_position.y() >= (segment.P1.y() - error_margin)
        && pillar_position.z() <= (segment.P1.z() + error_margin) && pillar_position.z() >= (segment.P1.z() - error_margin)
        && selected_point == "P2")
      return segment.P2;

    if(pillar_position.x() <= (segment.P2.x() + error_margin) && pillar_position.x() >= (segment.P2.x() - error_margin)
        && pillar_position.y() <= (segment.P2.y() + error_margin) && pillar_position.y() >= (segment.P2.y() - error_margin)
        && pillar_position.z() <= (segment.P2.z() + error_margin) && pillar_position.z() >= (segment.P2.z() - error_margin)
        && selected_point == "P1")
      return segment.P1;
  }

  return Eigen::Vector4f(-1.0f, -1.0f, -1.0f, -1.0f);
}

bool BoxesToPalletDescription::are_collinear_segments(const Segment& s1, const Segment& s2)
{
  Eigen::Vector3f segment_1_normal = get_plane_values(s1.P1, s1.P2).head<3>();
  Eigen::Vector3f segment_2_normal = get_plane_values(s2.P1, s2.P2).head<3>();
  float dot_product = segment_1_normal.dot(segment_2_normal);

  return (dot_product <= 1.01f && dot_product >= 0.99f);
}

bool BoxesToPalletDescription::are_adjacent_segments(const Segment& s1, const Segment& s2)
{
  bool collinear = are_collinear_segments(s1, s2);
  bool S1P1eqS2P2 = (s1.P1.x() <= s2.P2.x() + error_margin && s1.P1.x() >= s2.P2.x() - error_margin
                     && s1.P1.y() <= s2.P2.y() + error_margin && s1.P1.y() >= s2.P2.y() - error_margin
                     && s1.P1.z() <= s2.P2.z() + error_margin && s1.P1.z() >= s2.P2.z() - error_margin);
  bool S1P2eqS2P1 = (s1.P2.x() <= s2.P1.x() + error_margin && s1.P2.x() >= s2.P1.x() - error_margin
                     && s1.P2.y() <= s2.P1.y() + error_margin && s1.P2.y() >= s2.P1.y() - error_margin
                     && s1.P2.z() <= s2.P1.z() + error_margin && s1.P2.z() >= s2.P1.z() - error_margin);

  if(collinear)
  {
    if(S1P1eqS2P2 || S1P2eqS2P1)
      return true;
  }

  return false;
}

Eigen::Vector2f BoxesToPalletDescription::segment_intersection(const Eigen::Vector4f& cam_position,
                                                               const Eigen::Vector4f& target_position,
                                     const Eigen::Vector4f& P1_position, const Eigen::Vector4f& P2_position)
{
  Eigen::Vector2f vec(P1_position.x() - cam_position.x(), P1_position.y() - cam_position.y());
  Eigen::Vector2f res;
  Eigen::Matrix2f mat;

  mat.row(0) = Eigen::Vector2f(target_position.x() - cam_position.x(), P1_position.x() - P2_position.x());
  mat.row(1) = Eigen::Vector2f(target_position.y() - cam_position.y(), P1_position.y() - P2_position.y());

  res = mat.inverse() * vec;

  return res;
}

bool BoxesToPalletDescription::is_element_visible(const Eigen::Vector4f& cam_position,
                                                  const Eigen::Vector4f& target_position)
{
  float t, s;

  for(auto box_segment : box_segment_array)
  {
    t = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).x();
    s = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).y();

    if(t <= 0.99f && t > 0.0f && 0.0f <= s && s <= 1.0f)
      return false;
  }

  return true;
}

void BoxesToPalletDescription::generate_projection_points(const Eigen::Vector4f& cam_position,
                                                          const Eigen::Vector4f& target_position, int index_box)
{
  Eigen::Vector4f buf_pillar;
  float t, t_ref = 0, s, xp, yp, zp;

  for(auto box : boxes)
  {
    for(auto box_segment : box.getAllSegments())
    {
      t = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).x();
      s = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).y();

      if(t > 1.0f && 0.0f < s && s < 1.0f && index_box == box.getBoxNumber())
        return;

      if(t >= 0.99f && 0.0f <= s && s <= 1.0f && index_box != box.getBoxNumber())
      {
        if(t >= t_ref)
          t_ref = t;
      }
    }
  }

  for(auto box : boxes)
  {
    for(auto box_segment : box.getAllSegments())
    {
      t = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).x();
      s = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).y();

      if(t >= 0.99f && 0.0f <= s && s <= 1.0f && index_box != box.getBoxNumber())
      {
        if(t <= t_ref)
          t_ref = t;
      }
    }
  }

  for(auto box : boxes)
  {
    for(auto box_segment : box.getAllSegments())
    {
      t = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).x();
      s = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).y();

      if(t <= t_ref && 0.0f < s && s < 1.0f && get_segment_orientation(cam_position, (box_segment.P1 + box_segment.P2) / 2, box_segment.P1, box_segment.P2) > 0)
        return;
    }
  }

  for(auto& box : boxes)
  {
    for(auto box_segment : box.getAllSegments())
    {
      t = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).x();
      s = segment_intersection(cam_position, target_position, box_segment.P1, box_segment.P2).y();

      xp = box_segment.P1.x() + s * (box_segment.P2.x() - box_segment.P1.x());
      yp = box_segment.P1.y() + s * (box_segment.P2.y() - box_segment.P1.y());
      zp = box_segment.P1.z();

      buf_pillar = Eigen::Vector4f(xp, yp, zp, 1.0f);

      if(t == 1.0f && 0.0f <= s && s <= 1.0f && index_box == box.getBoxNumber())
        continue;

      if (t == t_ref && t_ref != 0 && 0.0f <= s && s <= 1.0f &&
          get_segment_orientation(cam_position, (box_segment.P1 + box_segment.P2) / 2, box_segment.P1, box_segment.P2) <= 0)
      {
        box.setVisiblePillar(buf_pillar);
        projection_points.push_back(buf_pillar);
        return;
      }
    }
  }
}

void BoxesToPalletDescription::generate_box_segments(const Eigen::Vector4f& cam_position, Box& box)
{
  Eigen::Vector4f avg_vector, plane_values;
  std::vector<Eigen::Vector4f> pillar_positions = box.getVisiblePillars();
  float dot_product, segment_length;

  if(pillar_positions.size() > 1)
  {
    for(int i = 0; i < (int)(pillar_positions.size() - 1); i++)
    {
      for(int j = i + 1; j < (int)pillar_positions.size(); j++)
      {
        avg_vector = (pillar_positions.at(i) + pillar_positions.at(j)) / 2;
        segment_length = (pillar_positions.at(j) - pillar_positions.at(i)).norm();

        if(is_element_visible(cam_position, avg_vector) && segment_length >= minimum_segment_length)
        {
          dot_product = get_segment_orientation(cam_position, avg_vector, pillar_positions.at(i), pillar_positions.at(j));

          if(dot_product > 0)
          {
            if(!is_in_vector(box_visible_segment_array, {pillar_positions.at(j), pillar_positions.at(i)}))
              create_box_segment(pillar_positions.at(j), pillar_positions.at(i), box, true, false);
          }
          else
          {
            if(!is_in_vector(box_visible_segment_array, {pillar_positions.at(i), pillar_positions.at(j)}))
              create_box_segment(pillar_positions.at(i), pillar_positions.at(j), box, true, false);
          }
        }
      }
    }
  }
}

void BoxesToPalletDescription::create_cam_segment(const Eigen::Vector4f& cam_position, const Eigen::Vector4f& pillar_final_position, const Eigen::Vector4f& segment_color)
{
  ColoredSegment segment;

  segment.P_cam = cam_position;
  segment.P_pillar = pillar_final_position;
  segment.color = segment_color;

  cam_segment_array.push_back(segment);
}

void BoxesToPalletDescription::create_box_segment(const Eigen::Vector4f& pillar_1_position, const Eigen::Vector4f& pillar_2_position,
                                                  const Box& box, bool is_visible, bool show)
{
  Segment segment, buf_segment;

  segment.P1 = pillar_1_position;
  segment.P2 = pillar_2_position;

  box_segment_array.push_back(segment);

  if(is_visible)
    box_visible_segment_array.push_back(segment);
}

Segment BoxesToPalletDescription::get_merged_segment(const std::vector<Segment>& adjacent_segments,
                                                     const Eigen::Vector4f& cam_position)
{
  Segment ret_segment;
  std::vector<Eigen::Vector4f> points;
  float max_dist = 0, dist;

  for(auto segment : adjacent_segments)
  {
    points.push_back(segment.P1);
    points.push_back(segment.P2);
  }

  for(auto p1 : points)
  {
    for(auto p2 : points)
    {
      dist = (p2 - p1).norm();

      if(dist > max_dist)
        max_dist = dist;
    }
  }

  for(auto p1 : points)
  {
    for(auto p2 : points)
    {
      dist = (p2 - p1).norm();

      if(dist == max_dist)
      {
        if(get_segment_orientation(cam_position, (p1 + p2) / 2, p1, p2) > 0)
        {
          ret_segment.P1 = p2;
          ret_segment.P2 = p1;
        }
        else
        {
          ret_segment.P1 = p1;
          ret_segment.P2 = p2;
        }
      }
    }
  }

  return ret_segment;
}

void BoxesToPalletDescription::append_adjacent_segment(std::vector<Segment>& adjacent_segments, const Segment& segment)
{
  if(is_in_vector(adjacent_segments, segment))
    return;

  adjacent_segments.push_back(segment);

  for(auto s : box_visible_segment_array)
  {
    if(are_adjacent_segments(segment, s))
    {
      append_adjacent_segment(adjacent_segments, s);
    }
  }
}

void BoxesToPalletDescription::merge_segments(const Eigen::Vector4f& cam_position)
{
  std::vector<Segment> adjacent_segments;
  std::vector<Segment> buf_segment_array;
  Segment buf_segment;

  for(auto segment : box_visible_segment_array)
  {
    append_adjacent_segment(adjacent_segments, segment);

    if(adjacent_segments.size() == 1)
      buf_segment_array.push_back(adjacent_segments.at(0));
    else
    {
      buf_segment = get_merged_segment(adjacent_segments, cam_position);

      if(!is_in_vector(buf_segment_array, buf_segment))
      {
        buf_segment_array.push_back(buf_segment);
      }
    }

    adjacent_segments.clear();
  }

  box_visible_segment_array = buf_segment_array;
}

void BoxesToPalletDescription::create_new_plane(const Eigen::Vector4f& pillar_1_position,
                                                const Eigen::Vector4f& pillar_2_position,
                                                double box_z_scale, int plane_number)
{
  ExpectedElement plane;
  std::stringstream plane_name_ss;
  double z_min = pillar_1_position.z() - (box_z_scale / 2);
  double z_max = pillar_1_position.z() + (box_z_scale / 2);
  Eigen::Vector2d plane_z(z_min, z_max);
  Eigen::Vector3d avg_point = Eigen::Vector3d((pillar_1_position.x() + pillar_2_position.x()) / 2, (pillar_1_position.y() + pillar_2_position.y()) / 2, (pillar_1_position.z() + pillar_2_position.z()) / 2);

  plane_name_ss << "P" << plane_number << "_Z" << pillar_1_position.z() * 10 << "CM";

  plane.type = ExpectedElementType::PLANE;
  plane.name = plane_name_ss.str();
  plane.plane = (get_plane_values(pillar_1_position, pillar_2_position)).cast<double>();
  plane.plane_z = plane_z;
  plane.plane_point = avg_point;

  visible_planes.push_back(plane);
}

void BoxesToPalletDescription::create_new_plane_pillar(const Eigen::Vector4f& pillar_position, double box_z_scale)
{
  ExpectedElement pillar;
  std::stringstream left_plane_id_ss;
  std::stringstream right_plane_id_ss;
  double z_min = pillar_position.z() - (box_z_scale / 2);
  double z_max = pillar_position.z() + (box_z_scale / 2);
  Eigen::Vector4d pillar_data = Eigen::Vector4d(pillar_position.x(), pillar_position.y(), z_min, z_max);

  left_plane_id_ss << "";
  right_plane_id_ss << "";

  for(int i = 0; i < (int)visible_planes.size(); i++)
  {
    if(((box_visible_segment_array.at(i).P1)-pillar_position).norm() <= 0.01f)
    {
      left_plane_id_ss << visible_planes.at(i).name;
      break;
    }
  }

  for(int i = 0; i < (int)visible_planes.size(); i++)
  {
    if(((box_visible_segment_array.at(i).P2)-pillar_position).norm() <= 0.01f)
    {
      right_plane_id_ss << visible_planes.at(i).name;
      break;
    }
  }

  pillar.type = ExpectedElementType::PILLAR;
  pillar.pillar = pillar_data;
  pillar.pillar_left_plane_name = left_plane_id_ss.str();
  pillar.pillar_right_plane_name = right_plane_id_ss.str();

  visible_plane_pillars.push_back(pillar);
}

void BoxesToPalletDescription::fill_elements_array()
{
  for(auto plane : visible_planes)
  {
    elements.push_back(plane);

    for(auto pillar : visible_plane_pillars)
    {
      if(pillar.pillar_left_plane_name == plane.name && !is_pillar_in_elements(elements, pillar))
        elements.push_back(pillar);
    }

    for(auto pillar : visible_plane_pillars)
    {
      if(pillar.pillar_right_plane_name == plane.name && !is_pillar_in_elements(elements, pillar))
        elements.push_back(pillar);
    }
  }
}

void BoxesToPalletDescription::generate_segments(const Box& box, const Eigen::Vector4f& cam_position)
{
  Eigen::Vector4f segment_1_color(1.0f, 0.5f, 0.0f, 0.5f);
  Eigen::Vector4f segment_2_color(1.0f, 1.0f, 1.0f, 0.5f);
  Eigen::Vector4f segment_3_color(0.0f, 1.0f, 0.5f, 0.5f);
  Eigen::Vector4f segment_4_color(1.0f, 0.5f, 1.0f, 0.5f);

  create_cam_segment(cam_position, box.getP1(), segment_1_color);
  create_cam_segment(cam_position, box.getP2(), segment_2_color);
  create_cam_segment(cam_position, box.getP3(), segment_3_color);
  create_cam_segment(cam_position, box.getP4(), segment_4_color);

  create_box_segment(box.getP1(), box.getP2(), box, false, false);
  create_box_segment(box.getP2(), box.getP3(), box, false, false);
  create_box_segment(box.getP3(), box.getP4(), box, false, false);
  create_box_segment(box.getP4(), box.getP1(), box, false, false);
}

BoxesToPalletDescription::ExpectedPallet BoxesToPalletDescription::Run(const ExpectedPallet & epal, const Eigen::Vector3f & camera_pos, std::vector<Eigen::Vector4f> & cam_positions, std::vector<ColoredSegment> & visible_cam_segments)
{
  n_boxes = epal.size();
  z_levels = get_all_z_levels(epal);
  pl_number = 0;
  error_margin = 0.02f;
  minimum_segment_length = 0.02f;

  Eigen::Vector4f color_vector(1.0f, 1.0f, 0.0f, 1.0f);

  for(auto z_level : z_levels)
  {
    Eigen::Vector4f cam_position(camera_pos.x(), camera_pos.y(), z_level, 1.0f);

    cam_positions.push_back(cam_position);

    box_segment_array.clear();
    box_visible_segment_array.clear();
    boxes.clear();
    projection_points.clear();
    visible_planes.clear();
    visible_plane_pillars.clear();
    cam_segment_array.clear();

    for(uint64 i = 0; i < epal.size(); i++)
    {
      const ExpectedElement & e = epal[i];

      if (e.type != ExpectedElementType::BOX)
        continue;

      if (std::abs(e.box.z() - z_level) < 0.001f)
      {
        const Eigen::Vector4f box_central_position(e.box.x(), e.box.y(), e.box.z(), 1.0f);
        const Eigen::Vector4f box_scale(e.box_size.x(), e.box_size.y(), e.box_size.z(), 1.0f);
        const float box_z_rotation = e.box.w();

        Box box(i, box_central_position, box_scale, box_z_rotation, color_vector);
        //Box box(i, Eigen::Vector4f(data_matrix.matrix()(i, 3), data_matrix.matrix()(i, 4), data_matrix.matrix()(i, 5), 1.0f), Eigen::Vector4f(data_matrix.matrix()(i, 0), data_matrix.matrix()(i, 1), data_matrix.matrix()(i, 2), 1.0f), data_matrix.matrix()(i, 6), color_vector);

        boxes.push_back(box);

        generate_segments(box, cam_position);
      }
    }

    for(auto& box : boxes)
    {
      for(auto pillar_position : box.getPillarsPositions())
      {
        if(is_element_visible(cam_position, pillar_position))
          box.setVisiblePillar(pillar_position);
      }
    }

    for(auto box : boxes)
    {
      for(auto pillar_position : box.getVisiblePillars())
      {
        generate_projection_points(cam_position, pillar_position, box.getBoxNumber());
      }
    }

    for(auto& box : boxes)
    {
      generate_box_segments(cam_position, box);
    }

    for(auto cam_segment : cam_segment_array)
    {
      if(is_element_visible(cam_position, cam_segment.P_pillar) && get_n_similar_pillar(cam_segment.P_pillar) < 2)
        visible_cam_segments.push_back(cam_segment);
    }

    merge_segments(cam_position);

    for(auto segment : box_visible_segment_array)
    {
      create_new_plane(segment.P1, segment.P2, get_box_by_pillar(segment.P1).getBoxScale().z(), pl_number++);
      create_box_segment(segment.P1, segment.P2, get_box_by_pillar(segment.P1), false, true);
    }

    for(auto box : boxes)
    {
      for(auto pillar_position : box.getVisiblePillars())
      {
        if(get_n_similar_pillar(pillar_position) < 2 && is_in_vector(box.getPillarsPositions(), pillar_position))
          create_new_plane_pillar(pillar_position, box.getBoxScale().z());
        else if(get_n_similar_pillar(pillar_position) >= 2 && is_in_vector(box.getPillarsPositions(), pillar_position))
        {
          if(!are_collinear_segments({get_other_point(pillar_position, "P1"), pillar_position}, {pillar_position, get_other_point(pillar_position, "P2")}))
          {
            create_new_plane_pillar(pillar_position, box.getBoxScale().z());
          }
        }
      }
    }

    fill_elements_array();
  }

  return elements;
}

