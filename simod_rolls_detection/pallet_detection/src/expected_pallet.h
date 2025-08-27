#ifndef EXPECTED_PALLET_H
#define EXPECTED_PALLET_H

#include <Eigen/Dense>

#include <vector>
#include <limits>
#include <stdint.h>
#include <string>

namespace pallet_detection
{
  typedef uint64_t uint64;

  enum class ExpectedElementType
  {
    PILLAR,
    PLANE,
    BOX,
  };

  struct ExpectedElement
  {
    ExpectedElementType type = ExpectedElementType(0);
    std::string name;
    Eigen::Vector4d pillar = Eigen::Vector4d::Zero(); // x, y, min_z, max_z
    uint64 pillar_left_plane_id = uint64(-1); // if not -1, this is a left pillar for plane element plane_id
    std::string pillar_left_plane_name = "";  // if not empty, name of plane pillar_left_plane_id
    uint64 pillar_right_plane_id = uint64(-1); // same, but for right pillar
    std::string pillar_right_plane_name = "";  // if not empty, name of plane pillar_right_plane_id
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();  // x, y, z, d
    Eigen::Vector2d plane_z = Eigen::Vector2d(0.0, 1.0); // min_z, max_z
    Eigen::Vector3d plane_point = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()); // one significant point of this plane

    Eigen::Vector3d box_size = Eigen::Vector3d::Zero();
    Eigen::Vector4d box = Eigen::Vector4d::Zero(); // x, y, z, rotation

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  typedef std::vector<ExpectedElement, Eigen::aligned_allocator<ExpectedElement> > ExpectedElementVector;
  typedef ExpectedElementVector ExpectedPallet;

  inline std::string UpdatePlaneIds(ExpectedPallet & pallet)
  {
    for (ExpectedElement & elem : pallet)
    {
      if (elem.type != ExpectedElementType::PILLAR)
        continue;

      if (elem.pillar_left_plane_name != "")
      {
        uint64 found = uint64(-1);
        for (uint64 i = 0; i < pallet.size(); i++)
        {
          const ExpectedElement & e = pallet[i];
          if (e.type != ExpectedElementType::PLANE)
            continue;
          if (e.name == elem.pillar_left_plane_name)
            found = i;
        }
        if (found == uint64(-1))
        {
          return "Unable to find match for left plane \"" + elem.pillar_left_plane_name + "\" in pillar " + elem.name;
        }

        elem.pillar_left_plane_id = found;
      }

      if (elem.pillar_right_plane_name != "")
      {
        uint64 found = uint64(-1);
        for (uint64 i = 0; i < pallet.size(); i++)
        {
          const ExpectedElement & e = pallet[i];
          if (e.type != ExpectedElementType::PLANE)
            continue;
          if (e.name == elem.pillar_right_plane_name)
            found = i;
        }
        if (found == uint64(-1))
        {
          return "Unable to find match for right plane \"" + elem.pillar_right_plane_name + "\" in pillar " + elem.name;
        }

        elem.pillar_right_plane_id = found;
      }
    }

    return "";
  }
}

#endif // EXPECTED_PALLET_H
