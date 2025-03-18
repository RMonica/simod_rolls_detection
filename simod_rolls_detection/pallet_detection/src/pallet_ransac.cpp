#include "pallet_ransac.h"

#include "pallet_detection_solver.h"

typedef PalletRansac::ExpectedElement ExpectedElement;
typedef PalletRansac::ExpectedPallet ExpectedPallet;

#ifndef NAN
  #define NAN (std::numeric_limits<double>::quiet_NaN())
#endif

PalletRansac::PalletRansac(const double pallet_convergence_threshold,
                           const uint64 iterations,
                           const double max_pose_correction_distance,
                           const double max_pose_correction_angle,
                           const double planes_similarity_max_angle,
                           const double planes_similarity_max_distance,
                           const double point_similarity_max_distance,
                           const uint64 random_seed)
  : m_random_gen(random_seed)
{
  m_pallet_convergence_threshold = pallet_convergence_threshold;
  m_iterations = iterations;
  m_max_pose_correction_distance = max_pose_correction_distance;
  m_max_pose_correction_angle = max_pose_correction_angle;

  m_planes_similarity_max_angle = planes_similarity_max_angle;
  m_planes_similarity_max_distance = planes_similarity_max_distance;
  m_point_similarity_max_distance = point_similarity_max_distance;
}

Eigen::Vector4d PalletRansac::TransformPlane(const Eigen::Vector4d & plane, const Eigen::Affine3d & transform)
{
  Eigen::Vector4d result;

  Eigen::Vector3d normal = plane.head<3>();
  Eigen::Vector3d pt = -normal * plane.w();
  normal = transform.linear() * normal;
  pt = transform * pt;
  result.head<3>() = normal;
  result.w() = -normal.dot(pt);

  return result;
}

Eigen::Vector4d PalletRansac::TransformPlane(const Eigen::Vector4d & plane, const Eigen::Vector3d & transform2d, const double ztransl)
{
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation().head<2>() = transform2d.head<2>();
  transform.translation().z() = ztransl;
  transform.linear() = Eigen::AngleAxisd(transform2d.z(), Eigen::Vector3d::UnitZ()).matrix();
  return TransformPlane(plane, transform);
}

Eigen::Vector4d PalletRansac::TransformPillar(const Eigen::Vector4d & pillar, const Eigen::Affine3d & transform)
{
  Eigen::Vector4d result;

  Eigen::Vector3d pillar_point1 = pillar.head<3>();
  Eigen::Vector3d pillar_point2 = pillar_point1;
  pillar_point2.z() = pillar.w();
  pillar_point1 = transform * pillar_point1;
  pillar_point2 = transform * pillar_point2;
  result.head<3>() = pillar_point1;
  result.w() = pillar_point2.z();

  return result;
}

Eigen::Vector4d PalletRansac::TransformPillar(const Eigen::Vector4d & pillar, const Eigen::Vector3d & transform2d, const double ztransl)
{
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation().head<2>() = transform2d.head<2>();
  transform.translation().z() = ztransl;
  transform.linear() = Eigen::AngleAxisd(transform2d.z(), Eigen::Vector3d::UnitZ()).matrix();
  return TransformPillar(pillar, transform);
}

Eigen::Vector4d PalletRansac::TransformBox(const Eigen::Vector4d & box, const Eigen::Affine3d & transform, const double angle)
{
  Eigen::Vector4d result;

  Eigen::Vector3d box_point = box.head<3>();
  result.head<3>() = transform * box_point;
  result.w() = box.w() + angle;

  return result;
}

Eigen::Vector4d PalletRansac::TransformBox(const Eigen::Vector4d & box, const Eigen::Vector3d & transform2d, const double ztransl)
{
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation().head<2>() = transform2d.head<2>();
  transform.translation().z() = ztransl;
  transform.linear() = Eigen::AngleAxisd(transform2d.z(), Eigen::Vector3d::UnitZ()).matrix();
  return TransformBox(box, transform, transform2d.z());
}

ExpectedElement PalletRansac::TransformElement(const ExpectedElement & ee, const Eigen::Vector3d & transform2d, const double ztransl)
{
  ExpectedElement result = ee;
  switch (result.type)
  {
  case ExpectedElementType::PILLAR:
    result.pillar = TransformPillar(result.pillar, transform2d, ztransl);
    break;
  case ExpectedElementType::PLANE:
    result.plane = TransformPlane(result.plane, transform2d, ztransl);
    result.plane_z += Eigen::Vector2d::Ones() * ztransl;
    result.plane_point.z() += ztransl;
    break;
  case ExpectedElementType::BOX:
    result.box = TransformBox(result.box, transform2d, ztransl);
    break;
  }
  return result;
}

ExpectedPallet PalletRansac::TransformPallet(const ExpectedPallet & ep, const Eigen::Vector3d & transform2d, const double ztransl)
{
  ExpectedPallet result;
  for (ExpectedElement ee : ep)
    result.push_back(TransformElement(ee, transform2d, ztransl));
  return result;
}

double PalletRansac::NormalDistance(const Eigen::Vector3d & n1, const Eigen::Vector3d & n2, const double max_angle)
{
  const double d1 = std::acos(std::min(std::abs(n1.dot(n2)), 1.0)) / max_angle;
  return d1;
}

double PalletRansac::PlanesDistance(const Eigen::Vector4d & p1, const Eigen::Vector4d & p2, const double max_angle, const double max_distance)
{
  const Eigen::Vector3d n1 = p1.head<3>();
  const Eigen::Vector3d n2 = p2.head<3>();
  const double d1 = NormalDistance(n1, n2, max_angle);
  const Eigen::Vector3d pt1 = n1 * p1.w();
  const Eigen::Vector3d pt2 = n2 * p2.w();
  const double d2 = (pt1 - pt2).norm() / max_distance;
  return std::max(d1, d2);
}

// first b then a
Eigen::Vector3d PalletRansac::CombinePoses(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  Eigen::Vector3d result;
  result.z() = a.z() + b.z();
  const Eigen::Vector2d pta = a.head<2>();
  const Eigen::Vector2d ptb = b.head<2>();
  result.head<2>() = Eigen::Rotation2Dd(a.z()).toRotationMatrix() * ptb + pta;
  return result;
}

void PalletRansac::Run(const ExpectedPallet & expected_pallet,
                       const ExpectedPallet & real_pallet,
                       const Eigen::Vector3d & initial_guess,
                       Eigen::Vector3d & best_pose,
                       Eigen::Vector3d & best_refined_pose,
                       Uint64PairVector & best_consensus)
{
  ExpectedPallet my_expected_pallet = TransformPallet(expected_pallet, initial_guess);

  Uint64PairVector available_corresp;
  Uint64PairSet available_corresp_set;
  for (uint64 real_i = 0; real_i < real_pallet.size(); real_i++)
    for (uint64 expected_i = 0; expected_i < my_expected_pallet.size(); expected_i++)
    {
      const ExpectedElement & exp_elem = my_expected_pallet[expected_i];
      const ExpectedElement & real_elem = real_pallet[real_i];
      if (exp_elem.type != real_elem.type) // cannot correspond: different type
        continue;

      if (exp_elem.type == ExpectedElementType::BOX)
        continue; // box correspondences not implemented

      if (exp_elem.type == ExpectedElementType::PLANE)
      {
        const Eigen::Vector2d exp_plane_z = exp_elem.plane_z;
        const Eigen::Vector2d real_plane_z = real_elem.plane_z;

        // if not overlapping, discard
        if (real_plane_z.x() > exp_plane_z.y() || real_plane_z.y() < exp_plane_z.x())
          continue;
      }

      if (exp_elem.type == ExpectedElementType::PILLAR)
      {
        const Eigen::Vector2d exp_pillar_z = exp_elem.pillar.tail<2>();
        const Eigen::Vector2d real_pillar_z = real_elem.pillar.tail<2>();

        // if not overlapping, discard
        if (real_pillar_z.x() > exp_pillar_z.y() || real_pillar_z.y() < exp_pillar_z.x())
          continue;
      }

      available_corresp.push_back(Uint64Pair(real_i, expected_i));
      available_corresp_set.insert(Uint64Pair(real_i, expected_i));
    }

  PalletDetectionPoseEstimation pdpe(m_pallet_convergence_threshold);
  pdpe.SetQuiet(true);

  best_pose = Eigen::Vector3d::Zero();
  best_refined_pose = Eigen::Vector3d::Zero();
  best_consensus.clear();

  const uint64 RANSAC_ITERATIONS = m_iterations;
  const uint64 INITIAL_CORRS = 2;
  if (available_corresp.size() < INITIAL_CORRS)
  {
    best_pose = initial_guess;
    best_refined_pose = initial_guess;
    return;
  }

  for (uint64 ransac_i = 0; ransac_i < RANSAC_ITERATIONS; ransac_i++)
  {
    bool fail = false;
    Uint64Vector already_selected_corrs;

    PalletDetectionPoseEstimation::PlaneCorrVector plane_corrs;
    PalletDetectionPoseEstimation::PointCorrVector point_corrs;
    for (uint64 corr_i = 0; corr_i < INITIAL_CORRS; corr_i++)
    {
      std::uniform_int_distribution<uint64> distrib(0, available_corresp.size() - 1);
      const uint64 selected_corr = distrib(m_random_gen);
      if (std::find(already_selected_corrs.begin(), already_selected_corrs.end(), selected_corr) != already_selected_corrs.end())
      {
        fail = true;
        break;
      }
      const uint64 exp_elem_i = available_corresp[selected_corr].second;
      const uint64 real_elem_i = available_corresp[selected_corr].first;
      const ExpectedElementType type = my_expected_pallet[exp_elem_i].type;

      if (type == ExpectedElementType::PLANE)
      {
        // if plane, then check if already selected another parallel plane
        // if so, fail to prevent instability
        for (const uint64 already_corr : already_selected_corrs)
        {
          const uint64 already_exp_elem_i = available_corresp[already_corr].second;
          const ExpectedElementType already_type = my_expected_pallet[already_exp_elem_i].type;
          if (already_type != ExpectedElementType::PLANE)
            continue;
          const Eigen::Vector3d n = my_expected_pallet[exp_elem_i].plane.head<3>();
          const Eigen::Vector3d already_n = my_expected_pallet[already_exp_elem_i].plane.head<3>();
          if (NormalDistance(n, already_n, m_planes_similarity_max_angle) < 1.0)
          {
            fail = true;
            break;
          }
        }
        if (fail)
          break;
      }

      if (type == ExpectedElementType::PILLAR)
      {
        const Eigen::Vector2d p = real_pallet[real_elem_i].pillar.head<2>();
        const Eigen::Vector2d ip = my_expected_pallet[exp_elem_i].pillar.head<2>();
        PalletDetectionPoseEstimation::PointCorr point_corr(ip, p, 1);
        point_corrs.push_back(point_corr);
      }

      if (type == ExpectedElementType::PLANE)
      {
        const Eigen::Vector4d p = real_pallet[real_elem_i].plane;
        const Eigen::Vector4d ip = my_expected_pallet[exp_elem_i].plane;
        PalletDetectionPoseEstimation::PlaneCorr plane_corr(Eigen::Vector3d(ip.x(), ip.y(), ip.w()),
                                                            Eigen::Vector3d(p.x(), p.y(), p.w()),
                                                            1);
        plane_corrs.push_back(plane_corr);
      }

      already_selected_corrs.push_back(selected_corr);
    }

    if (fail)
      continue;

    const Eigen::Vector3d pose = pdpe.estimate_pose(plane_corrs, point_corrs);
    if (std::isnan(pose[0]))
      continue;
    if (pose.head<2>().norm() > m_max_pose_correction_distance ||
        std::abs(pose.z()) > m_max_pose_correction_angle)
      continue;

    Uint64PairVector consensus;
    {
      PalletRansac::ExpectedPallet new_expected_pallet = TransformPallet(my_expected_pallet, pose);

      for (uint64 ei = 0; ei < new_expected_pallet.size(); ei++)
      {
        uint64 best_ri = 0;
        double best_distance = NAN;
        for (uint64 ri = 0; ri < real_pallet.size(); ri++)
        {
          const Uint64Pair maybe_corresp(ri, ei);

          if (available_corresp_set.find(maybe_corresp) == available_corresp_set.end())
            continue; // corresp not available

          const ExpectedElement ee = new_expected_pallet[ei];
          const ExpectedElement re = real_pallet[ri];

          double distance = NAN;
          if (re.type == ExpectedElementType::PILLAR)
          {
            distance = (ee.pillar.head<2>() - re.pillar.head<2>()).norm();
            if (distance > m_planes_similarity_max_distance)
              continue;
          }

          if (re.type == ExpectedElementType::PLANE)
          {
            distance = PlanesDistance(ee.plane, re.plane, m_planes_similarity_max_angle, m_planes_similarity_max_distance);
            if (distance > 1.0)
              continue;
          }

          if (std::isnan(best_distance) || distance < best_distance)
          {
            best_ri = ri;
            best_distance = distance;
          }
        }

        if (!std::isnan(best_distance))
          consensus.push_back(Uint64Pair(best_ri, ei));
      }
    }

    if (consensus.size() > best_consensus.size())
    {
      PalletDetectionPoseEstimation relaxed_pdpe(std::numeric_limits<float>::quiet_NaN());
      relaxed_pdpe.SetQuiet(true);

      plane_corrs.clear();
      point_corrs.clear();

      for (const Uint64Pair & corresp : consensus)
      {
        const uint64 exp_elem_i = corresp.second;
        const uint64 real_elem_i = corresp.first;
        const ExpectedElementType type = my_expected_pallet[exp_elem_i].type;

        if (type == ExpectedElementType::PILLAR)
        {
          const Eigen::Vector2d p = real_pallet[real_elem_i].pillar.head<2>();
          const Eigen::Vector2d ip = my_expected_pallet[exp_elem_i].pillar.head<2>();
          PalletDetectionPoseEstimation::PointCorr point_corr(ip, p, 0.5);
            // 0.5: expect two pillars for each plane, so halve weight
          point_corrs.push_back(point_corr);
        }

        if (type == ExpectedElementType::PLANE)
        {
          const Eigen::Vector4d p = real_pallet[real_elem_i].plane;
          const Eigen::Vector4d ip = my_expected_pallet[exp_elem_i].plane;
          PalletDetectionPoseEstimation::PlaneCorr plane_corr(Eigen::Vector3d(ip.x(), ip.y(), ip.w()),
                                                              Eigen::Vector3d(p.x(), p.y(), p.w()),
                                                              1);
          plane_corrs.push_back(plane_corr);
        }
      }

      Eigen::Vector3d refined_pose = relaxed_pdpe.estimate_pose(plane_corrs, point_corrs, pose);
      if (std::isnan(refined_pose[0]))
        continue;
      if (refined_pose.head<2>().norm() > m_max_pose_correction_distance ||
          std::abs(refined_pose.z()) > m_max_pose_correction_angle)
        continue; // too distant

      best_consensus = consensus;
      best_pose = pose;
      best_refined_pose = refined_pose;
    }
  }

  best_pose = CombinePoses(best_pose, initial_guess);
  best_refined_pose = CombinePoses(best_refined_pose, initial_guess);
}
