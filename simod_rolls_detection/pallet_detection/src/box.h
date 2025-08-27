#ifndef PALLET_DETECTION_BOX_H
#define PALLET_DETECTION_BOX_H

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <vector>
#include <sstream>
#include <iostream>
#include <cmath>

struct Segment
{
    Eigen::Vector4f P1;
    Eigen::Vector4f P2;
};

inline bool operator==(const Segment& s1, const Segment& s2)
{
    bool S1P1eqS2P1 = (s1.P1.x() <= s2.P1.x() + 0.01f && s1.P1.x() >= s2.P1.x() - 0.01f
                    && s1.P1.y() <= s2.P1.y() + 0.01f && s1.P1.y() >= s2.P1.y() - 0.01f
                    && s1.P1.z() <= s2.P1.z() + 0.01f && s1.P1.z() >= s2.P1.z() - 0.01f);
    bool S1P2eqS2P2 = (s1.P2.x() <= s2.P2.x() + 0.01f && s1.P2.x() >= s2.P2.x() - 0.01f
                    && s1.P2.y() <= s2.P2.y() + 0.01f && s1.P2.y() >= s2.P2.y() - 0.01f
                    && s1.P2.z() <= s2.P2.z() + 0.01f && s1.P2.z() >= s2.P2.z() - 0.01f);

    return (S1P1eqS2P1 && S1P2eqS2P2);
}

class Box
{
public:
    Box(int box_number, const Eigen::Vector4f& box_central_position,
        const Eigen::Vector4f& box_scale, float box_z_rotation, const Eigen::Vector4f& box_color) :
        _box_number(box_number), _box_central_position(box_central_position), _box_scale(box_scale),
        _box_z_rotation(box_z_rotation), _box_color(box_color)
    {
        Eigen::Vector4f pillar_inital_position(0.0f, 0.0f, 0.0f, 1.0f);
        Eigen::Affine3f model_translation_matrix(Eigen::Translation3f(box_central_position.head<3>()));
        Eigen::Affine3f model_rotation_matrix(Eigen::AngleAxisf(box_z_rotation, Eigen::Vector3f::UnitZ()));

        Eigen::Affine3f pillar_1_translation(Eigen::Translation3f(Eigen::Vector3f((-box_scale.x() / 2.0f), (box_scale.y() / 2.0f), 0.0f)));
        Eigen::Affine3f pillar_2_translation(Eigen::Translation3f(Eigen::Vector3f((-box_scale.x() / 2.0f), (-box_scale.y() / 2.0f), 0.0f)));
        Eigen::Affine3f pillar_3_translation(Eigen::Translation3f(Eigen::Vector3f((box_scale.x() / 2.0f), (-box_scale.y() / 2.0f), 0.0f)));
        Eigen::Affine3f pillar_4_translation(Eigen::Translation3f(Eigen::Vector3f((box_scale.x() / 2.0f), (box_scale.y() / 2.0f), 0.0f)));

        this->_P1_position = model_translation_matrix * model_rotation_matrix * pillar_1_translation * pillar_inital_position;
        this->_P2_position = model_translation_matrix * model_rotation_matrix * pillar_2_translation * pillar_inital_position;
        this->_P3_position = model_translation_matrix * model_rotation_matrix * pillar_3_translation * pillar_inital_position;
        this->_P4_position = model_translation_matrix * model_rotation_matrix * pillar_4_translation * pillar_inital_position;
    }

    Box(int box_number) : Box(box_number, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f), Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f), 0.0f, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f)) {}

    int getBoxNumber() const
    {
        return this->_box_number;
    }

    Eigen::Vector4f getBoxCentralPosition() const
    {
        return this->_box_central_position;
    }

    Eigen::Vector4f getBoxScale() const
    {
        return this->_box_scale;
    }

    Eigen::Vector4f getP1() const
    {
        return this->_P1_position;
    }

    Eigen::Vector4f getP2() const
    {
        return this->_P2_position;
    }

    Eigen::Vector4f getP3() const
    {
        return this->_P3_position;
    }

    Eigen::Vector4f getP4() const
    {
        return this->_P4_position;
    }

    float getZRotation() const
    {
        return this->_box_z_rotation;
    }
    
    Eigen::Vector4f getColor() const
    {
        return this->_box_color;
    }

    std::vector<Eigen::Vector4f> getPillarsPositions() const
    {
        std::vector<Eigen::Vector4f> pillar_vector;

        pillar_vector.push_back(this->_P1_position);
        pillar_vector.push_back(this->_P2_position);
        pillar_vector.push_back(this->_P3_position);
        pillar_vector.push_back(this->_P4_position);

        return pillar_vector;
    }

    std::vector<Segment> getAllSegments() const
    {
        std::vector<Segment> segment_vector;
        Segment s1, s2, s3, s4;

        s1.P1 = this->_P1_position;
        s1.P2 = this->_P2_position;

        s2.P1 = this->_P2_position;
        s2.P2 = this->_P3_position;

        s3.P1 = this->_P3_position;
        s3.P2 = this->_P4_position;

        s4.P1 = this->_P4_position;
        s4.P2 = this->_P1_position;

        segment_vector.push_back(s1);
        segment_vector.push_back(s2);
        segment_vector.push_back(s3);
        segment_vector.push_back(s4);

        return segment_vector;
    }

    std::vector<Eigen::Vector4f> getVisiblePillars() const
    {
        return this->_visible_pillars;
    }

    void setVisiblePillar(const Eigen::Vector4f& pillar_position)
    {
        this->_visible_pillars.push_back(pillar_position);
    }
    
public:
    int _box_number;
    Eigen::Vector4f _box_central_position, _box_scale;
    Eigen::Vector4f _P1_position, _P2_position, _P3_position, _P4_position;
    float _box_z_rotation;
    Eigen::Vector4f _box_color;
    std::vector<Eigen::Vector4f> _visible_pillars;
};

#endif // PALLET_DETECTION_BOX_H
