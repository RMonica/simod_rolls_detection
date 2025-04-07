This software is part of the [SiMOD](https://www.unibo.it/it/ricerca/progetti-e-iniziative/pr-fesr-emilia-romagna-2021-2027/1223/20430/20650) project.

simod_rolls_detection
=====================

Pallet detection
----------------

The `pallet_detection_node` node locates a specific configuration of boxes (aka "*pallet*") in a RGB-D image. Detection is started by calling a ROS action.

In summary, inputs are:

- A description of the configuration of boxes to be located.
- A RGB-D image of the boxes, taken from the front with the camera principal axis roughly parallel to the ground.
- The camera pose with respect to a *base* reference frame.<br />**Note**: Any frame can be used here, but the Z axis of the *base* frame must be  vertical, i.e. perpendicular to the ground.
- The height of the ground with respect to the world frame.
- An initial guess of the pose of the pallet.

Outputs are:

- The pose of the pallet with respect to the *base* frame.
- The pose of each box in the pallet, with respect to the *base* frame.

**Parameters**

The algorithm within the node has many configuration parameters, which are beyond the scope of this readme. Useful parameters for interfacing with the node are:

- `detect_pallet_action`: the name of the action server which will be created by this node. Default: `/detect_pallet`.
- `depth_image_topic`, `rgb_image_topic` and `camera_info_topic`: the node will subscribe to these topics to read depth, color and camera_info of the RGB-D image.

When the action is called, the node first waits for messages from the camera on the `depth_image_topic`, `rgb_image_topic` and `camera_info_topic` topics.
The detection algorithm is executed only when at least one message is received for each topic.

**Action**

Detection is started by calling the `/detect_pallet` action. The action is of type `simod_rolls_detection/DetectPallet.action`. The action goal has these fields:

- `camera_pose`: pose of the camera with respect to the *base* frame.
- `pallet_description_filename`: a string containing the file name where the box configuration is described.
- `floor_z`: height of the ground with respect to the *base* frame. Points below this level will be ignored, as they are considered part of the ground.
- `initial_guess_center`: initial guess of the center of the pallet (meters).
- `initial_guess_rotation`: initial guess of the pallet rotation (rad), around an axis passing through the center of the pallet and parallel to the Z axis of the *base* frame.
- `initial_guess_size`: size (meters) of a bounding box centered in `initial_guess_center` and oriented using `initial_guess_rotation`. Points outside this box will be ignored, hence the pallet can only be located within this box.

**Pallet description**

The pallet description file (whose name is in `pallet_description_filename`) contains a list of elements which describe the pallet, one for each line.
Available elements are *plane*, *pillar* and *box*. The *guess* element is also available but it is now unused.

Each element may be followed by `name *word*` which defines a name for the element. `*word*` may be only a single word.

Comments may also be added starting with `#` at the beginning of the line.

Coordinates of the elements are with respect to a local *pallet* reference frame, which is an arbitrary reference frame which should be roughly located at the center of the pallet.

*plane*

Planes are defined with the string `plane` followed by four numbers.
The four numbers are the plane equation parameters `a`, `b`, `c`, `d` so that:

```
ax + by + cz + d = 0
```

*pillar*

Pillars are vertical edges.
They are defined with the string `pillar` followed by four numbers: `x`, `y`, `min_z`, `max_z`.
`x` and `y` represent the position of the pillar on the ground.
The pillar extends from a minimum height of `min_z` and a maximum height of `max_z`.

Optionally, pillar definitions may be followed by one or more relations with planes in the pallet, by adding `left *plane_name*` or `right *plane_name*` at the end of the line.
These indicate that the pillar is in contact with a vertical box face which belongs to the plane.
`left` indicates that the pillar is at the left edge of the face, when the face is observed from the outside of the pallet and the pallet is upright (with respect to the *base* frame, regardless of camera pose). Conversely, `right` indicates that the pillar is at the right edge of the face.

*box*

The box command indicates the pose of a box within the pallet.
They are defined with the string `box` followed by 7 numbers: the box size `size_x`, `size_y`, `size_z`, the position of the center `x`, `y`, `z`, and the rotation around the vertical axis (rad).

*Example*

The following example defines two planes, with two pillars each, and three boxes.

```
# plane: a b c d z_min z_max
# pillar: x y z_min z_max
plane -1    0    0      0    0  0.2 name front_plane
pillar 0    0.425  0.0    0.2 left front_plane
pillar 0    -0.425 0.0    0.2 right front_plane
plane -1    0    0      0.35  0.2 0.6 name back_plane
pillar 0.35  0.425  0.2    0.6 left back_plane
pillar 0.35  -0.425 0.2    0.6 right back_plane
# box: size_x size_y size_z x    y    z     rotation_z
box    0.3    0.85   0.2    0.15 0.0   0.1  0
box    0.3    0.85   0.2    0.45 0.0   0.3  0
box    0.3    0.85   0.2    0.45 0.0   0.5  0
```

**Results**

The action result has these fields:
- `pallet_pose`: pose of the *pallet* reference frame, with respect to the *base* frame.
- `box_poses`: array of poses of each box in the pallet.
- `consensus`: integer, consensus size of the final RANSAC.
- `success`: a Boolean reporting if detection was successful. The consensus of the final RANSAC must be at least 2 to achieve a success. In case of failure, the initial guess will be returned as `pallet_pose`.

**Debug information**

These topics are published by the `pallet_detection_node` with some useful debug information:

- `point_cloud` (sensor_msgs/PointCloud2): point cloud with detected vertical planes in different colors.
- `input_point_cloud` (sensor_msgs/PointCloud2): initial point cloud computed from the depth image.
- `valid_points_cloud` (sensor_msgs/PointCloud2): point clouds where invalid points have been filtered according to some heuristics.
- `depth_image` (sensor_msgs/Image): depth imagewhere invalid points have been filtered according to some heuristics.
- `cluster_image` (sensor_msgs/Image): image with vertical planes in grayscale.
- `edge_image` (sensor_msgs/Image): image with detected lines.
- `plane_image` (sensor_msgs/Image): image with vertical planes in different colors.
- `markers` (visualization_msgs/MarkerArray): visualization of pallet elements (*yellow*: elements of the pallet description, *blue*: elements detected on the image, *green*: elements of the description transformed into the pallet pose, *red*: same as green, but with pose refinement).

Moreover, the TF reference frame of each detected box is published in the form `box_X` with respect to the *base* TF frame.


Saving images
-------------

The node `save_images` is a utility to save the information needed by the `pallet_detection` node, so that it can be used offline.
A launch file `save_images.launch` is also provided.
The launch file also starts the OAK-D Pro camera driver `oak_d_pro.launch` from `simod_camera`.

Upon startup, the node opens an OpenCV window with the current RGB frame from the camera. Press any key except `Esc` to save a frame. Press `Esc` to exit.

The following files are saved:

- `depthX.png`: depth image as 16-bit grayscale PNG file
- `imageX.png`: RGB image as 8-bit color PNG file
- `camera_info.txt`: focal lengths and image center from camera info
- `poseX.txt`: transformation between two TF frames, usually the *base* reference frame and the camera frame

`X` starts from 0 and auto-increments. `X` is reset to 0 if the node is restarted.

Parameters:

- `save_folder_name`: folder where the files will be saved.
- `tf_base_frame` name of the *base* reference TF frame.
- `tf_camera_frame` name of the camera TF frame.
- `depth_image_topic`, `color_image_topic` and `camera_info_topic`: topic names for the depth image, the color image and the camera info, respectively.
- `save_color`, `save_depth`, `save_pose` and `save_camera_info`: boolean flags. Set these to false to disable saving of the respective file.

Pallet detection test
---------------------

The `pallet_detection_test` node (`pallet_detection_test.cpp`) is an example node which calls the action of `pallet_detection_node`. An example launch file `pallet_detection.launch` is also provided.

The node can operate in two modes, depending on the parameter `use_real_camera`. If `use_real_camera` is false, the node loads the RGB and depth images from file and publishes them while the action is executing, to simulate the camera.

If `use_real_camera` is true, the node will not publish to the topics. Instead, in the launch file the `oak_d_pro.launch` file is included to connect to the real camera.

**Parameters**

- `rgb_filename`: name of the color image file (only if `use_real_camera` is false).
- `depth_filename`: name of the color image file (16-bit PNG) (only if `use_real_camera` is false).
- `camera_info_filename`: name of the camera info file (text file) (only if `use_real_camera` is false).
- `expected_pallet_filename`: file name of the pallet description.
- `camera_pose_filename`: file name of the camera pose.
- `initial_guess_filename`: file name of the initial guess.

*Camera info file*

Format of the camera info file is the same which is saved by the `save_images` node:

```
fx 1029.6
fy 1029.11
cx 632.373
cy 365.709
```

where `fx` and `fy` are the focal lengths and `cx`, `cy` is the image center.

*Initial guess file*

Format of the initial guess file is:

```
center   0.0 0.0 0.0
size     4.0 4.0 3.0
rotation 0.0
floor    -0.35
```

where `center` is the initial guess of the pallet center in *base* coordinates,  `size` is the initial guess size, `rotation` is the initial guess rotation, and `floor` is the ground height `floor_z`.

*Camera pose file*

The camera pose file contains a 4x4 3D transformation matrix, with four numbers on each line. Example:

```
0 0 1 0
1 0 0 0
0 1 0 0
0 0 0 1
```

