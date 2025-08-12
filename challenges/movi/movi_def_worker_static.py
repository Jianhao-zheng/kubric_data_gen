# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

"""

import logging

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.core import look_at_quat
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
os.environ["KUBRIC_USE_GPU"] = "1"
# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 10)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]


def rot_kubric_quat_by_euler_angle(current_quat, rot_angles):
  """Convert a kubric quaternion to a rotation matrix."""
  current_quat       = np.array(current_quat) # the input is a list of four floats
  current_rot = R.from_quat(current_quat[[1,2,3,0]]) # kubric output is scalar first, convert to xyzw
  rotation = R.from_euler('xyz', rot_angles, degrees=True)
  new_rot = current_rot * rotation
  new_quats = new_rot.as_quat() # convert back to xyzw
  
  return list(new_quats[[3,0,1,2]])  # return as a list of four floats

def add_random_jitter_rot(current_quat, max_jitter):
  jitter = np.random.uniform(-max_jitter, max_jitter, size=3)
  return rot_kubric_quat_by_euler_angle(current_quat, jitter)

def add_random_jitter_pos(current_pos, max_jitter):
  jitter = np.random.uniform(-max_jitter, max_jitter, size=3)
  return current_pos + jitter

def get_look_at_direction(current_quat):
  current_quat       = np.array(current_quat) # the input is a list of four floats
  current_rot = R.from_quat(current_quat[[1,2,3,0]]) # kubric output is scalar first, convert to xyzw
  # Kubric uses z-backward, see the negative of the third column of the rotation matrix is the forward direction
  forward = -current_rot.as_matrix()[:,2]

  return forward / np.linalg.norm(forward)

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_static_objects", type=int, default=10,
                    help="minimum number of static (distractor) objects")
parser.add_argument("--max_num_static_objects", type=int, default=10,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--min_num_dynamic_objects", type=int, default=0,
                    help="minimum number of dynamic (tossed) objects")
parser.add_argument("--max_num_dynamic_objects", type=int, default=0,
                    help="maximum number of dynamic (tossed) objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["pure_rotation_single_axis",
                                         "pure_rotation_complex", 
                                         "fixed_random", 
                                         "linear_movement", 
                                         "linear_movement_linear_lookat",
                                         "ego"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=8.0)
parser.add_argument("--max_motion_blur", type=float, default=1.0)


# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=36, frame_rate=12,
                    resolution=512)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

add_motion_blur = rng.choice([True,False], p=[0.1, 0.9])
motion_blur = rng.uniform(0, FLAGS.max_motion_blur)
if not add_motion_blur:
  motion_blur = 0.0
if motion_blur > 0.0:
  logging.info(f"Using motion blur strength {motion_blur}")
else:
  motion_blur = None # handled by blender renderer
  logging.info("Not using motion blur, setting strength to 0.0")

simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True, samples_per_pixel=64,
                   motion_blur=motion_blur)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Populate the scene
# background HDRI
train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.0)
if FLAGS.backgrounds_split == "train":
  logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
  hdri_id = rng.choice(train_backgrounds)
else:
  logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
  hdri_id = rng.choice(test_backgrounds)
background_hdri = hdri_source.create(asset_id=hdri_id)
#assert isinstance(background_hdri, kb.Texture)
logging.info("Using background %s", hdri_id)
scene.metadata["background"] = hdri_id
renderer._set_ambient_light_hdri(background_hdri.filename)

# Dome
dome = kubasic.create(asset_id="dome", name="dome",
                      friction=1.0,
                      restitution=0.0,
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)



def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 8.,
    outer_radius: float = 12.,
    z_offset: float = 0.1,
):
  """Sample a linear path which starts and ends within a half-sphere shell."""
  while True:
    camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                 outer_radius,
                                                                 z_offset))
    direction = rng.rand(3) - 0.5
    movement = direction / np.linalg.norm(direction) * movement_speed
    camera_end = camera_start + movement
    if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
        camera_end[2] > z_offset):
      return camera_start, camera_end

def get_linear_lookat_motion_start_end(
    inner_radius: float = 1.5,
    outer_radius: float = 4.0,
    camera_start: np.ndarray = None,
):
  """Sample a linear path which goes through the workspace center."""
  while True:
    # Sample a point near the workspace center that the path travels through
    camera_through = np.array(
        kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0)
    )
    camera_start_is_not_given = camera_start is None
    if camera_start_is_not_given:
      while True:
        # Sample one endpoint of the trajectory
        camera_start = np.array(
            kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0)
        )
        if camera_start[-1] < inner_radius:
          break

    # Continue the trajectory beyond the point in the workspace center, so the
    # final path passes through that point.
    continuation = rng.rand(1) * 2.0
    camera_end = camera_through + continuation * (camera_through - camera_start)

    if camera_start_is_not_given:
      # Second point will probably be closer to the workspace center than the
      # first point.  Get extra augmentation by randomly swapping first and last.
      if rng.rand(1)[0] < 0.5:
        tmp = camera_start
        camera_start = camera_end
        camera_end = tmp
    return camera_start, camera_end


# Camera
logging.info("Setting up the Camera...")
camera_meta_data = {'motion_blur': motion_blur,'motion_type':FLAGS.camera,}
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "fixed_random":
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
  scene.camera.look_at((0, 0, 0))
elif (
    FLAGS.camera == "linear_movement"
    or FLAGS.camera == "linear_movement_linear_lookat"
):
  # Choose motion mode
  mode = rng.choice(['smooth','jitter'], p=[0.75, 0.25])

  is_panning = FLAGS.camera == "linear_movement_linear_lookat"
  camera_inner_radius = 6.0 if is_panning else 8.0
  camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement),
      inner_radius=8.0,
      outer_radius=15.0,
      z_offset=1.0,
  )
  if is_panning:
    min_distance_to_center = min(np.linalg.norm(camera_start), np.linalg.norm(camera_end))
    lookat_start, lookat_end = get_linear_lookat_motion_start_end(outer_radius = max(0.8*min_distance_to_center,6.0))

  # linearly interpolate the camera position between these two points
  # while keeping it focused on the center of the scene
  # we start one frame early and end one frame late to ensure that
  # forward and backward flow are still consistent for the last and first frames
  per_step_motion = (np.array(camera_end) - np.array(camera_start)) / (FLAGS.frame_end - FLAGS.frame_start + 2)
  for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    interp = ((frame - FLAGS.frame_start + 1) /
              (FLAGS.frame_end - FLAGS.frame_start + 3))
    scene.camera.position = (interp * np.array(camera_start) +
                             (1 - interp) * np.array(camera_end))
    if is_panning:
      scene.camera.look_at(
          interp * np.array(lookat_start)
          + (1 - interp) * np.array(lookat_end)
      )
    else:
      scene.camera.look_at((0, 0, 0))

    
    if mode == 'jitter':
      scene.camera.quaternion = add_random_jitter_rot(scene.camera.quaternion, max_jitter=0.5)
      scene.camera.position = add_random_jitter_pos(scene.camera.position, max_jitter=min(0.1,per_step_motion* 0.5))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)
  camera_meta_data['mode'] = mode
elif FLAGS.camera == "pure_rotation_single_axis":
  # Choose motion mode
  mode = rng.choice(['smooth','jitter'], p=[0.75, 0.25])
  # Choose axis
  axis = rng.choice(['pitch','yaw','roll'])
  # axis = rng.choice(['pitch','yaw'])
  axis_map = {'pitch': np.array([1,0,0]), 'yaw': np.array([0,1,0]), 'roll': np.array([0,0,1])}  # your mapping
  axis_vec = axis_map[axis]

  while True:
    # Set up the initial camera position and orientation
    scene.camera.position = kb.sample_point_in_half_sphere_shell(
        inner_radius=7., outer_radius=15., offset=0.5) #make the z-axis higher
    lookat_start, lookat_end = get_linear_lookat_motion_start_end(outer_radius=6.0)
    scene.camera.look_at(np.array(lookat_start))

    # Maximum rotation angle
    if axis == 'roll':
      deg_total = rng.uniform(20.0, 360.0)  # If it's roll, we are fine with larger angles
    else:
      deg_total = rng.uniform(20.0, 120.0)  # we don't want the rotation to be too large to see the sky, this will be further constrained later

    # Randomly choose the sign of the rotation
    sign = 1 if rng.random() < 0.5 else -1

    final_quat = rot_kubric_quat_by_euler_angle(scene.camera.quaternion, (axis_vec * deg_total * sign).tolist())

    final_forward_direction = get_look_at_direction(final_quat)
    # We don't want the final camera to look at the sky, so we check the z component
    if final_forward_direction[2] > 0.0:
      continue

    # We also don't want the camera's look-at point to be too far from the center
    look_at_center_direction = -scene.camera.position
    look_at_center_xy = look_at_center_direction[:2]
    look_at_center_xy /= np.linalg.norm(look_at_center_xy)
    final_forward_xy = final_forward_direction[:2]
    final_forward_xy /= np.linalg.norm(final_forward_xy)
    if np.dot(look_at_center_xy, final_forward_xy) < 0.5:
      # less than 60 degrees difference
      continue
    break
  
  deg_per_frame_base = deg_total / FLAGS.frame_end

  for frame in range(FLAGS.frame_start-1, FLAGS.frame_end + 2):
    if mode == 'smooth':
        step = deg_per_frame_base
    else:
        step = deg_per_frame_base * rng.uniform(0.5, 1.5)  # higher jitter
    step *= sign
    scene.camera.quaternion = rot_kubric_quat_by_euler_angle(scene.camera.quaternion, (axis_vec * step).tolist())
    # print(scene.camera.quaternion)
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

  camera_meta_data['axis'] = axis
  camera_meta_data['deg_total'] = deg_total
  camera_meta_data['mode'] = mode
  camera_meta_data['sign'] = sign
elif FLAGS.camera == "pure_rotation_complex":
  # Choose motion mode
  mode = rng.choice(['smooth','jitter'], p=[0.75, 0.25])
  # Whether to force rotation on roll
  force_roll = rng.choice([True,False], p=[0.25, 0.75])

  num_lookats = rng.randint(2, 4)  # number of lookat points
  num_lookats = 2

  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=15., offset=0.5)
  distance_to_center = np.linalg.norm(scene.camera.position)

  scene.frame_end = FLAGS.frame_end*(num_lookats-1) # Change number of frames to render
  lookat_end = None
  frame_start = FLAGS.frame_start - 1
  roll_delta = 0.0
  for i in range(num_lookats-1):
    # Last lookat_end is the next lookat_start
    lookat_start, lookat_end = get_linear_lookat_motion_start_end(outer_radius=max(0.8*distance_to_center,6.0),camera_start=lookat_end)

    frame_end = frame_start + FLAGS.frame_end

    # we start one frame early and end one frame late to ensure that
    # forward and backward flow are still consistent for the last and first frames
    if i == 0:
      frame_end += 1
    if i == num_lookats - 2:
      frame_end += 1

    # Randomly choose the roll rotation magnitude per step
    if force_roll:
      roll_movement = rng.uniform(-5.0, 5.0)

    for frame in range(frame_start, frame_end):
      interp = ((frame - frame_start) /
                (frame_end - 1 - frame_start))
      scene.camera.look_at(
          interp * np.array(lookat_end)
          + (1 - interp) * np.array(lookat_start)
      )

      if force_roll:
        roll_delta += roll_movement
        scene.camera.quaternion = rot_kubric_quat_by_euler_angle(scene.camera.quaternion, (np.array([0,0,1])*roll_delta))

      if mode == 'jitter':
        scene.camera.quaternion = add_random_jitter_rot(scene.camera.quaternion, max_jitter=0.5)
      scene.camera.keyframe_insert("position", frame)
      scene.camera.keyframe_insert("quaternion", frame)

      # Next start_frame will be the current end_frame
    frame_start = frame_end

  # camera_meta_data['num_lookats'] = num_lookats # Useless, we always have 2 lookats
  camera_meta_data['mode'] = mode
  camera_meta_data['force_roll'] = str(force_roll) # Boolean is not json serializable
elif FLAGS.camera == "ego":
  camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
  )
  # # First, get a fixed lookat point
  # lookat_point = np.array((0, 0, 0))
  
  # # Sample a starting camera position
  # camera_start = kb.sample_point_in_half_sphere_shell(
  #     inner_radius=10., outer_radius=15., offset=0.1)
  
  # Calculate the optical axis direction (from camera to lookat point)
  optical_axis = camera_end - camera_start
  optical_axis = optical_axis / np.linalg.norm(optical_axis)
  
  # # Calculate the end position by moving along the optical axis
  # direction = 1 if rng.random() < 0.5 else -1
  # movement_distance = rng.uniform(low=4., high=FLAGS.max_camera_movement)
  # camera_end = camera_start + direction * optical_axis * movement_distance
  
  for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    interp = ((frame - FLAGS.frame_start + 1) /
              (FLAGS.frame_end - FLAGS.frame_start + 3))
    current_position = (interp * np.array(camera_start) +
                       (1 - interp) * np.array(camera_end))
    scene.camera.position = current_position
    
    # Always look at the fixed lookat point
    scene.camera.look_at(camera_end)

    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)



# ---- Object placement ----
train_split, test_split = gso.get_test_split(fraction=0.0)
if FLAGS.objects_split == "train":
  logging.info("Choosing one of the %d training objects...", len(train_split))
  active_split = train_split
else:
  logging.info("Choosing one of the %d held-out objects...", len(test_split))
  active_split = test_split



# add STATIC objects
num_static_objects = rng.randint(FLAGS.min_num_static_objects,
                                 FLAGS.max_num_static_objects+1)
logging.info("Randomly placing %d static objects:", num_static_objects)
for i in range(num_static_objects):
  obj = gso.create(asset_id=rng.choice(active_split))
  assert isinstance(obj, kb.FileBasedObject)
  scale = rng.uniform(0.75, 3.0)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION,
                           rng=rng)
  obj.friction = 1.0
  obj.restitution = 0.0
  obj.metadata["is_dynamic"] = False
  logging.info("    Added %s at %s", obj.asset_id, obj.position)


logging.info("Running 100 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-100, frame_end=0)


# stop any objects that are still moving and reset friction / restitution
for obj in scene.foreground_assets:
  if hasattr(obj, "velocity"):
    obj.velocity = (0., 0., 0.)
    obj.friction = 0.5
    obj.restitution = 0.5


dome.friction = FLAGS.floor_friction
dome.restitution = FLAGS.floor_restitution



# # Add DYNAMIC objects
# num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects,
#                                   FLAGS.max_num_dynamic_objects+1)
# logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
# for i in range(num_dynamic_objects):
#   obj = gso.create(asset_id=rng.choice(active_split))
#   assert isinstance(obj, kb.FileBasedObject)
#   scale = rng.uniform(0.75, 3.0)
#   obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
#   obj.metadata["scale"] = scale
#   scene += obj
#   kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
#                            rng=rng)
#   obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
#                   [obj.position[0], obj.position[1], 0])
#   obj.metadata["is_dynamic"] = True
#   logging.info("    Added %s at %s", obj.asset_id, obj.position)



# if FLAGS.save_state:
#   logging.info("Saving the simulator state to '%s' prior to the simulation.",
#                output_dir / "scene.bullet")
#   simulator.save_state(output_dir / "scene.bullet")

# # Run dynamic objects simulation
# logging.info("Running the simulation ...")
# animation, collisions = simulator.run(frame_start=0,
#                                       frame_end=scene.frame_end+1)

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' ",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")


logging.info("Rendering the scene ...")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                             if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets)


# Save to image files
kb.write_image_dict(data_stack, output_dir)
# kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir,max_write_threads=16)

kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                  visible_foreground_assets)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
    'camera_meta_data': camera_meta_data,
})
# kb.write_json(filename=output_dir / "events.json", data={
#     "collisions":  kb.process_collisions(
#         collisions, scene, assets_subset=visible_foreground_assets),
# })

kb.done()
