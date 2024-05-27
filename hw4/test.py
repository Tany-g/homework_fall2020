import os
import numpy as np
import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
# import gym
import torch
import time


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat



#creat sim
gym = gymapi.acquire_gym()
physics_engine = gymapi.SIM_PHYSX
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81
sim=gym.create_sim(0, 0, physics_engine, sim_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)


asset_root = "/home/tany/PROJECT/IsaacGymEnvs/assets"
franka_asset_file = "urdf/LiteParallelGriper13/urdf/LiteParallelGriper13.urdf"
# if "asset" in cfg["env"]:
#     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                               cfg["env"]["asset"].get("assetRoot", asset_root))
#     franka_asset_file = cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

# load franka asset
asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = False
asset_options.disable_gravity = True
asset_options.thickness = 0.001
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
asset_options.use_mesh_materials = True


device = "cuda" if torch.cuda.is_available() else "cpu"

franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)
franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 5000, 5000], dtype=torch.float, device=device)
franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=device)

# Create table asset
table_pos = [0.0, 0.0, 1.0]
table_thickness = 0.05
table_opts = gymapi.AssetOptions()
table_opts.fix_base_link = True
table_asset = gym.create_box(sim, *[0.8, 0.8, table_thickness], table_opts)

# Create table stand asset
table_stand_height = 0.000001
table_stand_pos = [-0.25, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
table_stand_opts = gymapi.AssetOptions()
table_stand_opts.fix_base_link = True
table_stand_asset = gym.create_box(sim, *[0.2, 0.2, table_stand_height], table_opts)

cubeA_size = 0.030
cubeB_size = 0.060

# Create cubeA asset
cubeA_opts = gymapi.AssetOptions()
cubeA_asset = gym.create_box(sim, *([cubeA_size] * 3), cubeA_opts)
cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

# Create cubeB asset
cubeB_opts = gymapi.AssetOptions()
cubeB_asset = gym.create_box(sim, *([cubeB_size] * 3), cubeB_opts)
cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

num_franka_bodies = gym.get_asset_rigid_body_count(franka_asset)
num_franka_dofs = gym.get_asset_dof_count(franka_asset)

print("num franka bodies: ", num_franka_bodies)
print("num franka dofs: ", num_franka_dofs)

# set franka dof properties
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_dof_lower_limits = []
franka_dof_upper_limits = []
_franka_effort_limits = []
for i in range(num_franka_dofs):
    franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    if physics_engine == gymapi.SIM_PHYSX:
        franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
        franka_dof_props['damping'][i] = franka_dof_damping[i]
    else:
        franka_dof_props['stiffness'][i] = 7000.0
        franka_dof_props['damping'][i] = 50.0

    franka_dof_lower_limits.append(franka_dof_props['lower'][i])
    franka_dof_upper_limits.append(franka_dof_props['upper'][i])
    _franka_effort_limits.append(franka_dof_props['effort'][i])

franka_dof_lower_limits = to_torch(franka_dof_lower_limits, device=device)
franka_dof_upper_limits = to_torch(franka_dof_upper_limits, device=device)
_franka_effort_limits = to_torch(_franka_effort_limits, device=device)
franka_dof_speed_scales = torch.ones_like(franka_dof_lower_limits)
franka_dof_speed_scales[[6, 7]] = 0.1
franka_dof_props['effort'][6] = 200
franka_dof_props['effort'][7] = 200

# Define start pose for franka
franka_start_pose = gymapi.Transform()
franka_start_pose.p = gymapi.Vec3(-0.25, 0.0, 1.0 + 0.07 + table_thickness / 2 + table_stand_height)
franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1, -3.14/4)

# Define start pose for table
table_start_pose = gymapi.Transform() # type: ignore
table_start_pose.p = gymapi.Vec3(*table_pos) # type: ignore
table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # type: ignore
_table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

reward_settings={}
reward_settings["table_height"] = _table_surface_pos[2]

# Define start pose for table stand
table_stand_start_pose = gymapi.Transform()
table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
cubeA_start_pose = gymapi.Transform()
cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

cubeB_start_pose = gymapi.Transform()
cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# compute aggregate size
num_franka_bodies = gym.get_asset_rigid_body_count(franka_asset)
num_franka_shapes = gym.get_asset_rigid_shape_count(franka_asset)
max_agg_bodies = num_franka_bodies + 4  # 1 for table, table stand, cubeA, cubeB
max_agg_shapes = num_franka_shapes + 4  # 1 for table, table stand, cubeA, cubeB

frankas = []
envs = []

# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise ValueError("Failed to create Viewer")
# cam_pos = gymapi.Vec3(3, 3, 2)  # 举例，根据需要调整
# cam_target = gymapi.Vec3(0, 0, 0)  # 举例，根据需要调整
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

spacing=1

lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)

asset_root = "/home/tany/PROJECT/IsaacGymEnvs/assets"
franka_asset_file = "urdf/LiteParallelGriper13/urdf/LiteParallelGriper13.urdf"


# env params
envSpacing= 1.5
episodeLength= 300
enableDebugVis= True

clipObservations= 5.0
clipActions= 1.0

startPositionNoise= 0.15
startRotationNoise= 0.785
xarmPositionNoise= 0.0
xarmRotationNoise= 0.0
xarmDofNoise= 0.25

aggregateMode= 3

actionScale= 1.0
distRewardScale= 0.5
liftRewardScale= 1.5
alignRewardScale= 2.0
stackRewardScale= 16
aggregate_mode =3
controlType= "osc"  # options are {joint_tor, osc}

num_envs=32

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError("Failed to create viewer")

# Set camera position
cam_pos = gymapi.Vec3(3, 3, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Create environments
for i in range(num_envs):
    # create env instance
    env_ptr = gym.create_env(sim, lower, upper, 5)

    # Create actors and define aggregate group appropriately depending on setting
    # NOTE: franka should ALWAYS be loaded first in sim!
    if aggregate_mode >= 3:
        gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

    # Create franka
    # Potentially randomize start pose
    if xarmPositionNoise > 0:
        rand_xy = xarmPositionNoise * (-1. + np.random.rand(2) * 2.0)
        franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                        1.0 + table_thickness / 2 + table_stand_height)
    # else:
    #     rand_xy = xarmPositionNoise * (-1. + np.random.rand(2) * 2.0)
    #     franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)

    if xarmRotationNoise > 0:
        rand_rot = torch.zeros(1, 3)
        rand_rot[:, -1] = xarmRotationNoise * (-1. + np.random.rand() * 2.0)
        new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
        franka_start_pose.r = gymapi.Quat(*new_quat)
    franka_actor = gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
    gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

    if aggregate_mode == 2:
        gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

    # Create table
    table_actor = gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
    table_stand_actor = gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                            i, 1, 0)

    if aggregate_mode == 1:
        gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

    # Create cubes
    _cubeA_id = gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
    _cubeB_id = gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
    # Set colors
    gym.set_rigid_body_color(env_ptr, _cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
    gym.set_rigid_body_color(env_ptr, _cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

    if aggregate_mode > 0:
        gym.end_aggregate(env_ptr)

    # Store the created env pointers
    envs.append(env_ptr)
    frankas.append(franka_actor)

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    a=gym.simulate(sim)
    results=gym.fetch_results(sim, True)
    print(results)
    # Step graphics
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # Sync frame rate
    time.sleep(1 / 60)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
# Setup init state buffer
_init_cubeA_state = torch.zeros(num_envs, 13, device=device)
_init_cubeB_state = torch.zeros(num_envs, 13, device=device)

# # Setup data
# init_data()



