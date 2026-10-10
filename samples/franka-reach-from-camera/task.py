"""Define a Franka Panda reaching task that finds its target through a camera.

A red cube rests at a random spot on a table in front of the arm. The policy
never receives the cube's coordinates: it sees one fixed camera image of the
scene, plus the arm's own joint positions and velocities. A three-value
Cartesian action moves the fingertips, and differential inverse kinematics
turns that motion into commands for the arm's seven joints.

The value function used during training (the critic) also receives the cube's
true position. The policy (the actor) never does, so a trained policy runs
from the camera and joint readings alone.
"""

from __future__ import annotations

from typing import Tuple

import simulo

# The Franka Panda arm.
franka = simulo.Asset.from_registry("simulo/robot/franka-panda:v2")

app = simulo.App("franka-reach-from-camera")

# The ONE module-level heavy import, deferred under the runtime guard.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only during execution)


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_distance: float,
    rew_scale_fine: float,
    rew_scale_action: float,
    distance_std: float,
    fine_std: float,
    fingertips: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Reward getting the fingertips onto the target point and holding still.

    A wide ``tanh`` shell pulls the fingertips across the table toward the
    cube, a narrow one pays out only in the last few centimetres, and a small
    action penalty stops the arm from jittering once it has arrived.
    """
    distance = torch.norm(target - fingertips, dim=-1)
    rew_coarse = rew_scale_distance * (1.0 - torch.tanh(distance / distance_std))
    rew_fine = rew_scale_fine * (1.0 - torch.tanh(distance / fine_std))
    rew_action = rew_scale_action * torch.sum(torch.square(actions), dim=-1)

    reward: torch.Tensor = rew_coarse + rew_fine + rew_action
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1.
    return reward.view(-1)


def _rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (N, 3) by the w-first unit quaternions ``quat`` (N, 4)."""
    w = quat[:, 0:1]
    xyz = quat[:, 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


class FrankaReachFromCameraTask(simulo.Task):
    """Move a Franka Panda's fingertips above a cube that only a camera can see.

    Actor inputs:

    * ``camera``: one fixed scene camera, RGB uint8 of shape (100, 100, 3).
    * ``joints``: the seven arm joint positions (relative to the home pose)
      and the seven arm joint velocities.

    Critic inputs: the same ``joints`` plus ``privileged``, the fingertip
    position, the target point and the vector between them, all in the arm's
    base frame.

    Action (3 values): a Cartesian move of the fingertip point, at most
    ``action_scale`` metres per step along each axis. The controller turns
    that into joint commands.

    An episode succeeds when the fingertip point finishes within 5 cm of the
    target point: ``target_height_above_cube`` metres above the cube's centre,
    so the open fingers end up just over the cube's top face.
    """

    action_dim = 3
    observation_dim = 14

    # 100 policy steps of 30 ms: physics runs at 200 Hz, 6 steps per action.
    episode_length_s = 3.0
    physics_dt = 0.005
    physics_steps_per_action = 6

    # How far one full-scale action moves the fingertip point, per step [m].
    action_scale = 0.03

    # -- the robot -----------------------------------------------------------

    end_effector = "hand"
    arm_joint_pattern = "joint[1-7]"
    finger_joint_pattern = "finger_joint[12]"
    arm_joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
    # Scene-specific home [rad]: the arm reaches across the table with the
    # hand pointing down, inside both the policy view and fingertip workspace.
    # The sideways shoulder/elbow configuration also clears the table and
    # floor during preview's individual joint sweeps.
    home_arm_positions = (-1.9670, -1.0720, 0.9882, -1.0661, 1.3236, 0.8577, -0.9218)
    home_finger_position = 0.04
    # Uniform noise added to each arm joint at reset [rad].
    reset_joint_noise = 0.05

    # Position-control drives. The robot file carries none, so the task
    # supplies them per joint: (stiffness, damping, effort limit, velocity
    # limit), in N m/rad, N m s/rad, N m and rad/s for the arm, and N/m,
    # N s/m, N and m/s for the fingers. These are the robot's shared task
    # settings, which other simulators reproduce for the same robot file; the
    # limits are the arm's published ones.
    joint_drives = {
        "joint1": (400.0, 80.0, 87.0, 2.175),
        "joint2": (400.0, 80.0, 87.0, 2.175),
        "joint3": (400.0, 80.0, 87.0, 2.175),
        "joint4": (400.0, 80.0, 87.0, 2.175),
        "joint5": (400.0, 80.0, 12.0, 2.61),
        "joint6": (400.0, 80.0, 12.0, 2.61),
        "joint7": (400.0, 80.0, 12.0, 2.61),
        "finger_joint1": (2000.0, 100.0, 50.0, 0.2),
        "finger_joint2": (2000.0, 100.0, 50.0, 0.2),
    }

    # The fingertip point sits this far along the hand's own z axis from the
    # hand frame [m]: midway between the fingertips.
    fingertip_offset = 0.1034
    # Hand orientation held for the whole episode, w-first quaternion in the
    # arm's base frame: pointing straight down, fingers spread along y. This
    # matches the nominal home; reset adds small joint-position noise.
    hand_orientation = (0.0, 0.70710678, 0.70710678, 0.0)

    # The fingertip point may be commanded only inside this box [m].
    fingertip_x_range = (0.25, 0.70)
    fingertip_y_range = (-0.40, 0.40)
    fingertip_z_range = (0.24, 0.65)

    # -- the scene -------------------------------------------------------------

    # A 6 cm red cube lies on a table whose top is 17 cm above the floor.
    cube_size = 0.06
    table_height = 0.17
    # Cube centre sampling region on the table, in the arm's base frame [m].
    cube_x_range = (0.35, 0.60)
    cube_y_range = (-0.25, 0.25)
    # The target point is this far above the cube's centre [m]: 3 cm above
    # its top face, where the open fingers clear the cube.
    target_height_above_cube = 0.06

    # The policy camera, fixed in each environment, in the arm's base frame.
    camera_resolution = 100
    camera_position = (1.30, 0.0, 1.05)
    camera_look_at = (0.475, 0.0, 0.26)
    # 23.533 mm over a 20.955 mm aperture is a 48 degree field of view.
    camera_focal_length = 23.533

    policy_input_schema = simulo.PolicyInputSchema(
        actor=(
            simulo.PolicyInput(
                name="camera",
                shape=(100, 100, 3),
                dtype="uint8",
                kind="image",
                camera="policy_camera",
            ),
            simulo.PolicyInput(name="joints", shape=(14,)),
        ),
        critic=(
            simulo.PolicyInput(name="joints", shape=(14,)),
            simulo.PolicyInput(name="privileged", shape=(9,)),
        ),
    )

    rew_scale_distance = 1.0
    rew_scale_fine = 1.0
    rew_scale_action = -0.01
    distance_std = 0.20
    fine_std = 0.04

    # Framework-injected at runtime by the training base (declared here only so
    # the type checker sees the names the methods read).
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2500.0, color=(0.75, 0.75, 0.75)),
            at="/",
            per_environment=False,
        )
        home = (*self.home_arm_positions, self.home_finger_position, self.home_finger_position)
        self.robot = simulo.Robot(
            asset=franka,
            initial_pose=simulo.Pose.identity(),
            # The arm is position-controlled and holds itself up: gravity off.
            disable_gravity=True,
            articulation=simulo.ArticulationConfig(
                default_joint_positions=list(home),
                default_joint_velocities=[0.0] * len(home),
                fix_root_link=True,
                enable_self_collision=True,
                solver_position_iterations=8,
                solver_velocity_iterations=0,
            ),
            joint_config=[
                simulo.JointConfig(
                    name=name,
                    type="prismatic" if name.startswith("finger") else "revolute",
                    stiffness=stiffness,
                    damping=damping,
                    effort_limit=effort,
                    velocity_limit=velocity,
                )
                for name, (stiffness, damping, effort, velocity) in self.joint_drives.items()
            ],
        )
        scene.add(self.robot, at="/World/Robot")

        table_x = (self.cube_x_range[0] + self.cube_x_range[1]) / 2
        scene.add(
            simulo.Prop(
                simulo.Cuboid(
                    name="table",
                    size=(0.45, 0.80, self.table_height),
                    pose=simulo.Pose.from_xyz(table_x, 0.0, self.table_height / 2),
                    material=simulo.Material(color=(0.22, 0.30, 0.42)),
                    physics=simulo.Physics.rigid(mass=10.0, kinematic=True),
                )
            ),
            at="/World/table",
        )
        # The cube is kinematic: it stays exactly where reset puts it.
        self.cube = simulo.Prop(
            simulo.Cuboid(
                name="cube",
                size=(self.cube_size, self.cube_size, self.cube_size),
                pose=simulo.Pose.from_xyz(0.0, 0.0, self.table_height + self.cube_size / 2),
                material=simulo.Material(color=(0.9, 0.08, 0.06)),
                physics=simulo.Physics.rigid(mass=0.1, kinematic=True),
            )
        )
        scene.add(self.cube, at="/World/cube")

        aperture = 20.955
        self.policy_camera = simulo.Camera(
            width=self.camera_resolution,
            height=self.camera_resolution,
            data_types=["rgb"],
            offset=simulo.SensorOffset.look_at(
                pos=self.camera_position, target=self.camera_look_at
            ),
            spawn=simulo.CameraSpawnConfig(
                focal_length=self.camera_focal_length,
                horizontal_aperture=aperture,
                clipping_range=(0.05, 6.0),
            ),
            purpose="policy",
        )
        scene.add(self.policy_camera, at="/World/policy_camera")

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._arm_ids = self.robot.find_joints(self.arm_joint_pattern)
        self._finger_ids = self.robot.find_joints(self.finger_joint_pattern)
        self._home = torch.zeros(self.robot.num_joints, device=self.device)
        for name, value in zip(self.arm_joint_names, self.home_arm_positions):  # noqa: B905
            self._home[self.robot.find_joints(name)[0]] = value
        self._home[self._finger_ids] = self.home_finger_position

        self.ik = simulo.DifferentialIKController(
            robot=self.robot,
            end_effector=self.end_effector,
            joints=self.arm_joint_pattern,
            ik_method="dls",
            command_type="pose",
        )
        self._orientation = torch.tensor(self.hand_orientation, device=self.device).repeat(
            self.num_envs, 1
        )
        self._offset_local = torch.tensor(
            (0.0, 0.0, self.fingertip_offset), device=self.device
        ).repeat(self.num_envs, 1)
        self._box_low = torch.tensor(
            (self.fingertip_x_range[0], self.fingertip_y_range[0], self.fingertip_z_range[0]),
            device=self.device,
        )
        self._box_high = torch.tensor(
            (self.fingertip_x_range[1], self.fingertip_y_range[1], self.fingertip_z_range[1]),
            device=self.device,
        )
        self.cube_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.cube_pos[:, 2] = self.table_height + self.cube_size / 2
        self._actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    # -- helpers ----------------------------------------------------------------

    def _hand_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        position, orientation = self.robot.get_body_pose_in_base_frame(self.end_effector)
        if position is None:
            zeros = torch.zeros(self.num_envs, 3, device=self.device)
            return zeros, self._orientation
        return position, orientation

    def fingertip_position(self) -> torch.Tensor:
        """Point midway between the fingertips, in the arm's base frame, (num_envs, 3)."""
        position, orientation = self._hand_pose()
        return position + _rotate(orientation, self._offset_local)

    def target_position(self) -> torch.Tensor:
        """Where the fingertips should finish: above the cube's centre, (num_envs, 3)."""
        target = self.cube_pos.clone()
        target[:, 2] += self.target_height_above_cube
        return target

    def _joint_inputs(self) -> torch.Tensor:
        joint_pos = self.robot.state.joint_positions[:, self._arm_ids]
        joint_vel = self.robot.state.joint_velocities[:, self._arm_ids]
        return torch.cat((joint_pos - self._home[self._arm_ids], joint_vel), dim=-1)

    # -- the Task contract --------------------------------------------------------

    def get_observations(self) -> torch.Tensor:
        return self._joint_inputs()

    def get_policy_inputs(self) -> dict[str, torch.Tensor]:
        return {"camera": self.policy_camera.read_rgb(), "joints": self._joint_inputs()}

    def get_critic_inputs(self) -> dict[str, torch.Tensor]:
        fingertips = self.fingertip_position()
        target = self.target_position()
        return {
            "joints": self._joint_inputs(),
            "privileged": torch.cat((fingertips, target, target - fingertips), dim=-1),
        }

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_distance,
            self.rew_scale_fine,
            self.rew_scale_action,
            self.distance_std,
            self.fine_std,
            self.fingertip_position(),
            self.target_position(),
            self._actions,
        )

    def get_dones(self) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return {}, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)
        self._actions = actions
        # Each action moves the fingertip point relative to where it is now,
        # so the command never drifts away from the arm it controls.
        fingertip_goal = self.fingertip_position() + self.action_scale * actions
        fingertip_goal = torch.max(torch.min(fingertip_goal, self._box_high), self._box_low)
        hand_goal = fingertip_goal - _rotate(self._orientation, self._offset_local)
        self.ik.move_to(hand_goal, orientation=self._orientation)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.robot.reset(env_ids)

        # Home pose with a little noise on the seven arm joints only; the
        # fingers stay open and are not part of this task.
        joint_pos = self._home.repeat(len(env_ids), 1)
        arm = self._arm_ids
        joint_pos[:, arm] += torch.empty_like(joint_pos[:, arm]).uniform_(
            -self.reset_joint_noise, self.reset_joint_noise
        )
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_state(joint_pos, velocities=joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # A new cube position on the table.
        count = len(env_ids)
        self.cube_pos[env_ids, 0] = torch.empty(count, device=self.device).uniform_(
            *self.cube_x_range
        )
        self.cube_pos[env_ids, 1] = torch.empty(count, device=self.device).uniform_(
            *self.cube_y_range
        )
        pose = self.cube.default_pose[env_ids].clone()
        pose[:, 0:2] += self.cube_pos[env_ids, 0:2]
        self.cube.set_pose(pose, env_ids=env_ids)
        self._actions[env_ids] = 0.0
