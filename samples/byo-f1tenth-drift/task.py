"""Define a drifting task for a published F1TENTH-compatible race car.

The task uses the organization's `robot/f1tenth:v1` asset and maps throttle
and steering actions to four driven wheels and two steering joints.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Tuple

import simulo

# Your F1TENTH-compatible race car, published to your OWN org catalog. No publisher
# segment => resolved against the caller's own org, not "simulo/...".
f1tenth = simulo.Asset.from_registry("robot/f1tenth:v1")

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06")); see the
# Runtimes docs.
app = simulo.App("byo-f1tenth-drift")

# The ONE module-level heavy import - deferred under the runtime guard so discovery
# records it as a remote import instead of resolving it.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only in execution mode)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a per-env vector by a per-env quaternion (both wxyz / (num_envs, *)).

    This plain typed module-level helper is recursively compiled by the decorated
    reward kernel that calls it and can also be called eagerly.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the conjugate of ``quat`` - world frame -> body frame."""
    quat_conj = quat.clone()
    quat_conj[:, 1:4] = -quat_conj[:, 1:4]
    return _quat_rotate(quat_conj, vec)


def _quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert a per-env wxyz quaternion to roll/pitch/yaw (Euler XYZ), each in radians."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def _track_measure(pos_x: torch.Tensor, pos_y: torch.Tensor, straight: float) -> torch.Tensor:
    """The stadium track's radial/perpendicular distance measure, used for both the
    out-of-bounds check and the cross-track reward.

    On a straight section (``|y| <= straight``): perpendicular distance from the track's
    central axis, ``|x|``. In a turn (``|y| > straight``): radial distance from that
    turn's center, ``(0, +-straight)``. Same shape logic on both sections, matching the
    upstream stadium-track geometry.
    """
    in_turn = torch.abs(pos_y) > straight
    turn_center_y = torch.where(
        pos_y > 0, torch.full_like(pos_y, straight), torch.full_like(pos_y, -straight)
    )
    turn_dist = torch.sqrt(pos_x**2 + (pos_y - turn_center_y) ** 2)
    straight_dist = torch.abs(pos_x)
    return torch.where(in_turn, turn_dist, straight_dist)


@app.runtime.torch_jit
def _compute_car_targets(
    max_speed: float,
    max_steer: float,
    base_length: float,
    base_width: float,
    wheel_radius: float,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a 2D ``[throttle, steer]`` action to per-joint targets - upstream's ``RCCar4WDAction``.

    Returns ``(steer_position_target, wheel_velocity_targets)``: ``steer_position_target``
    is ``tan(steer_angle)`` (shape ``(num_envs,)`` - BOTH steering joints get this same
    value, upstream's simplified-4WD quirk, not pure Ackermann geometry) and
    ``wheel_velocity_targets`` is ``(num_envs, 4)`` in ``[front_left, front_right,
    back_left, back_right]`` order, each independently scaled by the turn-radius geometry.

    ``@app.runtime.torch_jit`` is a no-op marker at submit and real ``torch.jit.script`` on
    execution, so this lives at module level and is torch-free to *define* during
    discovery (its body never runs at submit).
    """
    throttle = torch.clamp(actions[:, 0], -1.0, 1.0)
    steer = torch.clamp(actions[:, 1], -1.0, 1.0)

    # No reverse: a negative throttle clamps to a standstill target, not backward motion.
    target_velocity = torch.clamp(throttle * max_speed, min=0.0)
    target_steer = steer * max_steer

    tan_s = torch.tan(target_steer)
    degenerate = torch.abs(tan_s) < 1.0e-6
    safe_tan_s = torch.where(degenerate, torch.ones_like(tan_s), tan_s)
    # A very large turn radius stands in for "driving straight" (tan_s == 0), avoiding
    # the divide-by-zero the raw geometry would otherwise hit.
    turn_radius = torch.where(degenerate, torch.full_like(tan_s, 1.0e6), base_length / safe_tan_s)

    half_width = base_width / 2.0
    v_front_left = target_velocity * torch.abs(
        torch.sqrt((turn_radius - half_width) ** 2 + base_length**2) / (turn_radius * wheel_radius)
    )
    v_front_right = target_velocity * torch.abs(
        torch.sqrt((turn_radius + half_width) ** 2 + base_length**2) / (turn_radius * wheel_radius)
    )
    v_back_left = target_velocity * torch.abs(
        (turn_radius - half_width) / (turn_radius * wheel_radius)
    )
    v_back_right = target_velocity * torch.abs(
        (turn_radius + half_width) / (turn_radius * wheel_radius)
    )

    wheel_targets = torch.stack([v_front_left, v_front_right, v_back_left, v_back_right], dim=-1)
    return tan_s, wheel_targets


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_side_slip: float,
    rew_scale_vel: float,
    rew_scale_progress: float,
    rew_scale_turn_energy: float,
    rew_scale_cross_track: float,
    rew_scale_tlgr: float,
    rew_scale_out_of_bounds: float,
    slip_threshold: float,
    slip_engage_threshold: float,
    min_forward_speed_for_slip: float,
    max_speed: float,
    straight: float,
    line_radius: float,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel_z: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    steer_pos_mean: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled multi-term drift reward kernel - the ``humanoid``-style weighted-sum
    pattern, ported from WheeledLab's drift task and checked term-by-term against its
    source. This port uses fixed weights in place of a training-iteration curriculum.

    ``@app.runtime.torch_jit`` is a no-op marker at submit and real ``torch.jit.script`` on
    execution, so this lives at module level and is torch-free to *define* during
    discovery (its body never runs at submit). ``_quat_rotate_inverse`` /
    ``_track_measure`` are compiled transitively.
    """
    # side_slip: reward the (clamped) body-frame slip angle, but only inside a genuine
    # "drifting" band - not stationary/crawling, not an unstable spin, not just barely
    # turning.
    body_vel = _quat_rotate_inverse(root_quat, root_lin_vel)
    forward_speed = body_vel[:, 0]
    slip_angle = torch.abs(torch.atan2(body_vel[:, 1], body_vel[:, 0]))

    slip_reward = torch.clamp(slip_angle, min=0.0, max=slip_threshold)
    slip_active = (
        (forward_speed >= min_forward_speed_for_slip)
        & (slip_angle <= slip_threshold)
        & (slip_angle >= slip_engage_threshold)
    )
    side_slip = torch.where(slip_active, slip_reward, torch.zeros_like(slip_reward))

    # vel: penalize deviation of ground speed (world-frame xy velocity norm) from the
    # target cruise speed. Upstream's own default offset (-max_speed**2) is active at
    # this call site - its override is commented out in the source, not omitted - so
    # this term bottoms out AT -max_speed**2 exactly at the target speed (its minimum
    # value over all speeds, not zero) and rises from there as speed deviates. Combined with
    # this term's own negative weight, that makes hitting the target speed a positive
    # reward contribution, not merely a penalty-free one.
    ground_speed = torch.sqrt(root_lin_vel[:, 0] ** 2 + root_lin_vel[:, 1] ** 2)
    vel_term = (ground_speed - max_speed) ** 2 - max_speed**2

    # progress: the angular rate of the car's POSITION about the track centre
    # (rad/s), i.e. real motion around the loop. Upstream's ``track_progress_rate``
    # uses the root yaw rate as a cheap proxy for this; ported faithfully, that
    # proxy can pay the car for spinning inside the corridor instead of lapping it.
    # Using position around the track rewards actual progress while keeping the same
    # scale over a complete lap.
    progress = (pos_x * root_lin_vel[:, 1] - pos_y * root_lin_vel[:, 0]) / (
        pos_x * pos_x + pos_y * pos_y + 1.0e-3
    )

    # turn_energy: reward carrying speed through corners specifically.
    in_turn = torch.abs(pos_y) > straight
    turn_energy = torch.where(in_turn, ground_speed**2, torch.zeros_like(ground_speed))

    # cross_track: penalize perpendicular/radial distance from the racing line at
    # ``line_radius``. Verified against upstream's exact ``cross_track_dist`` source
    # (``track_radius=line_radius, p=1, offset=-1``): squaring and then square-rooting
    # a signed distance from the line is equivalent to this direct absolute-value form.
    track_measure = _track_measure(pos_x, pos_y, straight)
    cross_track = torch.abs(track_measure - line_radius) - 1.0

    # tlgr ("turn-left-go-right"): a counter-steer bonus. Upstream ramps this in only
    # after basic driving is learned (base/starting weight 0.0) - see the module
    # docstring's curriculum gap for why this port keeps the base weight. Computed
    # unconditionally; contributes nothing while its weight is 0.
    tlgr_raw = steer_pos_mean * torch.clamp(root_ang_vel_z, -1.0, 1.0) * -1.0
    tlgr = torch.clamp(tlgr_raw, min=0.0)

    # out_of_bounds: a one-shot penalty applied at the step the episode terminates.
    terminated = reset_terminated.float()

    reward = (
        rew_scale_side_slip * side_slip
        + rew_scale_vel * vel_term
        + rew_scale_progress * progress
        + rew_scale_turn_energy * turn_energy
        + rew_scale_cross_track * cross_track
        + rew_scale_tlgr * tlgr
        + rew_scale_out_of_bounds * terminated
    )
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1.
    return reward.view(-1)


class F1TenthDriftTask(simulo.Task):
    """Drift an F1TENTH-compatible 4WD race car around a stadium-shaped track.

    Observation (14-dim): root position (3) + root orientation as Euler XYZ (3) + root
    linear velocity (3) + root angular velocity (3) + last action (2). Action (2-dim):
    ``[throttle, steer]``. See the module docstring for the full reward/termination/
    domain-randomization shape and the documented gaps against upstream.

    ``with_camera=True`` adds a standalone,
    world-frame, top-down camera framing the whole track, so the MCAP flight recording
    carries a playable h264 video. Training keeps the default ``with_camera=False`` - no
    camera prim, no render cost.
    """

    observation_dim = 14
    action_dim = 2

    episode_length_s = 5.0  # matches upstream

    # -- Vehicle geometry (upstream WheeledLab constants, meters/radians) -----------
    max_speed = 3.0  # [m/s]
    max_steer = 0.488  # [rad]
    base_length = 0.365  # [m]
    base_width = 0.284  # [m]
    wheel_radius = 0.05  # [m]

    # -- Track geometry (upstream WheeledLab constants, meters) ----------------------
    straight = 0.8  # half-length of each straight section
    line_radius = 0.8  # target racing-line radius (straights at x=+-line_radius)
    corner_in_radius = 0.3  # inner keep-out bound of the drivable corridor
    corner_out_radius = 2.0  # outer bound of the drivable corridor

    # -- Reward shape -----------------------------------------------------------------
    slip_threshold = 0.55  # [rad] - above this, treat it as an unstable spin, not a drift
    slip_engage_threshold = 0.25  # [rad] - below this, not really drifting yet
    min_forward_speed_for_slip = 1.0  # [m/s]

    # Fixed at upstream's BASE (pre-curriculum) weights - see the module docstring's
    # curriculum gap for why the ramp is deliberately left out. ``rew_scale_tlgr``
    # at 0.0 means that term never contributes.
    rew_scale_side_slip = 10.0
    rew_scale_vel = -5.0
    rew_scale_progress = 40.0
    rew_scale_turn_energy = 20.0
    rew_scale_cross_track = -50.0
    rew_scale_tlgr = 0.0
    rew_scale_out_of_bounds = -5000.0

    # -- Domain randomization (spawn pose + periodic push only - see module docstring) -
    spawn_pos_noise = 0.5  # [m]
    spawn_yaw_noise = 1.0  # [rad]
    spawn_height = 0.08  # [m] - wheel_radius plus a small clearance to avoid penetration

    # A single fixed ~0.4s interval, a simplification of upstream's two-tier push
    # schedule (0.1-0.4s / 0.8-1.2s) into one; the yaw range below is upstream's LARGER
    # (slower-tier) bound, folded into this one tier rather than split across two.
    push_interval_steps = 24  # ~0.4s at this app's 60Hz action rate (dt=1/120, 2 substeps)
    push_linear_x = 0.1  # [m/s]
    push_linear_y = 0.03  # [m/s]
    push_angular_z = 0.6  # [rad/s]

    # Framework-injected at runtime by ``simulo.core.Task`` / ``LearningEnv`` (declared
    # here only so the type checker sees the names the methods read; PEP 563 strings).
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def __init__(self, with_camera: bool = False):
        super().__init__()
        self._with_camera = with_camera

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/World", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World",
            per_environment=False,
        )
        # Your published F1TENTH-compatible car - floating base (a mobile vehicle, not
        # bolted down).
        self.robot = simulo.Robot(asset=f1tenth, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

        # Single-environment use: a STANDALONE world-frame camera (not attached to any robot
        # body - see the module docstring's note on why) - a fixed top-down shot,
        # high enough to keep the whole stadium loop in frame regardless of where the
        # car is on the track. Not attached to the robot, so no ``add_sensor`` /
        # ordering concern applies; added directly to the scene like ``Terrain`` /
        # ``Light`` above.
        if self._with_camera:
            camera = simulo.Camera(
                width=640,
                height=480,
                data_types=["rgb"],
                update_period=1.0 / 30.0,  # 30 Hz - matches RecordConfig.video_fps
                offset=simulo.SensorOffset.look_at(pos=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)),
                spawn=simulo.CameraSpawnConfig(
                    focal_length=12.0,  # wide enough to keep the whole loop in frame from 5m up
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
            # The scene-path segment's last component becomes the sensor's name (and
            # therefore its recording topic, "overhead_cam") - see Scene._add_sensor's
            # name-derivation rule; keep this snake_case to match the topic named in
            # this app's docstrings/README.
            scene.add(camera, at="/World/overhead_cam", per_environment=False)

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._steer_joint_ids = self.robot.find_joints("rotator_left") + self.robot.find_joints(
            "rotator_right"
        )
        self._wheel_joint_ids = (
            self.robot.find_joints("wheel_front_left")
            + self.robot.find_joints("wheel_front_right")
            + self.robot.find_joints("wheel_back_left")
            + self.robot.find_joints("wheel_back_right")
        )
        self._last_action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        # Staggered per-env phase so every env doesn't push on the same step.
        self._push_counter = torch.randint(
            1, self.push_interval_steps + 1, (self.num_envs,), device=self.device
        ).float()

        # Per-env world-space grid offset (`env_spacing=7.0` in `_make_env`, so each
        # env's copy of the stadium track sits on its own grid cell) - captured once
        # here. Every read of
        # `robot.state.pose` below is WORLD frame; the track geometry (`_track_measure`,
        # `_randomize_spawn`) is written in one env's own LOCAL frame, centered on that
        # env's grid cell, so every position read subtracts this back out and every
        # position write adds it back in. Without this, every env's car would be
        # spawned at the SAME absolute world coordinates (all 256 stacked on one
        # point), not spread across the grid `env_spacing` sets up - training itself
        # would still be correct (every env's reward/termination math already agreed
        # with itself, just in the wrong frame), but the parallel-training visual this
        # sample exists to show would not be.
        origins = env.scene.env_origins
        if origins is not None:
            self._env_origin_xy = origins[:, :2].clone()
        else:
            self._env_origin_xy = torch.zeros((self.num_envs, 2), device=self.device)

    def _local_xy(self, pose: torch.Tensor) -> torch.Tensor:
        """This env's own track-local XY - world XY minus this env's grid origin.

        Env grids only offset X/Y, never Z, so Z stays raw world."""
        return pose[:, 0:2] - self._env_origin_xy

    def get_observations(self) -> torch.Tensor:
        pose = self.robot.state.pose  # (N, 7): [x, y, z, qw, qx, qy, qz]
        local_xy = self._local_xy(pose)
        euler = _quat_to_euler_xyz(pose[:, 3:7])
        return torch.cat(
            [
                local_xy,
                pose[:, 2:3],
                euler,
                self.robot.state.linear_velocity,
                self.robot.state.angular_velocity,
                self._last_action,
            ],
            dim=-1,
        )

    def get_rewards(self) -> torch.Tensor:
        pose = self.robot.state.pose
        local_xy = self._local_xy(pose)
        steer_pos_mean = self.robot.state.joint_positions[:, self._steer_joint_ids].mean(dim=-1)
        return _compute_rewards(
            self.rew_scale_side_slip,
            self.rew_scale_vel,
            self.rew_scale_progress,
            self.rew_scale_turn_energy,
            self.rew_scale_cross_track,
            self.rew_scale_tlgr,
            self.rew_scale_out_of_bounds,
            self.slip_threshold,
            self.slip_engage_threshold,
            self.min_forward_speed_for_slip,
            self.max_speed,
            self.straight,
            self.line_radius,
            pose[:, 3:7],
            self.robot.state.linear_velocity,
            self.robot.state.angular_velocity[:, 2],
            local_xy[:, 0],
            local_xy[:, 1],
            steer_pos_mean,
            self.reset_terminated,
        )

    def get_dones(self) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        pose = self.robot.state.pose
        local_xy = self._local_xy(pose)
        measure = _track_measure(local_xy[:, 0], local_xy[:, 1], self.straight)
        terminated = {
            "car left the track": (measure < self.corner_in_radius)
            | (measure > self.corner_out_radius)
        }
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        self._last_action = actions.clone()
        steer_target, wheel_targets = _compute_car_targets(
            self.max_speed,
            self.max_steer,
            self.base_length,
            self.base_width,
            self.wheel_radius,
            actions,
        )
        # Both steering joints share the SAME position target - upstream's simplified
        # 4WD quirk (see _compute_car_targets' docstring).
        steer_targets = steer_target.view(-1, 1).expand(-1, 2).contiguous()
        self.robot.set_joint_position_target(steer_targets, joint_ids=self._steer_joint_ids)
        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_ids)

    def on_post_physics_step(self) -> None:
        """Apply a small periodic random push - the domain-randomization piece the SDK
        can actually do (see the module docstring's domain-randomization gap)."""
        self._push_counter -= 1.0
        due = self._push_counter <= 0.0
        if bool(torch.any(due).item()):
            env_ids = torch.nonzero(due, as_tuple=False).squeeze(-1)
            n = len(env_ids)
            lin = self.robot.state.linear_velocity[env_ids].clone()
            ang = self.robot.state.angular_velocity[env_ids].clone()
            lin[:, 0] += (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.push_linear_x
            lin[:, 1] += (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.push_linear_y
            ang[:, 2] += (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.push_angular_z
            self.robot.set_root_velocity(torch.cat([lin, ang], dim=-1), env_ids=env_ids)
            self._push_counter[env_ids] = torch.randint(
                1, self.push_interval_steps + 1, (n,), device=self.device
            ).float()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        self._last_action[env_ids] = 0.0
        self._randomize_spawn(env_ids)
        self._push_counter[env_ids] = torch.randint(
            1, self.push_interval_steps + 1, (num_resets,), device=self.device
        ).float()

    def _randomize_spawn(self, env_ids: torch.Tensor) -> None:
        """Sample a random point along the stadium track (in this env's own LOCAL
        frame) and teleport the car there in WORLD frame, with zero velocity - the
        SDK-real half of upstream's domain randomization (spawn position/heading; see
        the module docstring's domain-randomization gap for what this does NOT
        cover)."""
        n = len(env_ids)
        straight = self.straight
        line_radius = self.line_radius

        l1 = 2.0 * straight  # right straight (x=+line_radius), traveling +y
        l2 = math.pi * line_radius  # top turn, center (0, +straight), angle 0 -> pi
        l3 = 2.0 * straight  # left straight (x=-line_radius), traveling -y
        # l4 == l2: bottom turn, center (0, -straight), angle pi -> 2*pi
        perimeter = l1 + l2 + l3 + l2

        s = torch.rand(n, device=self.device) * perimeter
        x = torch.zeros(n, device=self.device)
        y = torch.zeros(n, device=self.device)
        yaw = torch.zeros(n, device=self.device)

        m1 = s < l1
        x = torch.where(m1, torch.full_like(x, line_radius), x)
        y = torch.where(m1, -straight + s, y)
        yaw = torch.where(m1, torch.full_like(yaw, math.pi / 2.0), yaw)

        s2 = s - l1
        m2 = (s >= l1) & (s < l1 + l2)
        angle2 = s2 / line_radius
        x = torch.where(m2, line_radius * torch.cos(angle2), x)
        y = torch.where(m2, straight + line_radius * torch.sin(angle2), y)
        yaw = torch.where(m2, angle2 + math.pi / 2.0, yaw)

        s3 = s - l1 - l2
        m3 = (s >= l1 + l2) & (s < l1 + l2 + l3)
        x = torch.where(m3, torch.full_like(x, -line_radius), x)
        y = torch.where(m3, straight - s3, y)
        yaw = torch.where(m3, torch.full_like(yaw, -math.pi / 2.0), yaw)

        s4 = s - l1 - l2 - l3
        m4 = s >= l1 + l2 + l3
        angle4 = math.pi + s4 / line_radius
        x = torch.where(m4, line_radius * torch.cos(angle4), x)
        y = torch.where(m4, -straight + line_radius * torch.sin(angle4), y)
        yaw = torch.where(m4, angle4 + math.pi / 2.0, yaw)

        x = x + (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.spawn_pos_noise
        y = y + (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.spawn_pos_noise
        yaw = yaw + (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.spawn_yaw_noise

        # x, y above are this env's own LOCAL track coordinates; shift into WORLD
        # frame by this env's grid origin before writing the pose (set_root_pose is
        # WORLD frame, matching the coordinate conversion used by robot resets.
        origin = self._env_origin_xy[env_ids]
        x = x + origin[:, 0]
        y = y + origin[:, 1]

        z = torch.full((n,), self.spawn_height, device=self.device)
        half_yaw = yaw / 2.0
        qw = torch.cos(half_yaw)
        qz = torch.sin(half_yaw)
        qx = torch.zeros(n, device=self.device)
        qy = torch.zeros(n, device=self.device)

        pose = torch.stack([x, y, z, qw, qx, qy, qz], dim=-1)
        self.robot.set_root_pose(pose, env_ids=env_ids)
        self.robot.set_root_velocity(torch.zeros(n, 6, device=self.device), env_ids=env_ids)


def _make_env(num_envs: int, *, camera: bool = False) -> Any:
    """Construct the shared F1TENTH-drift environment (same sim settings across jobs).

    ``env_spacing=7.0``: the stadium track's footprint (bounded by ``corner_out_radius``)
    spans roughly 4m x 5.6m per env, so envs need clearance well past that to avoid
    neighboring tracks overlapping. ``camera=True`` (playback only) adds the overhead Camera
    in ``build`` AND passes ``enable_cameras=True`` - the simulator refuses to spawn
    Camera sensors unless camera rendering is enabled, and this LearningEnv kwarg is what
    wires that through, including headless offscreen rendering.
    """
    return simulo.LearningEnv(
        task=F1TenthDriftTask(with_camera=camera),
        num_envs=num_envs,
        device="cuda",
        env_spacing=7.0,
        headless=True,
        seed=42,
        enable_cameras=camera,
    )


def _ppo_overrides() -> dict[str, Any]:
    """Return PPO settings adapted from the task's upstream training setup.

    The scheduler import stays inside the training path so packaging does not need
    the training dependency. The scheduler target adjusts the learning rate, while
    the separate update threshold stops an oversized PPO update. Reward shaping
    scales values seen by the optimizer without changing recorded task rewards.
    """

    from skrl.resources.schedulers.torch import KLAdaptiveLR

    return {
        "learning_rate": 1e-3,
        # Two different knobs share the name ``kl_threshold``. The scheduler's is a
        # TARGET: the adaptive learning rate rises or falls to keep each update's
        # policy shift near 0.01. The update guard is a CEILING: an update stops taking
        # steps once its shift has already exceeded 0.02. They are deliberately
        # about two to one, so the stop fires only on outliers, not every round.
        "learning_rate_scheduler": functools.partial(KLAdaptiveLR, kl_threshold=0.01),
        "kl_threshold": 0.02,
        "entropy_loss_scale": 0.005,
        "rewards_shaper": lambda rewards, timestep, timesteps: rewards * 0.01,
        "clip_predicted_values": False,
    }


def _train(num_envs: int = 256, max_iterations: int = 500) -> dict[str, Any]:
    """Train the drift policy with PPO.

    Args:
        num_envs: Number of parallel environments to simulate. 256 matches upstream's own
            per-env-cfg default for this task.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict with training ``stats`` and the org asset ref.
    """
    env = _make_env(num_envs)
    trainer = simulo.RLTrainer(
        env=env,
        algorithm="PPO",
        device="cuda",
        seed=42,
        # Only the keys that differ from RLTrainer's own PPO defaults are listed there;
        # ``discount_factor`` (0.99), ``lambda`` (0.95), and ``ratio_clip`` (0.2) already
        # match upstream's requested gamma/lam/clip_param at RLTrainer's own defaults.
        # The actor/critic network shape (upstream requests [64, 64] hidden dims) is NOT
        # reachable here - RLTrainer's policy/value networks are a fixed architecture,
        # not an agent_cfg knob.
        agent_cfg=_ppo_overrides(),
    )

    try:
        stats = trainer.train(max_iterations=max_iterations)
        return {"num_envs": num_envs, "robot_asset": "robot/f1tenth:v1", **stats}
    finally:
        trainer.close()
        env.close()
