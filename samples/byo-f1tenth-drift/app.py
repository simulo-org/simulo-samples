"""Bring your own robot: publish an F1TENTH-compatible race car to your catalog, then
train a drifting policy around a stadium-shaped track.

Like ``byo-urdf-arm``, this sample consumes an asset published to **your own
organization's catalog**, never the Simulo catalog: ``robot/f1tenth:v1``, a 4WD,
2-wheel-steer race car built from ``assets/robot/f1tenth/f1tenth.usd`` at the root of
this repository. The reference carries no publisher segment, so it resolves against the
organization you are signed in as. Publish the asset before you run this app;
``README.md`` next to this file has the command. A ``completed`` result from ``train``
is proof the platform accepted your upload, validated it in the cloud, mounted the
version you published, and trained a real drifting policy on it — not a scripted sweep,
not a built-in.

This sample has two jobs. ``train`` builds the same shape every training sample here
uses — a module-level ``simulo.Task``, ``simulo.LearningEnv`` and ``simulo.RLTrainer``
inside a ``@app.job``, a checkpoint saved to a durable ``simulo.Volume`` — then exports a
TorchScript policy alongside the trainer's own checkpoint. ``rollout`` mirrors
``cartpole-eval``'s inference-and-recording job: it plays the exported policy back with
``simulo.RLPlayer`` and records an MCAP flight recording, with embedded video, through
``simulo.RecordConfig``.

Ported task, from WheeledLab (an external open-source project)
----------------------------------------------------------------
The task, track geometry, action mapping, and reward shape are ported from WheeledLab's
own drift task (``UWRobotLearning/WheeledLab``, BSD-3-Clause, University of Washington —
see ``README.md`` for the full attribution). Six joints, each resolved on its exact
name: steering ``rotator_left`` / ``rotator_right``; wheels ``wheel_front_left`` /
``wheel_front_right`` / ``wheel_back_left`` / ``wheel_back_right``.

What this sample shows
-----------------------
* An asset reference with no publisher segment, ``robot/f1tenth:v1``, resolving against
  your own organization rather than the Simulo catalog.
* Observation (14 values): root position (3, world) + root orientation as Euler XYZ (3)
  + root linear velocity (3, world) + root angular velocity (3, world) + last action (2).
  No sensor observations at all — the published asset carries no lidar or other sensors,
  and upstream's own task never reads any either.
* Action (2 values): ``[throttle, steer]``, both in ``[-1, 1]``. A single
  steering-angle-derived turn radius drives four independently scaled wheel velocity
  targets (real Ackermann-esque turn-radius geometry), while both steering joints share
  one position target, ``tan(steer_angle)`` — upstream's own simplified-4WD quirk,
  reproduced here rather than corrected into pure Ackermann geometry. See
  ``_compute_car_targets``.
* A single weighted-sum reward (``_compute_rewards``), the same multi-term pattern the
  other training samples here use: side-slip (rewarded only inside a real-drift band),
  speed-tracking, track progress, corner "turn energy", a cross-track penalty against the
  racing line, a counter-steer bonus, and an out-of-bounds termination penalty.
* Termination when the car leaves the drivable corridor (radial/perpendicular distance
  from the racing line outside a fixed inner/outer bound); otherwise every episode
  truncates at the 5 s time limit.

This port makes a few deliberate simplifications against the upstream task, stated here
rather than hidden:

* Upstream ramps three reward weights over the course of training: the side-slip reward
  grows, a delayed counter-steer bonus phases in once basic driving is learned, and the
  out-of-bounds penalty grows more severe. Its clock is a global simulation-step count
  divided by the episode length, which a Simulo ``Task`` can keep for itself — the
  mechanism is portable, not missing. This port leaves it out on evidence instead: at
  this trainer's fixed rollout length, the default training budget covers too few of
  upstream's "episodes" for a faithful schedule to fire more than once, and a ramp
  compressed to fit was measured to hold the lap less reliably and to end in a diverged
  update more often. All three weights are fixed at upstream's starting (pre-ramp)
  values instead. The concrete effect: the counter-steer bonus (``rew_scale_tlgr``)
  starts at ``0.0`` upstream, so it is computed every step but never contributes reward
  here.
* Upstream sets per-joint actuator limits (steering velocity/effort, wheel
  effort/velocity). The public robot-authoring surface has no value type for that yet, so
  this port relies on the published asset's own joint-drive defaults instead.
* Upstream randomizes tire friction, actuator damping, and body mass every episode, per
  environment, on top of spawn pose and a periodic push. ``simulo.Robot(...)`` does
  accept ``mass_scale`` and ``actuator_gains`` — but only as constructor arguments,
  applied once, uniformly, to every parallel environment when the scene is built, with
  no way to re-roll them per episode the way a spawn pose is re-rolled. Tire friction has
  no setter at all on the public robot-authoring surface — only joint state/targets and
  root pose/velocity — so this port randomizes only the spawn pose (position and
  heading, at reset) and a periodic small velocity push, folding upstream's two-tier
  push schedule into one.
* Upstream also requests a smaller actor/critic network (``[64, 64]``, ELU activation).
  The trainer's own network architecture is fixed (``[256, 128, 64]``, also ELU), so this
  port trains with that fixed shape instead.

The camera ``rollout`` uses (see ``build``) is a standalone, world-frame, top-down camera
added directly to the scene rather than attached to any robot body — a link name on a car
you bring yourself is never known in advance, so this avoids guessing one a camera would
silently fail to attach to.

Run it
------
Sign in with ``simulo login``, publish ``assets/robot/f1tenth/``, then run each job in
turn::

    simulo run samples/byo-f1tenth-drift/app.py --job train --num-envs 256 --max-iterations 500
    simulo run samples/byo-f1tenth-drift/app.py --job rollout --num-steps 300

``--num-envs`` and ``--max-iterations`` are ``train``'s own parameters; ``--num-steps``
is ``rollout``'s. See ``README.md`` for the publish command and what to expect from a
run.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Tuple

import simulo

# Your F1TENTH-compatible race car, published to your own organization's catalog. The
# reference carries no publisher segment, so it resolves against the organization you
# are signed in as rather than against the Simulo catalog.
f1tenth = simulo.Asset.from_registry("robot/f1tenth:v1")

# Two named, durable volumes: one for the trained checkpoint and the exported policy,
# one for the MCAP recording and the JSON rollout summary — the same split
# ``cartpole-eval`` uses for its own train -> rollout lifecycle. These lines only
# declare metadata; nothing is created at packaging time. Each job reads the volume's
# real directory through ``.path``, which resolves only inside a running job.
checkpoints = simulo.Volume.from_name("byo-f1tenth-drift-checkpoints", create_if_missing=True)
reports = simulo.Volume.from_name("byo-f1tenth-drift-reports", create_if_missing=True)

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see https://docs.simulo.ai/concepts/runtimes/.
app = simulo.App("byo-f1tenth-drift", mounts={"/checkpoints": checkpoints, "/reports": reports})

# The one heavy import, deferred: on your machine this block records the import
# instead of resolving it; in the cloud it is a plain import.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)

# Stable filenames inside the checkpoint volume, shared across the two jobs.
_CHECKPOINT_FILE = "f1tenth_drift_final.pt"
_POLICY_FILE = "f1tenth_drift_policy.pt"


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a per-env vector by a per-env quaternion (both wxyz, shape (num_envs, *)).

    Same helper as ``samples/humanoid/app.py`` (duplicated per this repo's per-sample
    convention rather than shared) — a plain typed module-level function, recursively
    compiled by ``torch.jit.script`` when called from an ``@app.runtime.torch_jit``
    kernel, and callable eagerly too.
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
    """Rotate ``vec`` by the conjugate of ``quat`` — world frame -> body frame."""
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
    """Map a 2D ``[throttle, steer]`` action to per-joint targets — upstream's ``RCCar4WDAction``.

    Returns ``(steer_position_target, wheel_velocity_targets)``: ``steer_position_target``
    is ``tan(steer_angle)`` (shape ``(num_envs,)`` — BOTH steering joints get this same
    value, upstream's simplified-4WD quirk, not pure Ackermann geometry) and
    ``wheel_velocity_targets`` is ``(num_envs, 4)`` in ``[front_left, front_right,
    back_left, back_right]`` order, each independently scaled by the turn-radius geometry.

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
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
    """JIT-compiled multi-term drift reward kernel — the same multi-term weighted-sum
    pattern the other training samples here use, ported from WheeledLab's drift task.
    See the module docstring for what this port simplifies against upstream.

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
    ``_quat_rotate_inverse`` and ``_track_measure`` are compiled transitively.
    """
    # side_slip: reward the (clamped) body-frame slip angle, but only inside a genuine
    # "drifting" band — not stationary/crawling, not an unstable spin, not just barely
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
    # this call site — its override is commented out in the source, not omitted — so
    # this term bottoms out AT -max_speed**2 exactly at the target speed (its minimum
    # value over all speeds, not zero) and rises from there as speed deviates. Combined with
    # this term's own negative weight, that makes hitting the target speed a positive
    # reward contribution, not merely a penalty-free one.
    ground_speed = torch.sqrt(root_lin_vel[:, 0] ** 2 + root_lin_vel[:, 1] ** 2)
    vel_term = (ground_speed - max_speed) ** 2 - max_speed**2

    # progress: the angular rate of the car's POSITION about the track centre (rad/s),
    # i.e. real motion around the loop. Upstream's own progress term uses the root yaw
    # rate as a cheap proxy for this; ported faithfully, that proxy paid the car for
    # spinning inside the corridor instead of lapping it. In an early comparison of the
    # two formulations (a design measurement for this term's shape, from before this
    # file's KL-guard fix and not comparable to the training-quality figures in
    # README.md), every surviving episode under the yaw-rate proxy swept 1,700-2,250
    # degrees of heading in 5s while its position covered under one lap, and only a
    # minority of survivors completed a lap; under this position-based term, nearly all
    # did. The two terms are NOT equal pointwise (the yaw-rate proxy is 3.75 rad/s in
    # the turns and 0 on the straights, this term is 1.87 and 3.74 respectively) but
    # both integrate to exactly 2*pi per lap, so their time-mean over a lap matches
    # (2.289 vs 2.291 rad/s) — the weight and the reward scale are unchanged for a car
    # that actually laps; only spinning in place stops paying.
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
    # after basic driving is learned (base/starting weight 0.0) — see the module
    # docstring's curriculum gap. Computed unconditionally; contributes nothing while
    # its weight is 0.
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

    ``with_camera=True`` (the ``rollout`` job's env only) additionally adds a standalone,
    world-frame, top-down camera framing the whole track, so the MCAP flight recording
    carries a playable h264 video. Training keeps the default ``with_camera=False`` — no
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
    slip_threshold = 0.55  # [rad] — above this, treat it as an unstable spin, not a drift
    slip_engage_threshold = 0.25  # [rad] — below this, not really drifting yet
    min_forward_speed_for_slip = 1.0  # [m/s]

    # Fixed at upstream's BASE (pre-curriculum) weights — see the module docstring's
    # curriculum gap. ``rew_scale_tlgr`` starting at 0.0 means that term never
    # contributes without the ramp this port cannot implement.
    rew_scale_side_slip = 10.0
    rew_scale_vel = -5.0
    rew_scale_progress = 40.0
    rew_scale_turn_energy = 20.0
    rew_scale_cross_track = -50.0
    rew_scale_tlgr = 0.0
    rew_scale_out_of_bounds = -5000.0

    # -- Domain randomization (spawn pose + periodic push only — see module docstring) -
    spawn_pos_noise = 0.5  # [m]
    spawn_yaw_noise = 1.0  # [rad]
    spawn_height = 0.08  # [m] — wheel_radius plus a small clearance to avoid penetration

    # A single fixed ~0.4s interval, a simplification of upstream's two-tier push
    # schedule (0.1-0.4s / 0.8-1.2s) into one; the yaw range below is upstream's LARGER
    # (slower-tier) bound, folded into this one tier rather than split across two.
    push_interval_steps = 24  # ~0.4s at this app's 60Hz action rate (dt=1/120, 2 substeps)
    push_linear_x = 0.1  # [m/s]
    push_linear_y = 0.03  # [m/s]
    push_angular_z = 0.6  # [rad/s]

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings
    # and never shadow the inherited values.
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
        # Your published race car, on a floating base: free to move rather than
        # bolted down.
        self.robot = simulo.Robot(asset=f1tenth, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

        # Rollout-only: a standalone world-frame camera (not attached to any robot
        # body — see the module docstring's note on why) — a fixed top-down shot,
        # high enough to keep the whole stadium loop in frame regardless of where the
        # car is on the track. Not attached to the robot, so no ordering concern
        # applies; added directly to the scene like ``Terrain`` / ``Light`` above.
        if self._with_camera:
            camera = simulo.Camera(
                width=640,
                height=480,
                data_types=["rgb"],
                update_period=1.0 / 30.0,  # 30 Hz — matches RecordConfig.video_fps
                offset=simulo.SensorOffset.look_at(pos=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0)),
                spawn=simulo.CameraSpawnConfig(
                    focal_length=12.0,  # wide enough to keep the whole loop in frame from 5m up
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
            # The last segment of the scene path becomes the sensor's name (and
            # therefore its recording topic, "overhead_cam") — keep this snake_case
            # to match the topic named in this sample's README.
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
        # environment's copy of the stadium track sits on its own grid cell) —
        # captured once here. Every read of `robot.state.pose` below is world frame;
        # the track geometry (`_track_measure`, `_randomize_spawn`) is written in one
        # environment's own local frame, centered on that environment's grid cell, so
        # every position read subtracts this back out and every position write adds
        # it back in.
        origins = env.scene.env_origins
        if origins is not None:
            self._env_origin_xy = origins[:, :2].clone()
        else:
            self._env_origin_xy = torch.zeros((self.num_envs, 2), device=self.device)

    def _local_xy(self, pose: torch.Tensor) -> torch.Tensor:
        """This environment's own track-local XY — world XY minus its grid origin.

        Environment grids only offset X/Y, never Z, so Z stays raw world."""
        return pose[:, 0:2] - self._env_origin_xy

    def get_observations(self) -> torch.Tensor:
        # robot.state is the supported, typed way to read live state. robot.internals
        # is the raw escape hatch; see
        # https://docs.simulo.ai/concepts/scene-robot-world/.
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

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pose = self.robot.state.pose
        local_xy = self._local_xy(pose)
        measure = _track_measure(local_xy[:, 0], local_xy[:, 1], self.straight)
        terminated = (measure < self.corner_in_radius) | (measure > self.corner_out_radius)
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
        # Both steering joints share the SAME position target — upstream's simplified
        # 4WD quirk (see _compute_car_targets' docstring).
        steer_targets = steer_target.view(-1, 1).expand(-1, 2).contiguous()
        self.robot.set_joint_position_target(steer_targets, joint_ids=self._steer_joint_ids)
        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_ids)

    def on_post_physics_step(self) -> None:
        """Apply a small periodic random push — the piece of upstream's domain
        randomization the public robot-authoring surface can actually do (see the
        module docstring)."""
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
        """Sample a random point along the stadium track and teleport the car there,
        with zero velocity — the piece of upstream's domain randomization the public
        robot-authoring surface can do (spawn position/heading; see the module
        docstring for what this does NOT cover)."""
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

        # x, y above are this environment's own local track coordinates; shift into
        # world frame by this environment's grid origin before writing the pose
        # (`set_root_pose` is world frame).
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
    """Construct the shared F1TENTH-drift environment (same simulation settings across
    both jobs).

    ``env_spacing=7.0``: the stadium track's footprint (bounded by ``corner_out_radius``)
    spans roughly 4m x 5.6m per env, so envs need clearance well past that to avoid
    neighboring tracks overlapping. ``camera=True`` (rollout only) adds the overhead
    Camera in ``build`` and passes ``enable_cameras=True``: the simulator refuses to
    spawn Camera sensors unless camera rendering is enabled, and this ``LearningEnv``
    argument is what wires that through, including headless offscreen rendering.
    """
    return simulo.LearningEnv(
        task=F1TenthDriftTask(with_camera=camera),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=7.0,
        headless=True,
        seed=42,
        enable_cameras=camera,
    )


def _ppo_overrides() -> dict[str, Any]:
    """The PPO settings that differ from ``simulo.RLTrainer``'s own defaults.

    Called from inside the ``train`` job body only, so its one heavy import (skrl's
    KL-adaptive scheduler, an execution-time-only dependency the trainer itself already
    carries) never runs at submit: the app stays torch-free to discover and package.

    Upstream trains this task with ``learning_rate=1e-3``, ``schedule="adaptive"``,
    ``desired_kl=0.01``, ``entropy_coef=0.005``. Four of the settings below exist to
    give this trainer's PPO the same effective optimizer and to keep it stable, and
    each closes a measured failure of an earlier version of this sample (256
    environments x 500 iterations on a local RTX 3090, traced per update). Two of
    them share the name ``kl_threshold`` and are different knobs; the comment beside
    them in the dict, and their two bullets here, say how:

    * ``learning_rate_scheduler``: a KL-adaptive rule at upstream's ``desired_kl`` (halve
      the learning rate at 2x the KL target, raise it 1.5x under half of it). With the
      learning rate pinned at 1e-3, the policy's own noise climbed monotonically and most
      runs ended in an all-NaN update; with the scheduler, the noise falls instead.
      Divergence is still intermittent — see the README's "What to expect" —
      which is what the best-checkpoint export further down is for. The scheduler takes
      its threshold as a constructor keyword, and this trainer's ``agent_cfg`` cannot
      carry a nested keyword-arguments section for it, so it is bound with
      ``functools.partial``.
    * ``kl_threshold`` (the agent's, not the scheduler's): skrl's early-stop for a
      single PPO update — a ceiling rather than a target. Each learning epoch stops
      taking further gradient steps as soon as a mini-batch's approximate KL against
      the rollout policy exceeds 0.02, twice the scheduler's own target, so one
      oversized step is not compounded by more steps in the same epoch. The
      scheduler's own reaction to an oversized update — collapsing the learning rate
      — comes too late once the policy has already moved that far; this guard
      measurably cut how often local testing ended in a non-finite update. A run can
      still turn out poorly, which is what the best-checkpoint export further down
      is for.
    * ``rewards_shaper``: scale the reward the *agent* sees by 0.01. This task's episode
      return is a few tens of thousands (upstream's own reward weights), which the
      trainer's value function cannot fit without this scaling — its clipped value loss
      saturates against returns that large, and the critic never learns. Shaping is
      applied after the trainer records the raw episode reward, so ``best_reward`` and
      the recorded reward stay in the task's own units.
    * ``clip_predicted_values=False``: this task's reward scale defeats the trainer's
      default value-loss clip the same way; disabling it is the closer match to
      upstream's own optimizer, and keeps the value loss usably small.
    """
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    return {
        "learning_rate": 1e-3,
        # Two different knobs share the name ``kl_threshold``. The scheduler's is a
        # TARGET: the adaptive learning rate rises or falls to keep each update's
        # policy shift near 0.01. The agent's is a CEILING: an update stops taking
        # steps once its shift has already exceeded 0.02. They are deliberately
        # about two to one, so the stop fires only on outliers, not every round.
        "learning_rate_scheduler": functools.partial(KLAdaptiveLR, kl_threshold=0.01),
        "kl_threshold": 0.02,
        "entropy_loss_scale": 0.005,
        "rewards_shaper": lambda rewards, timestep, timesteps: rewards * 0.01,
        "clip_predicted_values": False,
    }


# Retries are safe now that resumability exists: ResumableCheckpoint declares periodic
# checkpoints (every 50 iterations) and resume defaults to "auto" — see cartpole's/
# jetbot's identical comment on their own jobs.
@app.job(
    # Tier 1: T4 GPU, 16 GB VRAM. Run `simulo systems` for the full four-tier catalog.
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train(num_envs: int = 256, max_iterations: int = 500) -> dict[str, Any]:
    """Train the drift policy with PPO, then save the checkpoint AND export a JIT policy.

    Args:
        num_envs: Number of parallel environments to simulate. 256 matches upstream's
            own default for this task.
        max_iterations: Number of PPO policy-update iterations. Reward on this task can
            climb for a while and then diverge late in a run — see ``README.md``'s "What
            to expect" for measured numbers; the KL guard in ``_ppo_overrides`` cuts how
            often that happens without eliminating it. 500 is kept as the default
            anyway, because this job exports the tracked best-so-far checkpoint rather
            than the trainer's final state (see below), so a divergence never reaches
            the returned artifacts, though an early one can still export a weaker
            policy — ``rollout``'s recording is the way to check what a given run
            actually produced, not ``best_reward`` in the result.

    Returns:
        A JSON-serialisable dict: the saved ``checkpoint`` path, the exported ``policy``
        path (verified finite before being returned — see below), whether
        ``used_best_checkpoint`` (false only if no episode ever finished during
        training, in which case both files are the trainer's final state), training
        ``stats``, and the catalog reference the run trained against.
    """
    env = _make_env(num_envs)
    trainer = simulo.RLTrainer(
        env=env,
        algorithm="PPO",
        device="cuda",
        seed=42,
        # discount_factor (0.99), lambda (0.95), and ratio_clip (0.2) already match
        # upstream's requested gamma/lam/clip_param at this trainer's own defaults. The
        # actor/critic network shape (upstream requests [64, 64] hidden dims) is NOT
        # reachable here — the policy/value networks are a fixed architecture, not an
        # agent_cfg knob; see the module docstring's network-architecture gap.
        agent_cfg=_ppo_overrides(),
    )

    stats = trainer.train(max_iterations=max_iterations)

    # Load the trainer's own tracked BEST-so-far checkpoint before saving and
    # exporting, not its live (i.e. final) state. This task's training can climb
    # steadily for a while and then diverge late in a run (see README.md's "What
    # to expect") — without this load-back, a late divergence would
    # reach every artifact this job hands back, including the exported policy
    # ``rollout`` plays.
    checkpoint_dir = stats.get("checkpoint_dir")
    best_source = os.path.join(checkpoint_dir, "best.pt") if checkpoint_dir else None
    # Absent whenever no episode ever finished during training — a normal state, not
    # an error, so this stays best-effort rather than asserting.
    if best_source and os.path.isfile(best_source):
        trainer.load(best_source)

    checkpoint = f"{checkpoints.path}/{_CHECKPOINT_FILE}"
    policy_path = f"{checkpoints.path}/{_POLICY_FILE}"
    trainer.save(checkpoint)
    trainer.export_policy(policy_path)

    # Close the trainer before the environment so the RL library releases its
    # resources first. Deliberately BEFORE the finite-check below: that check only
    # reads the file ``export_policy`` already wrote, needs neither the live trainer
    # nor environment, and raising should never skip releasing resources this job
    # already owns.
    trainer.close()
    env.close()

    # Verify the exported policy is actually usable before returning: a non-finite
    # weight would otherwise reach ``rollout`` silently and fail there instead of
    # here, in the job that can still retry cheaply. ``torch.jit.load`` with no
    # ``map_location`` restores the module onto the device it was saved on (this job
    # always exports on "cuda", never CPU), so the probe tensor has to be built on
    # that same device — otherwise this check fails on a device mismatch instead of
    # the finiteness it exists to test.
    probe = torch.jit.load(policy_path)
    probe_action = probe(torch.zeros(1, F1TenthDriftTask.observation_dim, device="cuda"))
    if not bool(torch.isfinite(probe_action).all()):
        raise RuntimeError(
            f"Exported policy at {policy_path} produced a non-finite action "
            f"({probe_action.tolist()}) on an all-zero probe observation — refusing "
            "to hand back a broken checkpoint. This can happen even after loading "
            "the best-so-far checkpoint if no episode ever finished; rerun with "
            "more iterations or a different seed."
        )

    return {
        "checkpoint": checkpoint,
        "policy": policy_path,
        "used_best_checkpoint": bool(best_source and os.path.isfile(best_source)),
        "num_envs": num_envs,
        "robot_asset": "robot/f1tenth:v1",
        **stats,
    }


# Tier 1: T4, 16 GB VRAM. See `simulo systems`.
@app.job(system=simulo.SystemType.TIER_1, timeout=1 * 60 * 60)
def rollout(num_steps: int = 300) -> dict[str, Any]:
    """Play the exported policy with no trainer, and record the rollout to MCAP.

    ``simulo.RLPlayer`` detects the TorchScript policy ``train`` exported (via
    ``RLTrainer.export_policy``) and drives it directly. ``record=simulo.RecordConfig(...)``
    captures the rollout to an MCAP flight recording in the report volume: policy
    observations and actions, applied actions, rewards, terminations, episode
    boundaries, and robot commands. Open it in Foxglove or Lichtblick, or read it with
    the ``mcap`` library. With ``include_video=True`` and the overhead camera the task
    adds for this job, the recording also carries a playable h264 video of the whole
    loop on ``/sensors/camera/overhead_cam/video_foxglove``; add an Image panel on that
    topic to watch the car drift. ``num_steps=300`` is one full 5 s episode at this
    sample's 60 Hz action rate.
    """
    import json

    policy_path = f"{checkpoints.path}/{_POLICY_FILE}"
    mcap_path = f"{reports.path}/rollout.mcap"

    env = _make_env(num_envs=1, camera=True)
    # A TorchScript checkpoint makes the player run the policy directly, with no trainer.
    player = simulo.RLPlayer(env=env, checkpoint=policy_path, device="cuda")
    record = simulo.RecordConfig(
        output_path=mcap_path,
        policy_checkpoint=policy_path,
        robot_model="f1tenth",
        include_video=True,
        video_fps=30,
        video_bitrate="4M",  # plenty for a 640x480 overhead clip
    )
    stats = player.play(num_steps=num_steps, record=record)

    player.close()
    env.close()

    summary = {"policy": policy_path, "mcap": mcap_path, **stats}

    summary_path = f"{reports.path}/rollout_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[byo-f1tenth-drift] Wrote rollout summary to {summary_path}")

    return {"summary": summary_path, **summary}
