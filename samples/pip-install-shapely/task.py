"""Install a PyPI dependency: the task that ``train.py`` trains, with a Shapely reward.

The other training samples in this repository compute their rewards with plain tensor
math. The base runtime includes ``torch``, ``numpy``, and the scientific Python stack,
but not every third-party package a task may need. ``Runtime.pip_install()`` declares
a public PyPI package on the app's ``Runtime`` for Simulo to install before the job
runs. This sample makes
the reward genuinely depend on a third-party library, `shapely
<https://shapely.readthedocs.io/>`_, rather than importing it and never calling it.

It also uses the runtime's other extra, ``Runtime.env()``. The target zone's centre X
is a real task parameter, ``DEMO_ZONE_CENTER_X``, set once on the same ``Runtime``
(``.env({"DEMO_ZONE_CENTER_X": "2.5"})``) and read back from ``os.environ`` in the
``train`` job. The job logs the value it read, so the job log itself shows that
``env()`` reached the job.

The task
--------
The robot is the same catalog JetBot the ``jetbot`` sample trains, on the same ground
plane, but the goal is different: instead of following a commanded heading it drives
toward a target zone, a real ``shapely.geometry.Polygon``, and the reward is computed
with shapely geometry (``Polygon.distance(Point)`` and ``Polygon.contains(Point)``)
rather than a hand-written distance.

* Observation (4 values): the robot's forward direction as a unit vector in the XY
  plane (2) plus the vector from the robot to the zone's centroid (2, recomputed every
  step).
* Action (2 values): left and right wheel angular velocity, scaled by
  ``velocity_scale``. The same control surface as ``jetbot``.
* Reward: each environment's XY position, relative to its own spawn point, becomes a
  shapely ``Point`` every step. The reward is ``-zone.distance(point)`` (zero once
  inside the zone, negative and shrinking in magnitude as the robot approaches) plus a
  flat ``reach_bonus`` once ``zone.contains(point)`` is true. That is why
  ``_zone_distance_reward`` is a plain Python loop rather than a
  ``@app.runtime.torch_jit`` kernel: TorchScript cannot compile a call into a
  third-party CPU library.
* Termination: none, as in ``jetbot``. Every episode truncates at the time limit, and
  reaching the zone keeps paying the bonus rather than ending the episode.

Zone geometry
-------------
With the default centre X of 2.5, the one-metre square spans X=2.0 to 3.0 and Y=-0.5
to 0.5 in each environment's local frame. Reset jitter is at most 0.3 m per axis.
Move the centre or resize the square and the observation, the reward, and the zone
drawn in the scene all follow it.

Spawn jitter: ``reset_idx`` adds a small random XY offset (up to ``spawn_jitter_xy``)
and heading offset (up to ``spawn_jitter_yaw``) to each reset. Without it every
environment would reset to the same pose and drive the same trajectory toward the same
fixed zone. The ``jetbot`` sample gets that diversity for free from its randomised
command; here the zone is fixed, so the diversity has to come from the spawn.

The defaults are starting points, not a measured convergence calibration.

Files
-----
This file holds the application, its runtime, and the task. ``train.py`` holds the one
job you submit, and imports ``app`` and ``ShapelyZoneTask`` from here. This file
declares no job, so ``simulo run`` refuses it. Submit ``train.py`` instead.
"""

from __future__ import annotations

from typing import Tuple

import simulo

# The JetBot robot: the same version-pinned catalog asset the ``jetbot`` sample trains.
jetbot = simulo.Asset.from_registry("simulo/robot/jetbot:v2")

# The point of this app: layer BOTH Runtime extras onto the base Simulo runtime;
# see https://docs.simulo.ai/concepts/runtimes/. pip_install("shapely") adds the third-party
# geometry library the reward genuinely depends on; env() sets DEMO_ZONE_CENTER_X,
# a real task parameter the `train` job reads back from os.environ. Use a dedicated
# Runtime.from_registry(...) instance, NOT App("name").runtime: every App that omits
# runtime= shares the same default Runtime, and pip_install() and env() mutate
# whatever Runtime they are called on.
runtime = (
    simulo.Runtime.from_registry("simulo/gpu-rl:2026.06")
    .pip_install("shapely")
    .env({"DEMO_ZONE_CENTER_X": "2.5"})
)
app = simulo.App("pip-install-shapely", runtime=runtime)

# The heavy AND the third-party imports, both deferred: on your machine this block
# records them instead of resolving them; in the cloud they are plain imports.
# ``torch`` is already in the base runtime; ``shapely`` is there only because the
# ``pip_install("shapely")`` declaration above had it installed.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)
    from shapely.geometry import Point as ShapelyPoint  # noqa: F401
    from shapely.geometry import Polygon  # noqa: F401


# The target zone's corners, in each env's own spawn-relative XY frame: plain
# floats, no shapely needed to declare them. The real ``shapely.geometry.Polygon``
# is only constructed in ``on_start`` below, inside the running job. ``center_x``
# comes from ``DEMO_ZONE_CENTER_X`` (set via this app's ``Runtime.env()`` above,
# read in ``train.py`` and passed to ``ShapelyZoneTask`` below); the
# zone stays a fixed 1.0 m by 1.0 m square centred on that X, directly ahead of
# the robot's default spawn heading (+X, see Pose.identity() below) so a
# converged policy mostly just has to drive forward. The default,
# ``center_x=2.5``, is the value the module docstring's reachability numbers
# (the nearest corner about 2.06 m away) are measured against.
def _target_zone_vertices(center_x: float) -> list[tuple[float, float]]:
    """The target zone's four corners, centered on ``center_x`` (spawn-relative XY)."""
    return [
        (center_x - 0.5, -0.5),
        (center_x + 0.5, -0.5),
        (center_x + 0.5, 0.5),
        (center_x - 0.5, 0.5),
    ]


def _quat_to_forward(quat: torch.Tensor) -> torch.Tensor:
    """Rotate the unit X vector ``[1, 0, 0]`` by a per-env quaternion (wxyz).

    The same helper as the ``jetbot`` sample's. Each sample carries its own copy rather
    than importing across samples, so every sample stays self-contained.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + w * z)
    forward_z = 2.0 * (x * z - w * y)
    return torch.stack([forward_x, forward_y, forward_z], dim=-1)


def _zone_distance_reward(
    local_xy: torch.Tensor,
    zone: Polygon,
    reach_bonus: float,
    device: str,
) -> torch.Tensor:
    """The shapely-computed reward: ``-distance to zone`` plus a reach bonus.

    Deliberately NOT ``@app.runtime.torch_jit`` (unlike the ``jetbot`` sample's
    ``_compute_rewards``): shapely geometry objects are plain CPU Python objects, not
    GPU tensors, and TorchScript cannot compile a call into a third-party library. So
    this drops to the CPU once per step: a plain Python loop over ``num_envs`` XY
    positions, each becoming a ``shapely.geometry.Point``.

    ``Polygon.distance(Point)`` is the real Euclidean distance from the point to the
    polygon's boundary, and it is exactly ``0.0`` once the point is inside, never
    negative, so ``-distance`` alone plateaus at its maximum (0) anywhere inside the
    zone. ``Polygon.contains(Point)`` adds ``reach_bonus`` on top once the robot has
    actually arrived, so the reward keeps climbing after the distance term saturates.
    """
    positions = local_xy.detach().cpu().numpy()
    rewards = []
    for x, y in positions:
        point = ShapelyPoint(float(x), float(y))
        distance = zone.distance(point)
        bonus = reach_bonus if zone.contains(point) else 0.0
        rewards.append(bonus - distance)
    # ``.view(-1)`` locks in the per-env reward contract explicitly, (num_envs,)
    # even when num_envs == 1, the same belt and braces the ``jetbot`` sample
    # applies to its own JIT-compiled reward kernel.
    return torch.tensor(rewards, device=device, dtype=local_xy.dtype).view(-1)


class ShapelyZoneTask(simulo.Task):
    """Drive a JetBot into a target zone that is defined, and checked, with shapely.

    Observation (4-dim): forward direction unit vector, XY plane only (2), plus the
    vector from the robot to the target zone's centroid (2). Action (2-dim): left and
    right wheel angular velocity.

    ``zone_center_x`` is the target-zone centre read from the environment (the
    ``train`` job in ``train.py`` reads ``DEMO_ZONE_CENTER_X`` from ``os.environ`` and
    passes it here). It defaults to ``2.5`` to match this task's own reachability
    numbers when unset.
    """

    observation_dim = 4
    action_dim = 2

    def __init__(self, zone_center_x: float = 2.5):
        super().__init__()
        self._zone_center_x = zone_center_x

    # See "Zone geometry" in the module docstring. These set episode duration and the
    # wheel-velocity scale.
    episode_length_s = 10.0
    velocity_scale = 15.0  # [rad/s] wheel angular velocity scale
    reach_bonus = 2.0  # flat reward bonus once inside the target zone

    spawn_jitter_xy = 0.3  # [m] max per-axis random spawn position offset
    spawn_jitter_yaw = 0.3  # [rad] max random spawn heading offset (~17 deg)

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings.
    device: str
    num_envs: int
    max_episode_length: int
    episode_length_buf: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/World", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/World",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=jetbot, initial_pose=simulo.Pose.identity())
        scene.add(self.robot, at="/World/Robot")

    def on_start(self, env: simulo.LearningEnv) -> None:
        left = self.robot.find_joints("left_wheel_joint")
        right = self.robot.find_joints("right_wheel_joint")
        self._wheel_joint_ids = left + right

        # Per-env world-space grid offset, so the target zone below reads the
        # same regardless of where an env's patch sits on the replication grid.
        origins = env.scene.env_origins
        if origins is not None:
            self._env_origin_xy = origins[:, :2].clone()
        else:
            self._env_origin_xy = torch.zeros((self.num_envs, 2), device=self.device)

        # The real ``shapely.geometry.Polygon``, constructed only now. ``on_start``
        # runs only inside the job (nothing calls task methods at submit time), so
        # by the time this line runs the module-level ``with app.runtime.imports():``
        # block has done a REAL ``from shapely.geometry import ...`` and ``Polygon``
        # is the genuine shapely class. ``self._zone_center_x`` is the value
        # ``train`` read from ``DEMO_ZONE_CENTER_X`` (set via ``Runtime.env()``).
        self._target_zone = Polygon(_target_zone_vertices(self._zone_center_x))
        centroid = self._target_zone.centroid
        self._target_centroid_xy = torch.tensor([centroid.x, centroid.y], device=self.device)

    def _local_xy(self) -> torch.Tensor:
        """The robot's XY position relative to its own env's spawn point."""
        # robot.state.pose is [x, y, z, qw, qx, qy, qz]; XY is the first two columns.
        return self.robot.state.pose[:, :2] - self._env_origin_xy

    def get_observations(self) -> torch.Tensor:
        # robot.state is the supported, typed way to read live state. robot.internals
        # is the raw escape hatch; see
        # https://docs.simulo.ai/concepts/scene-robot-world/.
        forward_xy = _quat_to_forward(self.robot.state.pose[:, 3:7])[:, :2]
        target_xy = self._target_centroid_xy.unsqueeze(0) - self._local_xy()
        return torch.cat([forward_xy, target_xy], dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _zone_distance_reward(
            self._local_xy(), self._target_zone, self.reach_bonus, self.device
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        scaled_velocities = actions * self.velocity_scale
        self.robot.set_joint_velocity_target(scaled_velocities, joint_ids=self._wheel_joint_ids)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.robot.reset(env_ids)
        self._apply_spawn_jitter(env_ids)

    def _apply_spawn_jitter(self, env_ids: torch.Tensor) -> None:
        """Nudge each reset env's XY position and heading by a small random amount.

        Without this every env resets to the exact same local spawn pose (only the
        fixed per-env grid offset differs), so every env would drive the identical
        trajectory toward the same fixed-offset zone; the ``jetbot`` sample gets its
        per-env diversity for free from its randomised commanded direction. Built from
        two sources that never depend on reading back a value written moments ago:
        ``self._env_origin_xy`` (captured once in ``on_start``, the stable per-env grid
        offset) for XY, and the asset's own ``default_root_state`` (a static
        configuration tensor, the same one ``robot.reset()`` reads) for the authored
        rest height. Environment grids only offset X and Y, never Z, so the local
        default Z is already the world Z. Applied with ``set_root_pose`` (world frame,
        a teleport rather than a control input).
        """
        n = len(env_ids)
        jitter_xy = (torch.rand((n, 2), device=self.device) * 2.0 - 1.0) * self.spawn_jitter_xy
        jitter_yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.spawn_jitter_yaw

        pose = torch.zeros((n, 7), device=self.device)
        pose[:, :2] = self._env_origin_xy[env_ids] + jitter_xy
        pose[:, 2] = self.robot.internals.default_root_state[
            env_ids, 2
        ]  # the asset's authored rest height
        half_yaw = jitter_yaw * 0.5
        pose[:, 3] = torch.cos(half_yaw)  # qw
        pose[:, 6] = torch.sin(half_yaw)  # qz: pure yaw rotation about world Z
        self.robot.set_root_pose(pose, env_ids=env_ids)
