"""Cartpole Eval: train, evaluate, and record a policy with three jobs in one app.

The ``cartpole`` sample trains a policy and stops. This sample carries a policy
through its whole life in one application, with three jobs that share named volumes:

1. ``train``: a short PPO run, then ``RLTrainer.save`` writes the trainer's checkpoint
   and ``RLTrainer.export_policy`` writes a standalone TorchScript policy.
2. ``evaluate``: ``RLTrainer.evaluate(checkpoint=...)`` replays the checkpoint for
   several rounds and ``numpy`` reduces the rounds to percentiles and a success rate.
3. ``rollout``: ``simulo.RLPlayer`` plays the exported policy with no trainer involved
   and records every step, plus a side-view video, to an MCAP flight recording.

Run the jobs in that order. ``evaluate`` and ``rollout`` read what ``train`` wrote, and
each fails with a clear "checkpoint not found" error if the volume is still empty.

What this sample shows beyond ``cartpole``
------------------------------------------
* Several ``@app.job`` functions in one app, chosen with ``--job NAME``, sharing two
  volumes: one for checkpoints and one for reports and the recording.
* The Simulo runtime ships the scientific Python stack. ``evaluate`` imports ``numpy``
  inside its body without declaring or installing anything.
* Two kinds of saved policy. ``RLTrainer.save`` writes the trainer's own checkpoint,
  which ``RLTrainer.evaluate`` reloads. ``RLTrainer.export_policy`` writes the
  deterministic policy as TorchScript, which ``simulo.RLPlayer`` runs without a
  trainer. The export refuses to write a policy whose inputs or outputs would be
  silently transformed, so what ``rollout`` plays is exactly what was trained.
* Recording. ``player.play(record=simulo.RecordConfig(...))`` captures observations,
  actions, rewards, terminations, and episode boundaries to MCAP. With
  ``include_video=True`` and the camera the task attaches for ``rollout``, the file
  also carries a playable h264 clip on ``/sensors/camera/side_cam/video_foxglove``.
* Scene extras. ``build`` adds a decorative cuboid and a sphere marker that
  ``on_post_physics_step`` moves to follow the cart; the marker shows in the video.
* A tunable evaluation bar. ``evaluate`` reads ``SIMULO_EVAL_REWARD_TARGET`` from the
  job's environment, default ``75.0``. Each of five rounds yields one pass/fail value,
  so ``success_rate`` changes in increments of ``0.2``. The default is a starting point
  to tune, not a threshold a trained policy is expected to clear. To move the bar, give
  the app a runtime with ``.env({"SIMULO_EVAL_REWARD_TARGET": "60"})``; see
  https://docs.simulo.ai/concepts/runtimes/.

Run it
------
Sign in once with ``simulo login``, then run each stage after the previous one has
completed::

    simulo run samples/cartpole-eval/app.py --job train --num-envs 512 --max-iterations 20
    simulo run samples/cartpole-eval/app.py --job evaluate --num-episodes 10 --num-rounds 5
    simulo run samples/cartpole-eval/app.py --job rollout --num-steps 200

``simulo result`` prints each stage's numbers, and after ``rollout`` finishes
``simulo recordings`` downloads the MCAP file, which opens in Foxglove or Lichtblick.
"""

from __future__ import annotations

import math
import os
from typing import Any, Tuple

import simulo

# ``evaluate`` scores each round against a mean-reward target, read from the
# SIMULO_EVAL_REWARD_TARGET environment variable with a default of 75.0 in the job
# body. Each round yields one pass/fail value, so five rounds produce a rate in 0.2
# increments. The default is a starting point to tune, not a threshold a trained
# checkpoint is expected to clear. Set the target through a runtime ``.env({...})``
# layer on the app; see the module docstring.

# The cartpole robot: a version-pinned catalog reference.
cartpole = simulo.Asset.from_registry("simulo/robot/cartpole:v1")

# Two named, durable volumes: one for the trained checkpoints (the trainer checkpoint
# and the exported TorchScript policy), one for the JSON reports and the recording.
# Each one's real directory is read through ``.path``, which resolves only inside a
# running job.
checkpoints = simulo.Volume.from_name("cartpole-eval-checkpoints", create_if_missing=True)
reports = simulo.Volume.from_name("cartpole-eval-reports", create_if_missing=True)

# Advanced: pick a different Simulo runtime with
# App("name", runtime=simulo.Runtime.from_registry("simulo/gpu-rl:2026.06"));
# see https://docs.simulo.ai/concepts/runtimes/.
app = simulo.App("cartpole-eval", mounts={"/checkpoints": checkpoints, "/reports": reports})

# The one heavy import, deferred: on your machine this block records the import
# instead of resolving it; in the cloud it is a plain import.
with app.runtime.imports():
    import torch  # noqa: F401  (resolved only when the job runs in the cloud)

# Stable filenames inside the checkpoint volume, shared across the three jobs.
_CHECKPOINT_FILE = "cartpole_eval_final.pt"
_POLICY_FILE = "cartpole_eval_policy.pt"


@app.runtime.torch_jit
def _compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled balance reward kernel (the classic cartpole balance reward).

    ``@app.runtime.torch_jit`` is a marker on your machine and ``torch.jit.script`` in
    the cloud, so this can live at module level and its body never runs at submit.
    """
    pole_pos = pole_pos.squeeze()
    pole_vel = pole_vel.squeeze()
    cart_pos = cart_pos.squeeze()
    cart_vel = cart_vel.squeeze()
    reset_terminated = reset_terminated.squeeze()

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.square(pole_pos)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)

    reward: torch.Tensor = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    # Keep the (num_envs,) per-env reward contract even when num_envs == 1: the
    # squeezes above collapse a single-env batch to a 0-d scalar, which breaks
    # per-env consumers (e.g. the MCAP recorder's per-env /reward indexing).
    return reward.view(-1)


class CartpoleEvalTask(simulo.Task):
    """Balance a pole on a cart: the ``cartpole`` balance task, with scene props.

    Observation (4-dim): pole angle, pole angular velocity, cart position, cart
    velocity. Action (1-dim): scaled horizontal force on the cart. ``build`` additionally
    drops a decorative prop and a debug marker into the scene, and
    ``on_post_physics_step`` moves the marker to follow the cart each step.

    ``with_camera=True`` (the ``rollout`` job only) additionally attaches a static
    side-view camera (``simulo.Camera``) to the cartpole's fixed ``slider`` (rail) link,
    so the MCAP flight recording carries a playable h264 video of the cart sliding and
    the pole balancing. Training and evaluation keep the default ``with_camera=False``:
    no camera, no render cost.
    """

    observation_dim = 4
    action_dim = 1

    def __init__(self, with_camera: bool = False):
        super().__init__()
        self._with_camera = with_camera

    episode_length_s = 5.0
    action_scale = 100.0  # [N]

    max_cart_pos = 3.0  # [m]
    initial_pole_angle_range = (-0.25, 0.25)  # fraction of pi [rad]

    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    # Set by the training base class when the job runs. Declared here only so a
    # type checker sees the names the methods read; the annotations are strings.
    device: str
    max_episode_length: int
    episode_length_buf: torch.Tensor
    reset_terminated: torch.Tensor

    def build(self, scene: simulo.Scene) -> None:
        scene.add(simulo.Terrain.plane(name="ground"), at="/", per_environment=False)
        scene.add(
            simulo.Light.dome(name="light", intensity=2000.0, color=(0.75, 0.75, 0.75)),
            at="/",
            per_environment=False,
        )
        self.robot = simulo.Robot(asset=cartpole, initial_pose=simulo.Pose.identity())

        # Rollout-only: a static side-view camera so the MCAP flight recording
        # carries playable video. ORDER MATTERS: attach the camera to the robot
        # BEFORE ``scene.add(robot, ...)``. The scene collects a robot's sensors
        # once, during that call; a camera attached afterwards is never seen and
        # the recorder reports ``VIDEO_NO_CAMERA_FOUND`` (zero video channels).
        # The camera hangs off the cartpole's fixed ``slider`` (rail) link, 5 m
        # back along -X at rail height + 0.5 m, looking at the rail: the cart
        # slides along +/-Y and the pole swings in the Y-Z plane, so this view
        # faces both motions head-on. The sensor name is the tail of the
        # ``attach_to`` path (``side_cam``), so the video lands on
        # ``/sensors/camera/side_cam/video`` plus its ``_foxglove`` sibling.
        if self._with_camera:
            # ``simulo.Camera``, ``simulo.SensorOffset``, and ``simulo.CameraSpawnConfig``
            # are part of the same lazily resolved ``simulo.*`` surface as
            # ``simulo.Robot``, so no extra import is needed, and nothing here runs at
            # submit time: ``build()`` runs only in the cloud.
            self.robot.add_sensor(
                simulo.Camera(
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    update_period=1.0 / 30.0,  # 30 Hz, matches RecordConfig.video_fps
                    offset=simulo.SensorOffset.look_at(
                        pos=(-5.0, 0.0, 0.5), target=(0.0, 0.0, 0.5)
                    ),
                    spawn=simulo.CameraSpawnConfig(
                        focal_length=18.0,  # wide enough to keep the cart's ±3 m travel in frame
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 1.0e5),
                    ),
                ),
                attach_to="slider/side_cam",
            )

        scene.add(self.robot, at="/World/Robot")

        # Scene extra 1: a decorative static prop on the ground plane. The cart
        # rides an elevated rail (about 2 m up, as the robot model is built) and
        # travels along world Y, so a half-metre cuboid at ground level never
        # touches the cart or the pole's Y-Z swing plane, and it gives the rollout
        # video a static depth cue in the foreground.
        prop = simulo.Entity.primitive.cuboid(
            name="marker_post",
            size=(0.3, 0.3, 0.5),
            pose=simulo.Pose(position=(0.0, 1.5, 0.25)),
            material=simulo.Material.surface(color=(0.2, 0.6, 0.9)),
        )
        scene.add(prop, at="/World/Prop")

        # Scene extra 2: a debug sphere marker that tracks the cart (moved each step
        # in ``on_post_physics_step``). One shared instance; invisible when nothing
        # renders.
        self._cart_marker = simulo.Visual.sphere(
            name="cart_tracker", radius=0.08, color=(1.0, 0.4, 0.1)
        )
        scene.add(self._cart_marker, at="/World/Markers", per_environment=False)

    def on_start(self, env: simulo.LearningEnv) -> None:
        self._cart_dof_idx = self.robot.find_joints("slider_to_cart")
        self._pole_dof_idx = self.robot.find_joints("cart_to_pole")
        # robot.state is the supported, typed way to read live state. robot.internals
        # is the raw escape hatch; see
        # https://docs.simulo.ai/concepts/scene-robot-world/.
        self._joint_pos = self.robot.state.joint_positions
        self._joint_vel = self.robot.state.joint_velocities
        # Pre-allocated marker pose buffers (shape (1, 3) / (1, 4)) so updating the
        # marker each step never re-allocates or forces a host sync. The cart slides
        # along the rail's world **Y** axis (the robot model turns the sliding
        # joint 90 degrees about Z), so the marker is anchored at the robot root
        # and only its Y component moves each step, floating 0.35 m above the cart.
        root_pose = self.robot.get_root_pose()
        root = root_pose[0, :3] if root_pose is not None else torch.zeros(3, device=self.device)
        self._marker_base_y = root[1].clone()
        self._marker_pos = root.view(1, 3).clone()
        self._marker_pos[0, 2] += 0.35
        self._marker_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)

    def get_observations(self) -> torch.Tensor:
        pole_idx = self._pole_dof_idx[0]
        cart_idx = self._cart_dof_idx[0]
        pole_pos = self._joint_pos[:, pole_idx].view(-1, 1)
        pole_vel = self._joint_vel[:, pole_idx].view(-1, 1)
        cart_pos = self._joint_pos[:, cart_idx].view(-1, 1)
        cart_vel = self._joint_vel[:, cart_idx].view(-1, 1)
        return torch.cat((pole_pos, pole_vel, cart_pos, cart_vel), dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.rew_scale_alive,
            self.rew_scale_terminated,
            self.rew_scale_pole_pos,
            self.rew_scale_cart_vel,
            self.rew_scale_pole_vel,
            self._joint_pos[:, self._pole_dof_idx[0]],
            self._joint_vel[:, self._pole_dof_idx[0]],
            self._joint_pos[:, self._cart_dof_idx[0]],
            self._joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._joint_pos = self.robot.state.joint_positions
        self._joint_vel = self.robot.state.joint_velocities
        pole_idx = self._pole_dof_idx[0]
        cart_idx = self._cart_dof_idx[0]
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        cart_out = torch.abs(self._joint_pos[:, cart_idx]) > self.max_cart_pos
        pole_fallen = torch.abs(self._joint_pos[:, pole_idx]) > math.pi / 2
        terminated = cart_out | pole_fallen
        return terminated, truncated

    def apply_actions(self, actions: torch.Tensor) -> None:
        self.robot.set_joint_effort_target(
            self.action_scale * actions, joint_ids=self._cart_dof_idx
        )

    def on_post_physics_step(self) -> None:
        """Move the debug marker with the (env-0) cart each step.

        The cart's prismatic joint position maps to a world **Y** offset from the
        rail origin (see ``on_start``). Visible in the rollout video; invisible
        when rendering is off.
        """
        cart_idx = self._cart_dof_idx[0]
        self._marker_pos[0, 1] = self._marker_base_y + self.robot.state.joint_positions[0, cart_idx]
        self._cart_marker.set_pose(self._marker_pos, self._marker_quat)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        self.robot.reset(env_ids)
        pole_idx = self._pole_dof_idx[0]
        # robot.state has no default-joint-value equivalent, so this stays on the
        # internals escape hatch (there is nothing unstable about reading it here,
        # just no supported, typed name for it yet).
        joint_pos = self.robot.internals.default_joint_pos[env_ids].clone()
        random_angles = torch.empty(num_resets, device=self.device).uniform_(
            self.initial_pole_angle_range[0] * math.pi,
            self.initial_pole_angle_range[1] * math.pi,
        )
        joint_pos[:, pole_idx] += random_angles
        # set_joint_state writes positions and velocities in one call and refreshes
        # the buffers behind robot.state in place. self._joint_pos and
        # self._joint_vel are those same buffers, so they are already current after
        # this call and nothing needs to be written into them directly.
        joint_vel = self.robot.internals.default_joint_vel[env_ids]
        self.robot.set_joint_state(joint_pos, velocities=joint_vel, env_ids=env_ids)


def _make_env(num_envs: int, *, camera: bool = False) -> Any:
    """Construct the shared cartpole environment (same sim settings across all jobs).

    ``camera=True`` (rollout only) attaches the side-view Camera in ``build`` AND
    passes ``enable_cameras=True``: the simulator refuses to spawn Camera
    sensors unless camera rendering is enabled, and this LearningEnv kwarg is
    what wires that through, including headless offscreen rendering.
    """
    return simulo.LearningEnv(
        task=CartpoleEvalTask(with_camera=camera),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=4.0,
        headless=True,
        seed=42,
        enable_cameras=camera,
    )


@app.job(
    gpu="L4",
    timeout=8 * 60 * 60,
    retries=2,
    callbacks=[simulo.callbacks.ResumableCheckpoint(every=50)],
)
def train(num_envs: int = 512, max_iterations: int = 20) -> dict[str, Any]:
    """Train a short PPO policy, then save the checkpoint AND export a TorchScript policy.

    Saves two files to the checkpoint volume: the trainer's own checkpoint (consumed by
    ``evaluate``) and a deterministic TorchScript policy exported through the public
    ``RLTrainer.export_policy`` API (consumed by ``rollout`` via ``simulo.RLPlayer``).
    """
    env = _make_env(num_envs)
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    checkpoint = f"{checkpoints.path}/{_CHECKPOINT_FILE}"
    policy_path = f"{checkpoints.path}/{_POLICY_FILE}"
    trainer.save(checkpoint)
    trainer.export_policy(policy_path)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"checkpoint": checkpoint, "policy": policy_path, "num_envs": num_envs, **stats}


@app.job(gpu="L4", timeout=2 * 60 * 60)
def evaluate(num_episodes: int = 10, num_rounds: int = 5) -> dict[str, Any]:
    """Evaluate the saved checkpoint over several rounds and aggregate them with numpy.

    Each round runs ``RLTrainer.evaluate`` for ``num_episodes`` episodes and yields one
    mean-reward / mean-length sample; ``numpy`` then reduces the ``num_rounds`` per-round
    samples to percentiles and a success rate (the fraction of rounds whose mean reward
    clears ``SIMULO_EVAL_REWARD_TARGET``, default ``75.0``). Each round contributes one
    pass/fail value, so five rounds produce a rate in 0.2 increments. The default target
    is a starting point to tune. The JSON report is written to the report volume and
    also returned.
    """
    import json

    import numpy as np

    checkpoint = f"{checkpoints.path}/{_CHECKPOINT_FILE}"
    reward_target = float(os.environ.get("SIMULO_EVAL_REWARD_TARGET", "75.0"))

    env = _make_env(num_envs=64)
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    round_rewards: list[float] = []
    round_lengths: list[float] = []
    for _ in range(num_rounds):
        metrics = trainer.evaluate(checkpoint=checkpoint, num_episodes=num_episodes)
        round_rewards.append(float(metrics["mean_reward"]))
        round_lengths.append(float(metrics["mean_length"]))

    trainer.close()
    env.close()

    rewards = np.array(round_rewards, dtype=np.float64)
    lengths = np.array(round_lengths, dtype=np.float64)
    report = {
        "checkpoint": checkpoint,
        "num_rounds": num_rounds,
        "num_episodes": num_episodes,
        "reward_target": reward_target,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "reward_p25": float(np.percentile(rewards, 25)),
        "reward_p50": float(np.percentile(rewards, 50)),
        "reward_p75": float(np.percentile(rewards, 75)),
        "mean_length": float(lengths.mean()),
        "success_rate": float((rewards >= reward_target).mean()),
    }

    report_path = f"{reports.path}/eval_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[cartpole_eval] Wrote eval report to {report_path}")

    return {"report": report_path, **report}


@app.job(gpu="L4", timeout=1 * 60 * 60)
def rollout(num_steps: int = 200) -> dict[str, Any]:
    """Play the exported TorchScript policy with no trainer and record it to MCAP.

    ``simulo.RLPlayer`` detects the TorchScript policy ``train`` exported (via
    ``RLTrainer.export_policy``) and drives it directly. ``record=simulo.RecordConfig(...)``
    captures the rollout to an MCAP flight recording in the report volume: policy
    observations and actions, applied actions, rewards, terminations, episode boundaries,
    and robot commands. Open it in Foxglove or Lichtblick, or read it with the ``mcap``
    library. With ``include_video=True`` and the side-view camera the task attaches for
    this job, the recording also carries a playable h264 video of the rollout on
    ``/sensors/camera/side_cam/video_foxglove``; add an Image panel on that topic to
    watch the cartpole move. The player merges the recorder's statistics
    (``record_path``, ``messages_written``, ``recording_complete``, and so on) into the
    returned play statistics, which are written to the report volume as the JSON
    rollout summary and also returned.
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
        robot_model="cartpole",
        # profile="standard" (the default) records the core eval streams (policy
        # observations/actions, applied actions, rewards, terminations, episode
        # boundaries) for env 0.
        # include_video adds a playable h264 clip from the side-view Camera the
        # task attaches for this job: /sensors/camera/side_cam/video (raw h264)
        # plus its foxglove.CompressedVideo sibling ..._foxglove, which
        # Lichtblick / Foxglove Image panels render directly. video_encoder
        # stays "auto": hardware encoding when the GPU provides it, software
        # encoding otherwise.
        include_video=True,
        video_fps=30,
        video_bitrate="4M",  # plenty for a 640x480 cartpole clip
    )
    stats = player.play(num_steps=num_steps, record=record)

    player.close()
    env.close()

    summary = {"policy": policy_path, "mcap": mcap_path, **stats}

    summary_path = f"{reports.path}/rollout_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[cartpole_eval] Wrote rollout summary to {summary_path}")

    return {"summary": summary_path, **summary}
