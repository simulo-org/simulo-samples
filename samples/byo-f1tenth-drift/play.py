"""Play and record a trained F1TENTH drift policy."""

from __future__ import annotations

from task import _make_env, app, simulo


@app.job(type="play", system=simulo.SystemType.TIER_1, timeout=60 * 60)
def play(policy: simulo.PolicyCheckpoint, num_steps: int = 300) -> None:
    """Play a policy: simulo run play.py --policy policy_x:best."""
    env = _make_env(num_envs=1, camera=True)
    player = simulo.RLPlayer(env=env, policy=policy, device="cuda")
    try:
        player.play(
            num_steps=num_steps,
            record=simulo.RecordConfig(
                include_video=True,
                video_fps=30,
                video_bitrate="4M",
                robot_model="f1tenth",
            ),
        )
    finally:
        player.close()
        env.close()
