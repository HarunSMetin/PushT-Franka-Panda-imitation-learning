"""End-to-end pipeline test without a human. Run it before every collection campaign.

    .venv/bin/python scripts/smoke_test.py

1. controller: sweep the tip over the workspace; z drift, orientation error, joint limits
2. record scripted demos through the SAME recorder as teleop (-> data/raw_smoke)
3. re-render determinism: restoring a recorded state reproduces the block pose exactly
4. re-simulation determinism: replaying the recorded setpoints from step 0 reproduces
   the whole trajectory
5. export to LeRobot with two different camera setups (no recollection) + validation
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pusht.config import PROJECT_ROOT, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import EpisodeRecorder, list_episodes, load_episode  # noqa: E402
from pusht.scripted import symmetric_push_start  # noqa: E402

RAW = resolve("data/raw_smoke")
PY = sys.executable
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_controller():
    env = PushTEnv()
    env.reset(seed=0)
    env.set_block_pose([0.95, 0.45, 0.0])  # park the block off the workspace
    mujoco.mj_forward(env.model, env.data)
    lo, hi = env.ctrl.ws_lo, env.ctrl.ws_hi
    pts = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]], (lo + hi) / 2, [lo[0], 0], [hi[0], 0]]
    z, rot, near = [], [], 0
    for p in pts:
        for _ in range(150):
            env.step(p)
            q = env.data.qpos[env.ctrl.qadr]
            z.append(env.tip_xyz()[2] - env.ctrl.tip_z)
            rot.append(env.ctrl.tip_error(q)[1])
            near += np.any((q - env.ctrl.qlo < 0.02) | (env.ctrl.qhi - q < 0.02))
        reach = np.linalg.norm(env.tip_xyz()[:2] - np.asarray(p))
        check(f"controller reaches {np.round(p, 3)}", reach < 2e-3, f"err={reach * 1e3:.2f} mm")
    zmax = np.max(np.abs(z)) * 1e3
    check("controller z drift < 2 mm", zmax < 2.0, f"max |dz|={zmax:.2f} mm")
    rmax = np.degrees(np.max(rot))
    check("controller orientation error < 2 deg", rmax < 2.0, f"max={rmax:.2f} deg")
    check("controller never near joint limits", near == 0, f"{near} steps near limits")


def record_scripted(n: int = 3):
    if RAW.exists():
        shutil.rmtree(RAW)
    env = PushTEnv()
    rec = EpisodeRecorder(env, RAW)
    rng = np.random.default_rng(123)
    for i in range(n):
        block, tip, final_tip = symmetric_push_start(env, rng)
        env.reset(block_pose=block, tip_xy=tip)
        rec.start(None, block, tip, operator="scripted")
        success, tilt = False, 0.0
        # Push along the symmetry axis, slowing down in proportion to the remaining
        # block-to-target distance (like a human operator), then hold still.
        u = (final_tip - tip) / np.linalg.norm(final_tip - tip)
        for _ in range(int(20 * env.control_hz)):
            state, obs = env.get_state(), env.observe()
            remaining = float((obs["block_pose"][:2] - env.target_pose[:2]) @ -u)
            step = 0.0 if obs["coverage"] >= env.success_cov else np.clip(0.3 * remaining, 0.0005, 0.005)
            setpoint = env.ctrl.cmd_xy + step * u
            nxt = env.step(setpoint)
            rec.record(state, obs, setpoint, True)
            tilt = max(tilt, env.block_tilt())
            if nxt["success"]:
                success = True
                break
        path = rec.save(success=success, note="smoke test")
        check(f"scripted episode {i} succeeds", success,
              f"{path.name}: {len(load_episode(path)['state'])} steps, "
              f"final coverage {env.observe()['coverage']:.3f}, max tilt {np.degrees(tilt):.2f} deg")


def test_determinism():
    for path in list_episodes(RAW):
        ep = load_episode(path)
        env = PushTEnv(xml=ep["physics_xml"])
        err = 0.0
        for t in range(0, len(ep["state"]), 7):
            env.set_state(ep["state"][t])
            err = max(err, np.max(np.abs(env.block_pose() - ep["block_pose"][t])))
        check(f"re-render {path.stem}", err < 1e-9, f"max block pose err {err:.1e}")

        env.set_state(ep["state"][0], cmd_xy=ep["attrs"]["init_cmd_xy"])
        err = 0.0
        for t in range(len(ep["state"]) - 1):
            env.step(ep["setpoint_xy"][t])
            err = max(err, np.max(np.abs(env.get_state() - ep["state"][t + 1])))
        env.step(ep["setpoint_xy"][-1])
        err = max(err, np.max(np.abs(env.get_state() - ep["final_state"])))
        check(f"re-simulate {path.stem}", err < 1e-9, f"max state err {err:.1e}")


def run(cmd: list[str], name: str) -> bool:
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    ok = r.returncode == 0
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    check(name, ok, " | ".join(tail))
    return ok


def test_export():
    s = str(PROJECT_ROOT / "scripts")
    if run([PY, f"{s}/04_export_lerobot.py", "--raw-dir", str(RAW), "--out", "data/lerobot/smoke_a",
            "--cameras", "overhead", "corner", "--res", "96", "96", "--overwrite"], "export A (overhead+corner 96x96)"):
        run([PY, f"{s}/05_validate_dataset.py", "--root", "data/lerobot/smoke_a", "--raw-dir", str(RAW)],
            "validate A")
    if run([PY, f"{s}/04_export_lerobot.py", "--raw-dir", str(RAW), "--out", "data/lerobot/smoke_b",
            "--cameras", "side", "wrist", "--res", "120", "160", "--fps", "25", "--light-randomize",
            "--overwrite"], "export B (side+wrist 120x160 @25fps, same raw data)"):
        run([PY, f"{s}/05_validate_dataset.py", "--root", "data/lerobot/smoke_b", "--raw-dir", str(RAW)],
            "validate B")


if __name__ == "__main__":
    test_controller()
    record_scripted()
    test_determinism()
    test_export()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)
