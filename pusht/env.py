"""Push-T simulation environment (MuJoCo, Panda + stick, planar pushing).

One `step()` = one control period (default 50 Hz): the commanded tip xy is updated
(clamped + rate-limited), IK gives joint targets, physics runs for the substeps.
"""

from __future__ import annotations

import mujoco
import numpy as np
from shapely.geometry import Point, box

from .config import load_config
from .controller import PlanarController
from .geometry import TShape, coverage, quat_from_yaw, yaw_from_quat
from .scene import BLOCK_JOINT, TIP_SITE, model_from_xml, physics_xml

# Full integration state: exact re-simulation from any recorded step.
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class PushTEnv:
    def __init__(self, scene_cfg: dict | None = None, collect_cfg: dict | None = None,
                 cameras: dict[str, dict] | None = None, xml: str | None = None,
                 model: mujoco.MjModel | None = None):
        """`model`: reuse an already compiled model (many envs, one renderer); must match `xml`."""
        self.scene_cfg = scene_cfg or load_config("scene")
        self.collect_cfg = collect_cfg or load_config("collect")
        self.xml = xml or physics_xml(self.scene_cfg)
        self.model = model if model is not None else model_from_xml(self.xml, cameras)
        self.data = mujoco.MjData(self.model)

        self.control_hz = float(self.collect_cfg["control_hz"])
        self.control_dt = 1.0 / self.control_hz
        self.n_substeps = int(round(self.control_dt / self.model.opt.timestep))
        self.ctrl = PlanarController(self.model, self.scene_cfg,
                                     float(self.collect_cfg["max_tip_speed"]), self.control_dt)

        self.shape = TShape.from_config(self.scene_cfg["t_block"])
        tg = self.scene_cfg["target"]
        self.target_pose = np.array([tg["x"], tg["y"], tg["yaw"]], dtype=float)
        self.success_cov = float(self.collect_cfg["success_coverage"])
        self.hold_steps = max(1, int(round(float(self.collect_cfg["success_hold_time"]) * self.control_hz)))

        jnt = self.model.joint(BLOCK_JOINT)
        self.block_qadr = jnt.qposadr[0]
        self.block_dadr = jnt.dofadr[0]
        self.tip_id = self.model.site(TIP_SITE).id
        self.state_size = mujoco.mj_stateSize(self.model, STATE_SPEC)
        self._above = 0

    # ------------------------------------------------------------------ reset
    def sample_initial(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Random block pose (x, y, yaw) and tip start xy, per configs/scene.yaml `init`."""
        ws = self.scene_cfg["workspace"]
        ini = self.scene_cfg["init"]
        m = float(ini["margin"])
        inner = box(ws["x"][0] + m, ws["y"][0] + m, ws["x"][1] - m, ws["y"][1] - m)
        for _ in range(10000):
            pose = np.array([rng.uniform(*ws["x"]), rng.uniform(*ws["y"]), rng.uniform(-np.pi, np.pi)])
            poly = self.shape.polygon(*pose)
            if not inner.contains(poly):
                continue
            if coverage(self.shape, pose, self.target_pose) > float(ini["max_initial_coverage"]):
                continue
            for _ in range(1000):
                tip = np.array([rng.uniform(*ws["x"]), rng.uniform(*ws["y"])])
                if poly.distance(Point(tip)) > float(ini["tip_clearance"]):
                    return pose, tip
        raise RuntimeError("Could not sample an initial state; check workspace/init config.")

    def reset(self, seed: int | None = None, block_pose=None, tip_xy=None) -> dict:
        """Reset to a random (seeded) or explicitly given initial state and let it settle."""
        rng = np.random.default_rng(seed)
        if block_pose is None or tip_xy is None:
            block_pose, tip_xy = self.sample_initial(rng)
        mujoco.mj_resetData(self.model, self.data)
        q = self.ctrl.reset(tip_xy)
        self.data.qpos[self.ctrl.qadr] = q
        self.data.ctrl[:] = q
        self.set_block_pose(block_pose)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(float(self.scene_cfg["init"]["settle_time"]) / self.control_dt)):
            self.step(self.ctrl.cmd_xy)
        self.data.time = 0.0
        self._above = 0
        return self.observe()

    def set_block_pose(self, pose) -> None:
        a = self.block_qadr
        self.data.qpos[a:a + 3] = [pose[0], pose[1], self.shape.height / 2]
        self.data.qpos[a + 3:a + 7] = quat_from_yaw(pose[2])
        self.data.qvel[self.block_dadr:self.block_dadr + 6] = 0.0

    # ------------------------------------------------------------------ step
    def step(self, setpoint_xy) -> dict:
        """Advance one control period towards `setpoint_xy`. Returns the observation."""
        self.ctrl.command(setpoint_xy)
        self.data.ctrl[:] = self.ctrl.q_des
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        obs = self.observe()
        self._above = self._above + 1 if obs["coverage"] >= self.success_cov else 0
        obs["success"] = self._above >= self.hold_steps
        return obs

    # ------------------------------------------------------------------ observation
    def block_pose(self) -> np.ndarray:
        a = self.block_qadr
        q = self.data.qpos[a + 3:a + 7]
        return np.array([self.data.qpos[a], self.data.qpos[a + 1], yaw_from_quat(q)])

    def block_tilt(self) -> float:
        """Angle [rad] between block up-axis and world z (should stay ~0)."""
        a = self.block_qadr
        w, x, y, z = self.data.qpos[a + 3:a + 7]
        return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))

    def tip_xyz(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_id].copy()

    def observe(self) -> dict:
        bp = self.block_pose()
        return {
            "tip_xy": self.tip_xyz()[:2],
            "cmd_xy": self.ctrl.cmd_xy.copy(),
            "block_pose": bp,
            "coverage": coverage(self.shape, bp, self.target_pose),
            "success": self._above >= self.hold_steps,
        }

    # ------------------------------------------------------------------ full state I/O
    def get_state(self) -> np.ndarray:
        s = np.empty(self.state_size)
        mujoco.mj_getState(self.model, self.data, s, STATE_SPEC)
        return s

    def set_state(self, state: np.ndarray, cmd_xy=None) -> None:
        """Restore a recorded state (and the controller command, for re-simulation)."""
        mujoco.mj_setState(self.model, self.data, np.asarray(state, float), STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)
        self.ctrl.q_des = self.data.ctrl.copy()
        if cmd_xy is not None:
            self.ctrl.cmd_xy = np.asarray(cmd_xy, float).copy()
