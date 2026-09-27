"""Planar Cartesian controller for the stick tip.

Input: a desired tip position (x, y) on the table. The tip height and the stick
orientation (pointing straight down) are fixed, so the robot can only move in the
table plane. The xy command is clamped to the workspace and rate-limited; the
resulting command is what gets recorded as the action.

Joint targets come from damped-least-squares IK on the `tip` site with a nullspace
pull towards the home posture; the Panda's position actuators then track them.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import ARM_JOINTS, TIP_SITE


class PlanarController:
    def __init__(self, model: mujoco.MjModel, scene_cfg: dict, max_speed: float, control_dt: float):
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.site_id = model.site(TIP_SITE).id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.qlo = np.array([model.joint(j).range[0] for j in ARM_JOINTS])
        self.qhi = np.array([model.joint(j).range[1] for j in ARM_JOINTS])

        rob = scene_cfg["robot"]
        ws = scene_cfg["workspace"]
        self.tip_z = float(rob["tip_height"])
        self.q_home = np.array(rob["home_qpos"], dtype=float)
        self.ws_lo = np.array([ws["x"][0], ws["y"][0]], dtype=float)
        self.ws_hi = np.array([ws["x"][1], ws["y"][1]], dtype=float)
        self.max_step = max_speed * control_dt

        # Desired tip orientation = orientation at the home posture (stick pointing down).
        self.ik_data.qpos[self.qadr] = self.q_home
        mujoco.mj_kinematics(model, self.ik_data)
        self.quat_des = np.zeros(4)
        mujoco.mju_mat2Quat(self.quat_des, self.ik_data.site_xmat[self.site_id])

        self.q_des = self.q_home.copy()
        self.cmd_xy = None

    # ------------------------------------------------------------------ IK
    def _tip_error(self, q: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.ik_data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        pos_err = np.array([xy[0], xy[1], self.tip_z]) - d.site_xpos[self.site_id]
        q_cur = np.zeros(4)
        mujoco.mju_mat2Quat(q_cur, d.site_xmat[self.site_id])
        q_neg, q_err, rot_err = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(q_neg, q_cur)
        mujoco.mju_mulQuat(q_err, self.quat_des, q_neg)
        mujoco.mju_quat2Vel(rot_err, q_err, 1.0)
        return pos_err, rot_err

    def solve_ik(self, xy, q_init=None, iters: int = 20, tol: float = 1e-5) -> np.ndarray:
        q = (self.q_des if q_init is None else np.asarray(q_init, float)).copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lam2 = 1e-4
        for _ in range(iters):
            pos_err, rot_err = self._tip_error(q, xy)
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.site_id)
            J = np.vstack([jacp, jacr])[:, self.dadr]
            JJt = J @ J.T + lam2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Nullspace: stay close to the home posture (keeps the elbow up, avoids limits).
            J_pinv = J.T @ np.linalg.inv(JJt)
            dq += (np.eye(7) - J_pinv @ J) @ (0.1 * (self.q_home - q))
            q = np.clip(q + dq, self.qlo, self.qhi)
        return q

    def tip_error(self, q) -> tuple[float, float]:
        """(position error [m], orientation error [rad]) of joint config q w.r.t. the command."""
        pos_err, rot_err = self._tip_error(np.asarray(q, float), self.cmd_xy)
        return float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

    # ------------------------------------------------------------------ commands
    def clamp(self, xy) -> np.ndarray:
        return np.clip(np.asarray(xy, float), self.ws_lo, self.ws_hi)

    def reset(self, xy) -> np.ndarray:
        """Jump (no rate limit) to a tip position; returns the joint configuration."""
        self.cmd_xy = self.clamp(xy)
        self.q_des = self.solve_ik(self.cmd_xy, q_init=self.q_home, iters=300, tol=1e-8)
        return self.q_des.copy()

    def command(self, setpoint_xy) -> np.ndarray:
        """Move the command towards `setpoint_xy` (clamped + rate-limited).

        Returns the new commanded tip xy (this is the recorded action).
        """
        goal = self.clamp(setpoint_xy)
        delta = goal - self.cmd_xy
        dist = np.linalg.norm(delta)
        if dist > self.max_step:
            delta *= self.max_step / dist
        self.cmd_xy = self.cmd_xy + delta
        self.q_des = self.solve_ik(self.cmd_xy)
        return self.cmd_xy.copy()
