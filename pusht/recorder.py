"""Raw episode storage (HDF5). One file per demo: data/raw/episode_XXXXXX.hdf5.

Per control step t (observation BEFORE the action, action applied during t -> t+1):
    state        full MuJoCo integration state (exact replay / re-render / re-simulation)
    qpos, qvel   convenience copies
    tip_xy       measured stick tip position
    block_pose   (x, y, yaw) of the T block
    coverage     target coverage in [0, 1]
    setpoint_xy  raw operator input (mouse on the table), before clamp/rate-limit
    action_xy    commanded tip xy after clamp + rate-limit  <- the policy action
    q_des        joint targets sent to the actuators for this step
    mouse_down   operator was actively steering
    time         sim time [s]
Metadata: seeds, success flag, discard flag, notes, physics XML and all configs (verbatim).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

import h5py
import mujoco
import numpy as np

from .config import config_text

_EP_RE = re.compile(r"episode_(\d{6})\.hdf5$")
_SERIES = ("state", "qpos", "qvel", "tip_xy", "block_pose", "coverage",
           "setpoint_xy", "action_xy", "q_des", "mouse_down", "time")


def list_episodes(raw_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).glob("episode_*.hdf5") if _EP_RE.search(p.name))


def next_episode_index(raw_dir: str | Path) -> int:
    eps = list_episodes(raw_dir)
    return int(_EP_RE.search(eps[-1].name).group(1)) + 1 if eps else 0


class EpisodeRecorder:
    def __init__(self, env, raw_dir: str | Path):
        self.env = env
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.buf: dict[str, list] | None = None
        self.meta: dict = {}

    @property
    def active(self) -> bool:
        return self.buf is not None

    def __len__(self) -> int:
        return len(self.buf["time"]) if self.buf else 0

    def start(self, seed: int | None, init_block_pose, init_tip_xy, **meta) -> None:
        self.buf = {k: [] for k in _SERIES}
        self.meta = dict(seed=-1 if seed is None else int(seed),
                         init_block_pose=np.asarray(init_block_pose, float),
                         init_tip_xy=np.asarray(init_tip_xy, float),
                         init_cmd_xy=self.env.ctrl.cmd_xy.copy(),  # controller state for re-simulation
                         **meta)

    def record(self, state, obs: dict, setpoint_xy, mouse_down: bool) -> None:
        """Call right AFTER env.step(): `state`/`obs` are from BEFORE that step."""
        d = self.env.data
        b = self.buf
        nq, nv = self.env.model.nq, self.env.model.nv
        b["state"].append(np.asarray(state, float))
        b["qpos"].append(np.asarray(state[1:1 + nq], float))
        b["qvel"].append(np.asarray(state[1 + nq:1 + nq + nv], float))
        b["tip_xy"].append(obs["tip_xy"])
        b["block_pose"].append(obs["block_pose"])
        b["coverage"].append(obs["coverage"])
        b["setpoint_xy"].append(np.asarray(setpoint_xy, float))
        b["action_xy"].append(self.env.ctrl.cmd_xy.copy())
        b["q_des"].append(d.ctrl.copy())
        b["mouse_down"].append(bool(mouse_down))
        b["time"].append(float(state[0]))

    def discard(self) -> None:
        self.buf = None

    def save(self, success: bool, note: str = "") -> Path:
        """Write the episode atomically (tmp file + rename). Returns the final path."""
        assert self.buf and len(self) > 0, "nothing recorded"
        env = self.env
        idx = next_episode_index(self.raw_dir)
        path = self.raw_dir / f"episode_{idx:06d}.hdf5"
        tmp = path.with_suffix(".hdf5.tmp")
        with h5py.File(tmp, "w") as f:
            for k in _SERIES:
                arr = np.asarray(self.buf[k])
                f.create_dataset(k, data=arr, compression="gzip" if arr.ndim > 1 else None)
            f.create_dataset("final_state", data=env.get_state())
            f.create_dataset("final_block_pose", data=env.block_pose())
            m = f.create_group("meta")
            m.create_dataset("physics_xml", data=env.xml)
            for name in ("scene", "collect", "cameras"):
                m.create_dataset(f"{name}_yaml", data=config_text(name))
            a = f.attrs
            a["episode_index"] = idx
            a["n_steps"] = len(self)
            a["control_hz"] = env.control_hz
            a["success"] = bool(success)
            a["final_coverage"] = float(self.buf["coverage"][-1])
            a["max_coverage"] = float(np.max(self.buf["coverage"]))
            a["discard"] = False
            a["note"] = note
            a["mujoco_version"] = mujoco.__version__
            a["state_spec"] = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            a["created"] = _dt.datetime.now().isoformat(timespec="seconds")
            for k, v in self.meta.items():
                a[k] = v
            f.flush()
            os.fsync(f.id.get_vfd_handle())
        os.replace(tmp, path)
        self.buf = None
        return path


def load_episode(path: str | Path) -> dict:
    """All series as numpy arrays + attrs + stored config/XML strings."""
    out: dict = {}
    with h5py.File(path, "r") as f:
        for k in (*_SERIES, "final_state", "final_block_pose"):
            out[k] = f[k][()]
        for k in f["meta"]:
            v = f["meta"][k][()]
            out[k] = v.decode() if isinstance(v, bytes) else v
        out["attrs"] = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    out["path"] = Path(path)
    return out


def episode_attrs(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}


def set_attr(path: str | Path, key: str, value) -> None:
    with h5py.File(path, "r+") as f:
        f.attrs[key] = value
