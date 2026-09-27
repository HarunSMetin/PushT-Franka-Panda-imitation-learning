"""Load a trained policy and turn observations into actions.

This is the only piece that sits between a trained checkpoint and a robot. The simulator
evaluation uses it now; the real Franka will use the same class in phase 3.

    runner = PolicyRunner("outputs/dp_image")            # newest checkpoint of a run
    runner = PolicyRunner("outputs/dp_image", "050000")  # a specific checkpoint
    runner.reset()                                       # at the start of every episode
    xy = runner.act({
        "observation.images.overhead": rgb_uint8_hxwx3,
        "observation.images.wrist":    rgb_uint8_hxwx3,
        "observation.state":           np.array([tip_x, tip_y]),
    })                                                   # -> commanded tip [x, y] in metres

Call act() once per control tick at the dataset fps (10 Hz). Internally the policy keeps the
last n_obs_steps observations and a queue of planned actions; it only runs the (slow)
diffusion model when the queue is empty, i.e. every n_action_steps ticks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_FILE = "pusht_meta.json"  # written by scripts/08_package_model.py into a packaged model
# Hugging Face Hub model repo that scripts/09_download_model.py and demo/sim_demo.py fetch from.
PRETRAINED_REPO = "HSM1/pusht-franka-diffusion-policy"


def resolve_checkpoint(run_or_ckpt: str | Path, checkpoint: str | None = None) -> Path:
    """Run dir / checkpoint dir / pretrained_model dir -> the pretrained_model directory."""
    p = Path(run_or_ckpt)
    if (p / "config.json").exists() and (p / "model.safetensors").exists():
        return p
    if (p / "pretrained_model").is_dir():
        return p / "pretrained_model"
    ckpts = p / "train" / "checkpoints"
    if not ckpts.is_dir():
        raise FileNotFoundError(f"No checkpoints found under {p}")
    name = checkpoint or "last"
    path = ckpts / name / "pretrained_model"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found. Available: {list_checkpoints(p)}")
    return path.resolve()


def list_checkpoints(run_dir: str | Path) -> list[str]:
    ckpts = Path(run_dir) / "train" / "checkpoints"
    return sorted(d.name for d in ckpts.iterdir() if d.name != "last" and d.is_dir()) if ckpts.is_dir() else []


class PolicyRunner:
    def __init__(self, run_or_ckpt: str | Path, checkpoint: str | None = None, device: str = "cuda",
                 inference_steps: int | None = None):
        """inference_steps: None = same sampler as training (DDPM, 100 denoising steps).
        A small number (e.g. 10) switches to the DDIM sampler: ~10x faster planning with the
        same trained weights, usually at similar quality; useful for real-time use."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.path = resolve_checkpoint(run_or_ckpt, checkpoint)
        cfg = PreTrainedConfig.from_pretrained(self.path)
        cfg.pretrained_path = self.path
        cfg.device = device
        if inference_steps is not None:
            cfg.noise_scheduler_type = "DDIM"
            cfg.num_inference_steps = int(inference_steps)
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(self.path, config=cfg)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=self.path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.device = torch.device(device)
        self.input_keys = list(cfg.input_features)
        self.image_keys = [k for k in self.input_keys if k.startswith("observation.images.")]
        self.cameras = [k.split(".")[-1] for k in self.image_keys]

    @cached_property
    def meta(self) -> dict:
        """How the training data was made (see task_meta)."""
        return task_meta(self.path)

    @property
    def fps(self) -> int:
        """Control rate the policy was trained at (from the dataset)."""
        return int(self.meta["fps"])

    def reset(self) -> None:
        """Clear the observation history and planned actions (start of an episode)."""
        self.policy.reset()

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """One observation -> action (2,), or a batch (leading dim N) -> actions (N, 2).

        Images are HxWx3 (or NxHxWx3) uint8, vectors float arrays. A batch runs N independent
        episodes in one GPU call (they must all be reset() together and stepped in lockstep).
        """
        batched = np.asarray(obs["observation.state"]).ndim == 2
        batch = {}
        for k in self.input_keys:
            v = obs[k]
            if k in self.image_keys:
                t = torch.from_numpy(np.ascontiguousarray(v)).float() / 255.0
                t = t.permute(0, 3, 1, 2) if batched else t.permute(2, 0, 1)  # -> channels first
            else:
                t = torch.as_tensor(np.asarray(v, dtype=np.float32))
            batch[k] = t
        batch = self.preprocessor(batch)
        amp = torch.autocast(device_type=self.device.type) if self.cfg.use_amp else nullcontext()
        with torch.inference_mode(), amp:
            action = self.policy.select_action(batch)
        action = self.postprocessor(action).float().cpu().numpy()
        return action if batched else action[0]

    def planned_actions(self) -> np.ndarray:
        """The rest of the current plan (actions queued for the next ticks), in metres. (k, 2)"""
        queue = getattr(self.policy, "_queues", {}).get("action", [])
        if not queue:
            return np.zeros((0, 2))
        return np.stack([self.postprocessor(a.clone()).squeeze(0).float().cpu().numpy() for a in queue])


def eval_results(run_dir: str | Path) -> dict[str, dict]:
    """All evaluation summaries of a run, keyed by eval folder name ("080000", "080000_ddim10", ...)."""
    out = {}
    for f in sorted((Path(run_dir) / "eval").glob("*/results.json")):
        out[f.parent.name] = json.loads(f.read_text())
    return out


def best_checkpoint(run_dir: str | Path) -> str | None:
    """Checkpoint with the highest success rate (ties: higher mean coverage), from evaluate.py results
    obtained with the training sampler. None if the run has not been evaluated yet."""
    ranked = [(r["success_rate"], r["mean_max_coverage"], name) for name, r in eval_results(run_dir).items()
              if "_" not in name and r.get("episodes", 0) >= 20]
    return max(ranked)[2] if ranked else None


def task_meta(pretrained_dir: str | Path) -> dict:
    """What the simulator needs to run a policy exactly as in training:
        fps          dataset / control rate of the policy
        cameras      camera definitions the images were rendered with (configs/cameras.yaml format)
        resolution   {camera: [height, width]}
        physics_xml  physics of the recorded demos (None: build it from configs/scene.yaml)

    A packaged model (scripts/08_package_model.py) carries this in pusht_meta.json. A local training
    checkpoint reads it from its dataset (meta/info.json, meta/export_info.json) and raw demos."""
    pretrained_dir = Path(pretrained_dir)
    if (pretrained_dir / META_FILE).exists():
        return json.loads((pretrained_dir / META_FILE).read_text())
    train_cfg = json.loads((pretrained_dir / "train_config.json").read_text())
    root = PROJECT_ROOT / train_cfg["dataset"]["root"]  # absolute paths pass through the join
    if not (root / "meta" / "export_info.json").exists():
        raise FileNotFoundError(
            f"{pretrained_dir} has no {META_FILE} and its dataset {root} is missing. "
            "Export the dataset (scripts/04_export_lerobot.py) or use a packaged model "
            "(scripts/09_download_model.py).")
    info = json.loads((root / "meta" / "info.json").read_text())
    export = json.loads((root / "meta" / "export_info.json").read_text())
    physics_xml = None
    raw = sorted((PROJECT_ROOT / export["raw_dir"]).glob("episode_*.hdf5"))
    if raw:
        import h5py
        with h5py.File(raw[0], "r") as f:
            physics_xml = f["meta/physics_xml"][()]
        physics_xml = physics_xml.decode() if isinstance(physics_xml, bytes) else physics_xml
    return {"fps": int(info["fps"]), "cameras": export["cameras"], "resolution": export["resolution"],
            "physics_xml": physics_xml}


def install_pretrained(src_dir: str | Path, run_dir: str | Path) -> str:
    """Copy a packaged model (a folder made by scripts/08_package_model.py, e.g. a Hub snapshot) into
    the run layout that evaluate.py and the demo expect; returns the checkpoint name:
        <run_dir>/train/checkpoints/<ckpt>/pretrained_model/   weights, configs, pusht_meta.json
        <run_dir>/train/checkpoints/last -> <ckpt>
        <run_dir>/eval/<name>/results.json                    its evaluation results"""
    src, run_dir = Path(src_dir), Path(run_dir)
    ckpt = json.loads((src / META_FILE).read_text())["checkpoint"]
    ckpts = run_dir / "train" / "checkpoints"
    dst = ckpts / ckpt / "pretrained_model"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "README.md" and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)
    for res in (src / "eval").glob("*/results.json"):
        (run_dir / "eval" / res.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(res, run_dir / "eval" / res.parent.name / "results.json")
    last = ckpts / "last"
    if not last.exists():
        os.symlink(ckpt, last)
    return ckpt


def download_pretrained(run_dir: str | Path, repo_id: str = PRETRAINED_REPO) -> str:
    """Download the packaged model from the Hugging Face Hub and install it into run_dir."""
    from huggingface_hub import snapshot_download

    print(f"downloading pretrained model {repo_id} (~350 MB) ...", flush=True)
    return install_pretrained(snapshot_download(repo_id), run_dir)
