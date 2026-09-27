"""Train a Diffusion Policy with LeRobot from one of training/configs/*.yaml.

    .venv/bin/python training/train.py training/configs/dp_image.yaml
    .venv/bin/python training/train.py training/configs/dp_image.yaml --resume            # continue
    .venv/bin/python training/train.py training/configs/dp_image.yaml --resume --steps 150000
    .venv/bin/python training/train.py training/configs/dp_image.yaml --steps 300 --run-name test

Everything a run produces goes to outputs/<run_name>/:
    run_config.yaml     copy of the config used
    train_log.txt       full training log (loss, learning rate, speed)
    train/checkpoints/  LeRobot checkpoints: 010000/, 020000/, ..., last -> newest
    eval/               written later by training/evaluate.py

The model, optimizer and batches run on the GPU (cuda); `use_amp` enables mixed precision.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pusht  # noqa: E402,F401  (quiets video-encoder / dataset progress output)

# torchcodec needs system FFmpeg; LeRobot falls back to PyAV. Drop the 100-line warning.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

OUTPUTS = PROJECT_ROOT / "outputs"


class Tee:
    """Write everything printed to the console into the log file as well."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        if "\r" not in data:  # skip progress-bar redraws, keep real log lines
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


def build_train_config(cfg: dict, run_dir: Path, steps_override: int | None):
    """YAML dict -> LeRobot TrainPipelineConfig (policy inputs restricted to cfg['inputs'])."""
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features

    ds_root = PROJECT_ROOT / cfg["dataset"]["root"]
    if not (ds_root / "meta" / "info.json").exists():
        sys.exit(f"Dataset not found at {ds_root}. Export it first (see the comment in the config).")
    meta = LeRobotDatasetMetadata(cfg["dataset"]["repo_id"], root=ds_root)
    features = dataset_to_policy_features(meta.features)

    missing = [k for k in cfg["inputs"] if k not in features]
    if missing:
        sys.exit(f"Inputs {missing} are not in the dataset. Available: {sorted(features)}")
    input_features = {k: features[k] for k in cfg["inputs"]}
    output_features = {"action": features["action"]}

    pol = dict(cfg["policy"])
    for k in ("crop_shape", "down_dims", "resize_shape"):  # YAML lists -> tuples
        if pol.get(k) is not None:
            pol[k] = tuple(pol[k])
    tr = cfg["training"]
    policy = DiffusionConfig(
        **pol,
        input_features=input_features,
        output_features=output_features,
        device="cuda",
        use_amp=bool(tr.get("use_amp", False)),
        push_to_hub=False,
    )
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=cfg["dataset"]["repo_id"], root=str(ds_root), video_backend="pyav"),
        policy=policy,
        output_dir=run_dir / "train",
        job_name=cfg["run_name"],
        steps=int(steps_override or tr["steps"]),
        batch_size=int(tr["batch_size"]),
        save_freq=int(tr["save_freq"]),
        log_freq=int(tr["log_freq"]),
        num_workers=int(tr["num_workers"]),
        seed=int(tr["seed"]),
        wandb=WandBConfig(enable=False),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="training/configs/<name>.yaml")
    ap.add_argument("--resume", action="store_true", help="continue from outputs/<run>/train/checkpoints/last")
    ap.add_argument("--steps", type=int, default=None, help="override total training steps")
    ap.add_argument("--run-name", default=None, help="override run_name (e.g. for a quick test)")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run with the same name")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.run_name:
        cfg["run_name"] = args.run_name
    run_dir = OUTPUTS / cfg["run_name"]
    last = run_dir / "train" / "checkpoints" / "last" / "pretrained_model" / "train_config.json"

    if args.resume:
        if not last.exists():
            sys.exit(f"Nothing to resume: {last} not found")
    elif run_dir.exists():
        if not args.overwrite:
            sys.exit(f"{run_dir} already exists. Use --resume to continue it, --overwrite to restart, "
                     f"or --run-name to start a separate run.")
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_file = open(run_dir / "train_log.txt", "a")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    from lerobot.scripts.lerobot_train import train

    if args.resume:
        # LeRobot's own resume path: reload the saved config + optimizer state from the checkpoint.
        sys.argv = [sys.argv[0], f"--config_path={last}", "--resume=true"]
        if args.steps:
            sys.argv.append(f"--steps={args.steps}")
        print(f"Resuming {run_dir} from {last.parent.parent.name}")
        train()
    else:
        sys.argv = [sys.argv[0]]  # LeRobot also inspects sys.argv; keep it clean
        print(f"Training run '{cfg['run_name']}' -> {run_dir}")
        print(f"Inputs: {cfg['inputs']}")
        train(build_train_config(cfg, run_dir, args.steps))


if __name__ == "__main__":
    main()
