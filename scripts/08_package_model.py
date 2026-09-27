"""Package a trained checkpoint so it runs anywhere, and optionally upload it to the Hugging Face Hub.

    .venv/bin/python scripts/08_package_model.py                                   # best checkpoint of outputs/dp_image
    .venv/bin/python scripts/08_package_model.py --checkpoint 080000
    .venv/bin/python scripts/08_package_model.py --upload <hf-user>/pusht-franka-diffusion-policy

The package (release/<run>/, ~350 MB) holds only what inference needs: the weights and configs of
pretrained_model/ (no optimizer state), plus pusht_meta.json with the fps, cameras, image resolution
and physics XML of the training data. With that file, evaluate.py and demo/sim_demo.py run the model
without the dataset or the raw demos. Evaluation results (eval/<name>/results.json) and a model card
(README.md) are included too.

Uploading needs a Hugging Face account and `.venv/bin/hf auth login` once. Afterwards, set
PRETRAINED_REPO in training/policy_runner.py to the same repo id, so that clones of this project can
fetch the model with scripts/09_download_model.py (the demo does it automatically).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from policy_runner import META_FILE, best_checkpoint, eval_results, resolve_checkpoint, task_meta  # noqa: E402
from pusht.config import resolve  # noqa: E402


def relative_paths(obj):
    """Absolute paths inside the project -> project-relative (train_config.json leaks local paths)."""
    if isinstance(obj, dict):
        return {k: relative_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relative_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(str(PROJECT_ROOT) + "/"):
        return obj[len(str(PROJECT_ROOT)) + 1:]
    return obj


def model_card(run: str, ckpt: str, meta: dict, results: dict[str, dict], inputs: list[str]) -> str:
    rows = "\n".join(f"| {r.get('sampler', 'DDPM')} | {r['episodes']} | {r['success_rate'] * 100:.0f} % |"
                     for r in results.values())
    return f"""---
library_name: lerobot
pipeline_tag: robotics
tags: [robotics, diffusion-policy, push-t, franka, mujoco, imitation-learning]
---

# Push-T Diffusion Policy for a Franka Panda with a stick (simulation)

Run `{run}`, checkpoint `{ckpt}`. A LeRobot Diffusion Policy that pushes a T-shaped block onto a fixed
T-shaped target with a stick held by a Franka Panda, in MuJoCo.

- **Inputs:** {", ".join(f"`{k}`" for k in inputs)} (images {", ".join(f"{c} {h}×{w}" for c, (h, w) in meta["resolution"].items())})
- **Output:** commanded stick-tip `[x, y]` on the table (metres) at {meta["fps"]} Hz
- **Data:** human mouse-teleoperated demos in simulation

| Sampler | Episodes | Success (≥ 90 % coverage for 0.5 s) |
|---|---|---|
{rows}

`{META_FILE}` stores the fps, camera definitions, image resolution and physics XML of the training data,
so the simulator can reproduce the exact training conditions. Code, simulator and usage: see the
project's GitHub repository (`scripts/09_download_model.py`, `demo/sim_demo.py`).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/dp_image", help="training run folder")
    ap.add_argument("--checkpoint", default=None, help="default: best evaluated checkpoint")
    ap.add_argument("--out", default=None, help="default: release/<run name>")
    ap.add_argument("--upload", metavar="REPO_ID", default=None, help="Hugging Face model repo to upload to")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    args = ap.parse_args()

    run_dir = resolve(args.run)
    ckpt = args.checkpoint or best_checkpoint(run_dir)
    if ckpt is None:
        sys.exit(f"{run_dir} has no evaluation results; pass --checkpoint or run training/evaluate.py first.")
    src = resolve_checkpoint(run_dir, ckpt)
    out = resolve(args.out or f"release/{run_dir.name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for f in src.iterdir():
        if f.name == "train_config.json":
            (out / f.name).write_text(json.dumps(relative_paths(json.loads(f.read_text())), indent=4))
        elif f.is_file() and f.name != META_FILE:
            shutil.copy2(f, out / f.name)

    meta = task_meta(src)
    if meta["physics_xml"] is None:
        sys.exit("No raw demos found for this checkpoint's dataset: cannot store the demo physics.")
    cfg = json.loads((src / "config.json").read_text())
    inputs = list(cfg["input_features"])
    meta = {"run": run_dir.name, "checkpoint": ckpt, "inputs": inputs, **meta,
            "cameras": {c: meta["cameras"][c] for c in meta["cameras"] if f"observation.images.{c}" in inputs},
            "resolution": {c: meta["resolution"][c] for c in meta["resolution"] if f"observation.images.{c}" in inputs}}
    (out / META_FILE).write_text(json.dumps(meta, indent=2))

    results = {name: r for name, r in eval_results(run_dir).items() if name == ckpt or name.startswith(ckpt + "_")}
    for name, r in results.items():
        (out / "eval" / name).mkdir(parents=True)
        (out / "eval" / name / "results.json").write_text(json.dumps(r, indent=2))
    (out / "README.md").write_text(model_card(run_dir.name, ckpt, meta, results, inputs))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"packaged {run_dir.name} / {ckpt} -> {out} ({size:.0f} MB)")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.upload, folder_path=out, repo_type="model",
                          commit_message=f"{run_dir.name} checkpoint {ckpt}")
        print(f"uploaded to https://huggingface.co/{args.upload}")
        print(f'now set PRETRAINED_REPO = "{args.upload}" in training/policy_runner.py')


if __name__ == "__main__":
    main()
