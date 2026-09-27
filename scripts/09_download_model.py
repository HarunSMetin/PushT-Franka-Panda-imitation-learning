"""Download the pretrained model from the Hugging Face Hub (or install a local package).

    .venv/bin/python scripts/09_download_model.py                      # repo = PRETRAINED_REPO in training/policy_runner.py
    .venv/bin/python scripts/09_download_model.py --repo <hf-user>/pusht-franka-diffusion-policy
    .venv/bin/python scripts/09_download_model.py --from-dir release/dp_image   # package made by 08_package_model.py

The model is installed as outputs/dp_image/train/checkpoints/<checkpoint>/pretrained_model/ with its
evaluation results, so it works like a locally trained checkpoint:

    .venv/bin/python demo/sim_demo.py
    .venv/bin/python training/evaluate.py outputs/dp_image
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from policy_runner import PRETRAINED_REPO, download_pretrained, install_pretrained  # noqa: E402
from pusht.config import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=PRETRAINED_REPO, help="Hugging Face model repo id")
    ap.add_argument("--from-dir", default=None, help="install a local package instead of downloading")
    ap.add_argument("--run", default="outputs/dp_image", help="run folder to install into")
    args = ap.parse_args()

    run_dir = resolve(args.run)
    if args.from_dir:
        ckpt = install_pretrained(resolve(args.from_dir), run_dir)
    else:
        ckpt = download_pretrained(run_dir, args.repo)
    print(f"installed checkpoint {ckpt} -> {run_dir / 'train' / 'checkpoints' / ckpt}")


if __name__ == "__main__":
    main()
