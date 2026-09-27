"""Check the software stack: MuJoCo + GPU (EGL) rendering, torch + CUDA, LeRobot, video I/O.

    .venv/bin/python scripts/00_check_env.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)

ok = True


def report(name, fn):
    global ok
    try:
        print(f"[ok]   {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def mujoco_render():
    import mujoco
    from OpenGL import GL

    from pusht.env import PushTEnv
    from pusht.render import CameraRenderer

    cams = {"c": {"pos": [1.0, 0, 0.8], "lookat": [0.5, 0, 0], "fovy": 45}}
    env = PushTEnv(cameras=cams)
    env.reset(seed=0)
    r = CameraRenderer(env.model, cams)
    r.render(env.data, "c", (480, 480))
    t = time.time()
    for _ in range(20):
        img = r.render(env.data, "c", (480, 480))
    ms = (time.time() - t) / 20 * 1e3
    gpu = GL.glGetString(GL.GL_RENDERER).decode()
    r.close()
    assert img.std() > 1, "blank image"
    return f"mujoco {mujoco.__version__}, {gpu}, {ms:.1f} ms/frame @480x480"


def torch_cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(1024, 1024, device="cuda")
    (x @ x).sum().item()
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, {mem:.1f} GB"


def lerobot_import():
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: F401

    return f"lerobot {lerobot.__version__} (dataset + diffusion policy)"


def video():
    import av

    have = [c for c in ("libsvtav1", "libx264", "h264") if c in av.codecs_available]
    try:
        import torchcodec  # noqa: F401

        tc = "torchcodec OK"
    except Exception:  # noqa: BLE001
        tc = "torchcodec unavailable (use video_backend=pyav, or `sudo apt install ffmpeg`)"
    return f"pyav {av.__version__}, encoders {have}; {tc}"


def pygame_ok():
    import pygame

    return f"pygame {pygame.version.ver}"


report("MuJoCo + EGL rendering", mujoco_render)
report("PyTorch + CUDA", torch_cuda)
report("LeRobot", lerobot_import)
report("Video I/O", video)
report("pygame (teleop window)", pygame_ok)
print("\nEnvironment OK" if ok else "\nSome checks FAILED")
sys.exit(0 if ok else 1)
