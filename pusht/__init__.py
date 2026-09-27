"""Push-T imitation learning: MuJoCo Panda-stick simulation + data pipeline."""

import os

# Offscreen rendering on the NVIDIA GPU. Must be set before `mujoco` is first imported.
# Without MUJOCO_EGL_DEVICE_ID, EGL picks the Intel iGPU here (~10x slower rendering).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


# Quiet third-party chatter during dataset export: SVT-AV1 prints ~20 info lines per
# encoded video (1 = errors only), and HF `datasets` shows a progress bar per episode.
os.environ.setdefault("SVT_LOG", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
