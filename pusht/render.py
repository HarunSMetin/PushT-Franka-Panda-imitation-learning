"""Offscreen rendering of any configured camera from a simulator state.

GPU offscreen rendering via EGL (configured in pusht/__init__.py).
"""

from __future__ import annotations

import mujoco
import numpy as np

ROBOT_VISUAL_GROUP = 2  # Panda visual meshes live in geom group 2 (stick is group 0)


class CameraRenderer:
    """Renders named cameras of `model`. Camera settings come from configs/cameras.yaml."""

    def __init__(self, model: mujoco.MjModel, cameras: dict[str, dict]):
        self.model = model
        self.cameras = cameras
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._opt_default = mujoco.MjvOption()
        self._opt_default.sitegroup[:] = 0  # no debug markers (red tip site) in dataset images
        self._opt_no_robot = mujoco.MjvOption()
        self._opt_no_robot.geomgroup[ROBOT_VISUAL_GROUP] = 0

    def _renderer(self, h: int, w: int) -> mujoco.Renderer:
        key = (h, w)
        if key not in self._renderers:
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
            self._renderers[key] = mujoco.Renderer(self.model, h, w)
        return self._renderers[key]

    def render(self, data: mujoco.MjData, name: str, resolution=None) -> np.ndarray:
        """RGB uint8 image (H, W, 3) from camera `name`."""
        cam = self.cameras[name]
        h, w = resolution or cam.get("resolution", (96, 96))
        r = self._renderer(int(h), int(w))
        opt = self._opt_no_robot if cam.get("hide_robot") else self._opt_default
        r.update_scene(data, camera=name, scene_option=opt)
        return r.render().copy()

    def render_free(self, data: mujoco.MjData, camera: mujoco.MjvCamera, resolution,
                    hide_robot: bool = False) -> np.ndarray:
        """RGB image from a free (orbit) camera: lookat / distance / azimuth / elevation."""
        r = self._renderer(int(resolution[0]), int(resolution[1]))
        r.update_scene(data, camera=camera, scene_option=self._opt_no_robot if hide_robot else self._opt_default)
        return r.render().copy()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


class ThreadedCameraRenderer:
    """CameraRenderer whose OpenGL work runs on a dedicated thread.

    Needed next to a pygame window: on Wayland, SDL presents the window through its own
    EGL context on the main thread. MuJoCo making its context current on that same thread
    makes SDL draw into MuJoCo's offscreen buffer -> black window. EGL contexts are
    per-thread, so rendering on a private thread keeps the two apart.
    """

    def __init__(self, model: mujoco.MjModel, cameras: dict[str, dict]):
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-render")
        self._inner = self._pool.submit(CameraRenderer, model, cameras).result()

    def render(self, data: mujoco.MjData, name: str, resolution=None) -> np.ndarray:
        # The caller blocks until the frame is done, so `data` is not modified meanwhile.
        return self._pool.submit(self._inner.render, data, name, resolution).result()

    def render_free(self, data, camera, resolution, hide_robot: bool = False) -> np.ndarray:
        return self._pool.submit(self._inner.render_free, data, camera, resolution, hide_robot).result()

    def close(self) -> None:
        self._pool.submit(self._inner.close).result()
        self._pool.shutdown()


def world_to_pixel(model: mujoco.MjModel, data: mujoco.MjData, cam_name: str, xyz, h: int, w: int):
    """Project a world point into (u, v) pixel coords of a pinhole camera."""
    cid = model.camera(cam_name).id
    pos = data.cam_xpos[cid]
    R = data.cam_xmat[cid].reshape(3, 3)
    p = R.T @ (np.asarray(xyz, float) - pos)  # camera frame; camera looks along -z
    f = 0.5 * h / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    return np.array([w / 2 + f * p[0] / -p[2], h / 2 - f * p[1] / -p[2]])


def pixel_to_plane(model: mujoco.MjModel, data: mujoco.MjData, cam_name: str, uv, h: int, w: int,
                   z: float = 0.0) -> np.ndarray:
    """Intersect the ray through pixel (u, v) with the horizontal plane at height z -> (x, y)."""
    cid = model.camera(cam_name).id
    pos = data.cam_xpos[cid]
    R = data.cam_xmat[cid].reshape(3, 3)
    f = 0.5 * h / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    d_cam = np.array([(uv[0] - w / 2) / f, -(uv[1] - h / 2) / f, -1.0])
    d = R @ d_cam
    t = (z - pos[2]) / d[2]
    return (pos + t * d)[:2]
