"""Build the MuJoCo Push-T scene from pusht_scene.xml + configs/*.yaml.

The *physics* scene (robot, table, T block, target) is saved as XML inside every recorded
episode. Cameras never affect physics, so they are added separately on top of that XML;
this is what lets us re-render old demos from new camera positions.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .config import ASSET_DIR, SCENE_XML
from .geometry import TShape, quat_from_yaw

BLOCK_BODY = "t_block"
BLOCK_JOINT = "t_block_joint"
TARGET_BODY = "target"
TIP_SITE = "tip"
ROBOT_ROOT = "link0"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def build_physics_spec(scene_cfg: dict) -> mujoco.MjSpec:
    """Robot + table + T block + target outline, no cameras."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.meshdir = str(ASSET_DIR / "assets")  # absolute, so the saved XML loads from anywhere

    shape = TShape.from_config(scene_cfg["t_block"])
    tb = scene_cfg["t_block"]
    tg = scene_cfg["target"]

    table = spec.geom("table")
    table.friction = scene_cfg["table"]["friction"]
    if "rgba" in scene_cfg["table"]:
        table.material = ""
        table.rgba = scene_cfg["table"]["rgba"]

    light = scene_cfg.get("lighting")
    if light:
        spec.visual.headlight.diffuse = [light["headlight_diffuse"]] * 3
        spec.visual.headlight.ambient = [light["headlight_ambient"]] * 3
        for lt in spec.lights:
            lt.diffuse = [light["top_light_diffuse" if lt.name == "top" else "directional_diffuse"]] * 3
            if lt.name == "top":
                lt.castshadow = False  # this light follows the robot; its shadows swing around

    # Movable T block (free joint: it rests on and slides over the table).
    block = spec.worldbody.add_body(name=BLOCK_BODY, pos=[tg["x"], tg["y"], shape.height / 2])
    block.add_freejoint(name=BLOCK_JOINT)
    for i, (ctr, half, frac) in enumerate(shape.rects()):
        block.add_geom(
            name=f"t_block_{'bar' if i == 0 else 'stem'}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half[0], half[1], shape.height / 2],
            pos=[ctr[0], ctr[1], 0.0],
            mass=float(tb["mass"]) * frac,
            friction=tb["friction"],
            rgba=tb["rgba"],
        )

    # Fixed target: a thin, visual-only T drawn on the table.
    target = spec.worldbody.add_body(name=TARGET_BODY, pos=[tg["x"], tg["y"], 0.0],
                                     quat=quat_from_yaw(tg["yaw"]))
    for i, (ctr, half, _) in enumerate(shape.rects()):
        target.add_geom(
            name=f"target_{'bar' if i == 0 else 'stem'}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half[0], half[1], 0.0005],
            pos=[ctr[0], ctr[1], 0.0006],
            contype=0, conaffinity=0,
            rgba=tg["rgba"],
        )

    if scene_cfg["robot"].get("gravity_compensation", True):
        _set_gravcomp(spec.body(ROBOT_ROOT))
    tau = scene_cfg["robot"].get("actuator_time_constant")
    if tau is not None:
        for act in spec.actuators:
            act.biasprm[2] = -float(tau) * act.gainprm[0]
    return spec


def _set_gravcomp(body: mujoco.MjsBody) -> None:
    body.gravcomp = 1.0
    for child in body.bodies:
        _set_gravcomp(child)


def _look_at_quat(pos, lookat, up) -> np.ndarray:
    """MuJoCo camera looks along its -z axis with +y as image up."""
    z = np.asarray(pos, float) - np.asarray(lookat, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up if up is not None else [0.0, 0.0, 1.0], float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # up parallel to viewing direction
        x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


def add_camera(spec: mujoco.MjSpec, name: str, cam: dict) -> None:
    parent = spec.body(cam["body"]) if "body" in cam else spec.worldbody
    parent.add_camera(
        name=name,
        pos=cam["pos"],
        quat=_look_at_quat(cam["pos"], cam["lookat"], cam.get("up")),
        fovy=float(cam.get("fovy", 45)),
    )


def physics_xml(scene_cfg: dict) -> str:
    """Compiled physics scene as a self-contained XML string (stored in each episode)."""
    spec = build_physics_spec(scene_cfg)
    spec.compile()
    return spec.to_xml()


def model_from_xml(xml: str, cameras: dict[str, dict] | None = None) -> mujoco.MjModel:
    """Load a physics XML (e.g. from a recorded episode) and attach the given cameras."""
    spec = mujoco.MjSpec.from_string(xml)
    # The stored XML names the meshdir of the machine that recorded it; the meshes are this project's.
    spec.meshdir = str(ASSET_DIR / "assets")
    for name, cam in (cameras or {}).items():
        add_camera(spec, name, cam)
    return spec.compile()


def all_cameras(cameras_cfg: dict) -> dict[str, dict]:
    """Dataset cameras + the teleop view camera, keyed by name."""
    cams = dict(cameras_cfg.get("cameras", {}))
    if "teleop_view" in cameras_cfg:
        cams["teleop_view"] = cameras_cfg["teleop_view"]
    return cams
