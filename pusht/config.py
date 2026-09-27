"""Config loading. All configs live in <project>/configs/*.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
ASSET_DIR = PROJECT_ROOT / "assets" / "franka_emika_panda"
SCENE_XML = ASSET_DIR / "pusht_scene.xml"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml ("scene", "cameras" or "collect")."""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def config_text(name: str) -> str:
    """Raw YAML text, stored verbatim inside every recorded episode."""
    return (CONFIG_DIR / f"{name}.yaml").read_text()


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
