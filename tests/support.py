"""Utilidades compartidas por las pruebas del paquete de Colab y del portal."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"
NOTEBOOK = COLAB_DIR / "TRELLIS_AutoRig_Colab.ipynb"
SETUP_SCRIPT = COLAB_DIR / "setup_colab.sh"
LOCKFILE = COLAB_DIR / "requirements-colab.lock"
LOCK_INPUT = COLAB_DIR / "requirements-colab.in"
PORTAL_DIR = ROOT / "portal"


def load_build_module():
    """Carga scripts/build_colab_zip.py (no es un paquete importable)."""
    path = ROOT / "scripts" / "build_colab_zip.py"
    spec = importlib.util.spec_from_file_location("build_colab_zip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
