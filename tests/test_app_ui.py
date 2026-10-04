"""Regresión de la interfaz de app.py (sin torch, CUDA ni TRELLIS).

La interfaz no llegó a construirse con Gradio real: `from __future__ import annotations` hacía
que `request: gr.Request` se resolviera en el módulo, donde `gr` no existe (es local de
build_demo), y Gradio fallaba con `NameError: name 'gr' is not defined`. En Colab eso ocurría
después de instalar y cargar el modelo.
"""

import ast
import importlib.util
import sys
import unittest
from types import SimpleNamespace

from support import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_GRADIO = importlib.util.find_spec("gradio") is not None


class AnnotationRegressionTests(unittest.TestCase):
    def test_app_does_not_defer_annotations(self):
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        deferred = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        ]
        self.assertEqual(deferred, [], "app.py no puede usar `from __future__ import annotations`")


@unittest.skipUnless(HAS_GRADIO, "requiere Gradio (pip install -r requirements.txt)")
class BuildDemoTests(unittest.TestCase):
    @staticmethod
    def stub_runtime():
        import gradio as gr

        return {
            "gr": gr,
            "np": SimpleNamespace(),
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "imageio": SimpleNamespace(),
            "Gaussian": object,
            "pipeline": SimpleNamespace(),
            "postprocessing_utils": SimpleNamespace(),
            "render_utils": SimpleNamespace(),
        }

    def test_build_demo_constructs_the_interface(self):
        import app

        demo = app.build_demo(self.stub_runtime())
        config = demo.get_config_file()
        labels = {c.get("props", {}).get("label") for c in config["components"]}
        values = {c.get("props", {}).get("value") for c in config["components"]}
        self.assertIn("Enable Auto-Rigging", labels)
        self.assertIn("Generate 3D", values)
        self.assertIn("Extract GLB", values)
        # Seed, imagen a 3D, extracción... más las actualizaciones de estado de los botones.
        self.assertGreaterEqual(len(config["dependencies"]), 5)


if __name__ == "__main__":
    unittest.main()
