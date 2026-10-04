"""Pruebas del cuaderno de Colab: estructura y ejecución de sus celdas con un Colab simulado.

No usan red, GPU ni Blender: el repositorio remoto, `nvidia-smi`, `app.py` e IPython son falsos.
"""

import ast
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from support import NOTEBOOK, SETUP_SCRIPT, load_build_module

build = load_build_module()


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_of(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def cell_by_id(cell_id):
    for cell in load_notebook()["cells"]:
        if cell["id"] == cell_id:
            return cell
    raise KeyError(cell_id)


class NotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook()
        cls.cells = cls.notebook["cells"]

    def test_is_nbformat_4_with_unique_valid_cell_ids(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        ids = [cell["id"] for cell in self.cells]
        self.assertEqual(len(ids), len(set(ids)))
        for cell_id in ids:
            self.assertRegex(cell_id, r"^[A-Za-z0-9_-]{1,64}$")

    def test_code_cells_are_valid_python_and_have_no_stored_outputs(self):
        for cell in self.cells:
            if cell["cell_type"] != "code":
                continue
            ast.parse(source_of(cell), filename=cell["id"])
            self.assertEqual(cell["outputs"], [], cell["id"])
            self.assertIsNone(cell["execution_count"], cell["id"])

    def test_requests_a_gpu_runtime(self):
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")

    def test_pinned_version_matches_the_bundle_version(self):
        build.validate_notebook(NOTEBOOK.read_bytes())

    def test_open_in_colab_badge_points_to_the_tagged_notebook(self):
        self.assertIn(build.colab_url(), source_of(cell_by_id("intro")))

    def test_referenced_repo_files_exist(self):
        text = "\n".join(source_of(cell) for cell in self.cells if cell["cell_type"] == "markdown")
        paths = set(re.findall(r"`(colab/[\w./-]+)`", text))
        self.assertIn("colab/setup_colab.sh", paths)
        self.assertIn("colab/requirements-colab.lock", paths)
        for rel in paths:
            self.assertTrue((SETUP_SCRIPT.parents[1] / rel).is_file(), rel)
        self.assertIn(SETUP_SCRIPT.name, source_of(cell_by_id("instalar")))

    def test_numbered_titles_match_what_the_setup_script_tells_the_user(self):
        titles = [
            line
            for cell in self.cells
            if cell["cell_type"] == "code"
            for line in source_of(cell).splitlines()
            if line.startswith("#@title")
        ]
        self.assertEqual(len(titles), 3)
        self.assertTrue(titles[0].startswith("#@title 1 ·"))
        self.assertTrue(titles[1].startswith("#@title 2 ·"))
        self.assertTrue(titles[2].startswith("#@title 3 · Iniciar la aplicación"))
        self.assertIn("3 · Iniciar la aplicación", SETUP_SCRIPT.read_text(encoding="utf-8"))


FAKE_SETUP = """#!/usr/bin/env bash
set -euo pipefail
echo "instalacion falsa; work=${TRELLIS_COLAB_WORK}"
cat > "${TRELLIS_COLAB_WORK}/trellis_colab_env.sh" <<EOF
export PATH="__STUBS__:\\$PATH"
export FAKE_ENV_LOADED=yes
EOF
"""

FAKE_APP = """import os
print("Running on local URL:  http://0.0.0.0:7860", flush=True)
print("FAKE_ENV_LOADED=" + os.environ.get("FAKE_ENV_LOADED", ""), flush=True)
print("Running on public URL: https://abc123def.gradio.live", flush=True)
"""

WORK_LINE = 'WORK = Path("/content") if Path("/content").is_dir() else Path.cwd()'


@unittest.skipUnless(
    os.name == "posix" and shutil.which("git") and shutil.which("bash"),
    "requiere Linux/macOS con git y bash",
)
class NotebookExecutionTests(unittest.TestCase):
    """Ejecuta las celdas reales del cuaderno contra un Colab simulado."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.work = root / "work"
        self.work.mkdir()
        self.stubs = root / "stubs"
        self.stubs.mkdir()
        self.origin = root / "origin"
        self.make_stubs()
        self.make_origin()
        path = str(self.stubs) + os.pathsep + os.environ.get("PATH", "")
        patcher = mock.patch.dict(os.environ, {"PATH": path})
        patcher.start()
        self.addCleanup(patcher.stop)

    # ---- escenario -------------------------------------------------------------------
    def make_stubs(self):
        smi = self.stubs / "nvidia-smi"
        smi.write_text('#!/bin/sh\necho "GPU 0: Tesla T4 (UUID: GPU-fake)"\n', encoding="utf-8")
        smi.chmod(0o755)
        (self.stubs / "python").symlink_to(sys.executable)

    def make_origin(self):
        (self.origin / "colab").mkdir(parents=True)
        (self.origin / "colab" / "setup_colab.sh").write_text(
            FAKE_SETUP.replace("__STUBS__", str(self.stubs)), encoding="utf-8"
        )
        (self.origin / "app.py").write_text(FAKE_APP, encoding="utf-8")
        env = dict(
            os.environ,
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.com",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.com",
        )
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "inicial"], ["tag", "v9.9.9"]):
            subprocess.run(["git", *args], cwd=self.origin, check=True, capture_output=True, env=env)

    def cell_code(self, cell_id):
        code = source_of(cell_by_id(cell_id))
        if cell_id == "descargar-codigo":
            # La celda real usaría /content: aquí todo ocurre en una carpeta temporal.
            self.assertIn(WORK_LINE, code)
            code = code.replace(WORK_LINE, "WORK = Path(%r)" % str(self.work))
            code, n_url = re.subn(r'^REPO_URL = ".*?"', "REPO_URL = %r" % str(self.origin), code, flags=re.M)
            code, n_ref = re.subn(r'^REPO_REF = ".*?"', 'REPO_REF = "v9.9.9"', code, flags=re.M)
            self.assertEqual((n_url, n_ref), (1, 1))
        return code

    def run_cell(self, cell_id, namespace, stdout=None):
        compiled = compile(self.cell_code(cell_id), "<celda %s>" % cell_id, "exec")
        with contextlib.redirect_stdout(stdout or io.StringIO()):
            exec(compiled, namespace)

    def prepared_namespace(self):
        namespace = {"__name__": "__main__"}
        self.run_cell("descargar-codigo", namespace)
        return namespace

    # ---- pruebas ----------------------------------------------------------------------
    def test_download_install_and_launch_cells_run_in_order(self):
        namespace = {"__name__": "__main__"}
        out = io.StringIO()

        self.run_cell("descargar-codigo", namespace, out)
        repo_dir = namespace["REPO_DIR"]
        self.assertTrue((repo_dir / "app.py").is_file())
        self.assertIn("Código listo:", out.getvalue())

        self.run_cell("instalar", namespace, out)
        self.assertTrue(namespace["ENV_FILE"].is_file())
        self.assertIn("instalacion falsa", out.getvalue())
        self.assertIn("Instalación terminada en", out.getvalue())

        displayed = []
        fake_display = types.ModuleType("IPython.display")
        fake_display.HTML = lambda html: ("HTML", html)
        fake_display.display = displayed.append
        fake_ipython = types.ModuleType("IPython")
        fake_ipython.display = fake_display
        with mock.patch.dict(sys.modules, {"IPython": fake_ipython, "IPython.display": fake_display}):
            self.run_cell("iniciar-app", namespace, out)

        self.assertIn("FAKE_ENV_LOADED=yes", out.getvalue())
        self.assertIn("Running on public URL", out.getvalue())
        self.assertEqual(len(displayed), 1)
        self.assertIn("https://abc123def.gradio.live", displayed[0][1])

    def test_download_cell_requires_a_gpu(self):
        namespace = {"__name__": "__main__"}
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No hay GPU"):
                self.run_cell("descargar-codigo", namespace)

    def test_rerunning_the_download_cell_resets_the_checkout(self):
        namespace = self.prepared_namespace()
        edited = namespace["REPO_DIR"] / "app.py"
        edited.write_text("# editado por el usuario\n", encoding="utf-8")
        self.run_cell("descargar-codigo", namespace)
        self.assertNotIn("editado por el usuario", edited.read_text(encoding="utf-8"))

    def test_launch_cell_requires_the_install_step(self):
        namespace = self.prepared_namespace()
        fake_display = types.ModuleType("IPython.display")
        fake_display.HTML = str
        fake_display.display = print
        fake_ipython = types.ModuleType("IPython")
        fake_ipython.display = fake_display
        with mock.patch.dict(sys.modules, {"IPython": fake_ipython, "IPython.display": fake_display}):
            with self.assertRaisesRegex(RuntimeError, "Falta el entorno"):
                self.run_cell("iniciar-app", namespace)

    def test_run_streams_output_and_reports_the_exit_code(self):
        namespace = self.prepared_namespace()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaisesRegex(RuntimeError, "código 3"):
                namespace["run"](["bash", "-c", "echo salida; echo error >&2; exit 3"])
        self.assertIn("salida", out.getvalue())
        self.assertIn("error", out.getvalue())

    def test_run_keeps_carriage_returns_for_progress_bars(self):
        namespace = self.prepared_namespace()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            namespace["run"](["bash", "-c", "printf '10%%\\r50%%\\r100%%\\n'"])
        self.assertEqual(out.getvalue(), "10%\r50%\r100%\n")

    def test_interrupt_stops_the_whole_process_group(self):
        namespace = self.prepared_namespace()
        pids = []

        def interrupt_after_first_output(text):
            pids.extend(int(n) for n in re.findall(r"^(\d+)$", text, re.M))
            raise KeyboardInterrupt

        started = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                namespace["run"](
                    ["bash", "-c", "echo $$; sleep 60 & wait"],
                    on_output=interrupt_after_first_output,
                )
        self.assertLess(time.time() - started, 15)
        self.assertEqual(len(pids), 1)
        time.sleep(0.2)
        with self.assertRaises(ProcessLookupError):
            os.kill(pids[0], 0)


if __name__ == "__main__":
    unittest.main()
