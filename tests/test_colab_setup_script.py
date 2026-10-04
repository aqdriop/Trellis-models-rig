"""Pruebas de colab/setup_colab.sh y del lockfile, sin GPU, sin red y sin instalar nada real."""

import ast
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from support import LOCK_INPUT, LOCKFILE, ROOT, SETUP_SCRIPT

BASH = shutil.which("bash")

PHASES = (
    "phase_apt phase_cuda phase_python phase_torch phase_xformers phase_lock "
    "phase_rasterizer phase_nvdiffrast phase_trellis phase_blender phase_verify"
).split()

EXPECTED_STEP_ORDER = [
    "Paquetes del sistema",
    "Toolkit CUDA 12.1",
    "Python 3.10 aislado",
    "PyTorch 2.4.0",
    "xformers",
    "Dependencias de Python",
    "Compilar diff-gaussian-rasterization",
    "Instalar nvdiffrast",
    "Descargar TRELLIS",
    "Descargar Blender",
    "Verificar la instalación",
]


@unittest.skipUnless(BASH, "requiere bash")
class SetupScriptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.stubs = self.root / "stubs"
        self.stubs.mkdir()

    def bash(self, body, path=None, **extra_env):
        env = dict(os.environ, TRELLIS_COLAB_WORK=str(self.work))
        env["PATH"] = path if path is not None else str(self.stubs) + os.pathsep + env.get("PATH", "")
        env.update(extra_env)
        script = 'source "%s"\n%s' % (SETUP_SCRIPT, body)
        return subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=env, timeout=60)

    def fake_nvidia_smi(self, line):
        stub = self.stubs / "nvidia-smi"
        stub.write_text('#!/bin/sh\necho "%s"\n' % line, encoding="utf-8")
        stub.chmod(0o755)

    # ---- sintaxis y versiones ---------------------------------------------------------
    def test_has_valid_bash_syntax(self):
        result = subprocess.run([BASH, "-n", str(SETUP_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck no está instalado")
    def test_passes_shellcheck(self):
        result = subprocess.run(["shellcheck", "-x", str(SETUP_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_embedded_verification_script_is_valid_python(self):
        # phase_verify lleva un programa Python en un heredoc: una errata solo se vería en Colab.
        text = SETUP_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r"<<'PY'\n(.*?)\nPY\n", text, re.S)
        self.assertIsNotNone(match)
        tree = ast.parse(match.group(1), filename="phase_verify")
        calls = [n.args[0].value for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "check"
                 and n.args and isinstance(n.args[0], ast.Constant)]
        # Lo imprescindible para app.py: CUDA, el pipeline de TRELLIS y el plugin de nvdiffrast.
        self.assertIn("PyTorch + CUDA", calls)
        self.assertIn("trellis.pipelines", calls)
        self.assertIn("trellis.utils", calls)
        self.assertIn("nvdiffrast: contexto CUDA", calls)

    def test_is_executable_so_it_can_be_run_directly(self):
        self.assertTrue(os.access(SETUP_SCRIPT, os.X_OK))

    def test_pinned_versions_are_well_formed(self):
        names = "TRELLIS_COMMIT MIPSPLAT_COMMIT BLENDER_SHA256 BLENDER_VERSION BLENDER_URL CUDA_RUNFILE_URL TORCH_SPEC TORCHVISION_SPEC TORCH_INDEX XFORMERS_SPEC NVDIFFRAST_REF"
        result = self.bash("for v in %s; do printf '%%s=%%s\\n' \"$v\" \"${!v}\"; done" % names)
        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertRegex(values["TRELLIS_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertRegex(values["MIPSPLAT_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertRegex(values["BLENDER_SHA256"], r"^[0-9a-f]{64}$")
        self.assertIn(values["BLENDER_VERSION"], values["BLENDER_URL"])
        self.assertTrue(values["BLENDER_URL"].startswith("https://download.blender.org/"))
        self.assertIn("12.1.1", values["CUDA_RUNFILE_URL"])
        self.assertEqual(values["TORCH_SPEC"], "torch==2.4.0")
        self.assertEqual(values["TORCHVISION_SPEC"], "torchvision==0.19.0")
        self.assertTrue(values["TORCH_INDEX"].endswith("/cu121"))
        self.assertEqual(values["XFORMERS_SPEC"], "xformers==0.0.27.post2")
        self.assertRegex(values["NVDIFFRAST_REF"], r"^v\d+\.\d+\.\d+$")

    # ---- GPU ---------------------------------------------------------------------------
    def test_picks_the_compile_architecture_from_the_gpu(self):
        cases = {
            "Tesla T4, 15360, 7.5": "7.5",
            "NVIDIA L4, 23034, 8.9": "8.9",
            "NVIDIA A100-SXM4-40GB, 40960, 8.0": "8.0",
            "NVIDIA H100 80GB HBM3, 81559, 9.0": "9.0",
            "Tesla T4, 15360, [N/A]": "7.5;8.0;8.6;8.9;9.0",
        }
        for line, arch in cases.items():
            with self.subTest(gpu=line):
                result = self.bash('parse_gpu "%s"; check_gpu; echo "ARCH=$ARCH"' % line)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ARCH=%s" % arch, result.stdout)

    def test_parses_gpu_fields(self):
        result = self.bash(
            'parse_gpu "NVIDIA RTX A6000 , 49140 , 8.6"; echo "[$GPU_NAME][$GPU_MEM_MIB][$GPU_CC]"'
        )
        self.assertEqual(result.stdout.strip(), "[NVIDIA RTX A6000][49140][8.6]")

    def test_rejects_blackwell_gpus_with_an_actionable_message(self):
        result = self.bash(
            'parse_gpu "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0"; check_gpu; echo "NO LLEGA"'
        )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("NO LLEGA", result.stdout)
        self.assertIn("Blackwell", result.stderr)
        self.assertIn("T4", result.stderr)

    def test_warns_when_gpu_memory_is_tight_or_insufficient(self):
        tight = self.bash('parse_gpu "Tesla T4, 15360, 7.5"; check_gpu')
        self.assertIn("al límite", tight.stderr)
        small = self.bash('parse_gpu "Small GPU, 8192, 7.5"; check_gpu')
        self.assertIn("falte memoria", small.stderr)
        roomy = self.bash('parse_gpu "NVIDIA L4, 23034, 8.9"; check_gpu')
        self.assertEqual(roomy.stderr, "")

    def test_preflight_fails_without_a_gpu(self):
        bin_dir = self.root / "minimal-bin"
        bin_dir.mkdir()
        for tool in ("uname", "id", "python3", "df", "awk", "head", "mkdir", "dirname", "tr", "sed", "cat"):
            found = shutil.which(tool)
            if not found:
                self.skipTest("falta la herramienta %s" % tool)
            (bin_dir / tool).symlink_to(found)
        result = self.bash("preflight", path=str(bin_dir))
        self.assertEqual(result.returncode, 1)
        self.assertIn("No hay GPU", result.stderr)

    # ---- entorno y pasos -----------------------------------------------------------------
    def test_env_file_exports_what_the_app_and_the_compilers_need(self):
        self.bash('ARCH="7.5"; write_env_file')
        env_file = self.work / "trellis_colab_env.sh"
        self.assertTrue(env_file.is_file())
        names = "CUDA_HOME TORCH_CUDA_ARCH_LIST CC CXX ATTN_BACKEND SPCONV_ALGO GRADIO_SHARE TRELLIS_ROOT BLENDER_BIN VIRTUAL_ENV"
        result = subprocess.run(
            [BASH, "-c", 'source "%s"; for v in %s; do printf "%%s=%%s\\n" "$v" "${!v}"; done; echo "PATH=$PATH"' % (env_file, names)],
            capture_output=True,
            text=True,
            env=dict(os.environ, PATH="/usr/bin:/bin"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(values["TORCH_CUDA_ARCH_LIST"], "7.5")
        self.assertEqual(values["CUDA_HOME"], "/opt/cuda-12.1")
        self.assertEqual((values["CC"], values["CXX"]), ("gcc-12", "g++-12"))
        self.assertEqual(values["ATTN_BACKEND"], "xformers")
        self.assertEqual(values["SPCONV_ALGO"], "native")
        self.assertEqual(values["GRADIO_SHARE"], "True")
        self.assertEqual(values["TRELLIS_ROOT"], str(self.work / "TRELLIS"))
        self.assertEqual(values["BLENDER_BIN"], str(self.work / "blender" / "blender"))
        self.assertEqual(values["VIRTUAL_ENV"], str(self.work / "trellis-venv"))
        self.assertTrue(values["PATH"].startswith(str(self.work / "trellis-venv" / "bin") + ":/opt/cuda-12.1/bin:"))

    def test_step_reports_success_and_shows_the_log_on_request(self):
        result = self.bash('ok() { echo "hola desde la fase"; }; step "Fase OK" ok; step --show "Fase con salida" ok')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Fase OK", result.stdout)
        self.assertEqual(result.stdout.count("hola desde la fase"), 1)

    def test_step_prints_a_heartbeat_during_long_phases(self):
        result = self.bash(
            'slow() { sleep 3.2; }; step "Fase lenta" slow',
            TRELLIS_COLAB_HEARTBEAT="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        beats = [line for line in result.stdout.splitlines() if line.strip().startswith("…")]
        self.assertGreaterEqual(len(beats), 2, result.stdout)

    def test_step_aborts_and_prints_the_log_tail_on_failure(self):
        result = self.bash('bad() { echo "linea1"; echo "boom" >&2; return 3; }; step "Fase mala" bad; echo "NO LLEGA"')
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("NO LLEGA", result.stdout)
        self.assertIn("Falló: Fase mala", result.stderr)
        self.assertIn("boom", result.stderr)
        self.assertTrue((self.work / "setup_logs" / "Fase_mala.log").is_file())

    def stub_every_phase(self):
        lines = ["%s() { touch \"%s/ran_%s\"; }" % (phase, self.work, phase) for phase in PHASES]
        return "\n".join(lines)

    def test_main_runs_every_phase_in_order(self):
        self.fake_nvidia_smi("Tesla T4, 15360, 7.5")
        result = self.bash(self.stub_every_phase() + "\nmain")
        self.assertEqual(result.returncode, 0, result.stderr)
        titles = [line[2:] for line in result.stdout.splitlines() if line.startswith("▶ ")]
        self.assertEqual(len(titles), len(EXPECTED_STEP_ORDER), titles)
        for title, expected in zip(titles, EXPECTED_STEP_ORDER):
            self.assertTrue(title.startswith(expected), "%r no empieza por %r" % (title, expected))
        for phase in PHASES:
            self.assertTrue((self.work / ("ran_" + phase)).exists(), phase)
        self.assertIn("Instalación completada", result.stdout)
        self.assertTrue((self.work / "trellis_colab_env.sh").is_file())

    def test_main_installs_nothing_on_an_unsupported_gpu(self):
        self.fake_nvidia_smi("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0")
        result = self.bash(self.stub_every_phase() + "\nmain")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Blackwell", result.stderr)
        self.assertEqual(list(self.work.glob("ran_*")), [])

    def test_help_prints_the_header_comment(self):
        result = subprocess.run([BASH, str(SETUP_SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("setup_colab.sh", result.stdout)
        self.assertIn("TRELLIS_COLAB_WORK", result.stdout)


NAME_PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.!+\-]+)$")
UTILS3D_COMMIT = "9a4eb15e4021b67b12c460c7057d642626897ec8"


def lock_entries():
    entries = {}
    for line in LOCKFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = NAME_PIN.match(line)
        if match:
            entries[match.group(1).lower().replace("_", "-")] = match.group(2)
        else:
            entries[line.split(" @ ")[0].strip().lower()] = line
    return entries


class LockfileTests(unittest.TestCase):
    def test_every_line_is_an_exact_pin_or_the_pinned_utils3d_commit(self):
        for line in LOCKFILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("utils3d @ "):
                self.assertTrue(line.endswith("@" + UTILS3D_COMMIT), line)
            else:
                self.assertRegex(line, NAME_PIN, line)

    def test_excludes_what_setup_installs_separately(self):
        entries = lock_entries()
        for name in ("torch", "torchvision", "xformers", "nvdiffrast", "diff-gaussian-rasterization", "triton"):
            self.assertNotIn(name, entries)
        self.assertFalse([n for n in entries if n.startswith("nvidia-")])

    def test_matches_the_constraints_trellis_and_the_app_need(self):
        entries = lock_entries()
        self.assertTrue(entries["numpy"].startswith("1."), "TRELLIS/requirements.txt exigen numpy < 2")
        gradio_major = int(entries["gradio"].split(".")[0])
        self.assertIn(gradio_major, (4, 5), "requirements.txt acota gradio a <6")
        self.assertLess(int(entries["setuptools"].split(".")[0]), 76)
        self.assertIn("spconv-cu120", entries)
        self.assertIn("utils3d", entries)

    def test_every_direct_requirement_is_locked(self):
        names = set()
        for line in LOCK_INPUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(re.split(r"[ <>=!~@\[]", line, maxsplit=1)[0].lower().replace("_", "-"))
        self.assertTrue(names)
        self.assertEqual(sorted(n for n in names if n not in lock_entries()), [])

    def test_input_repeats_the_requirements_txt_of_the_app(self):
        colab_input = LOCK_INPUT.read_text(encoding="utf-8").splitlines()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self.assertIn(line, colab_input, "requirements-colab.in debe incluir %r" % line)


if __name__ == "__main__":
    unittest.main()
