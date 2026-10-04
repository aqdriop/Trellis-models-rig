import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rigging import AutoRigError, auto_rig_glb


class AutoRigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.glb"
        self.destination = self.root / "rigged.glb"
        self.source.write_bytes(b"test glb")

    def test_reports_missing_blender(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("rigging.shutil.which", return_value=None):
                with self.assertRaisesRegex(AutoRigError, "requiere Blender"):
                    auto_rig_glb(self.source, self.destination)

    def test_rejects_missing_or_non_glb_input_before_starting_blender(self):
        with self.assertRaisesRegex(AutoRigError, "No existe"):
            auto_rig_glb(self.root / "missing.glb", self.destination)

        obj_source = self.root / "source.obj"
        obj_source.write_text("o mesh\n", encoding="utf-8")
        with self.assertRaisesRegex(AutoRigError, r"Se esperaba un archivo \.glb"):
            auto_rig_glb(obj_source, self.destination)

    def test_runs_blender_and_checks_the_created_file(self):
        def fake_run(command, **kwargs):
            output_arg = command[command.index("--output") + 1]
            Path(output_arg).write_bytes(b"rigged glb")
            return SimpleNamespace(returncode=0, stdout="Export complete", stderr="")

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("rigging.shutil.which", return_value="/fake/blender"):
                with mock.patch("rigging.subprocess.run", side_effect=fake_run) as run:
                    result = auto_rig_glb(self.source, self.destination)

        self.assertEqual(result, self.destination.resolve())
        self.assertEqual(result.read_bytes(), b"rigged glb")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/fake/blender")
        self.assertIn("--background", command)
        self.assertIn("--factory-startup", command)
        self.assertIn("--python", command)
        self.assertEqual(command[command.index("--input") + 1], str(self.source.resolve()))

    def test_removes_partial_output_and_reports_blender_failure(self):
        self.destination.write_bytes(b"stale output")

        def fake_run(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"partial")
            return SimpleNamespace(returncode=1, stdout="", stderr="weight generation failed")

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("rigging.shutil.which", return_value="/fake/blender"):
                with mock.patch("rigging.subprocess.run", side_effect=fake_run):
                    with self.assertRaisesRegex(AutoRigError, "weight generation failed"):
                        auto_rig_glb(self.source, self.destination)

        self.assertFalse(self.destination.exists())

    def test_timeout_removes_partial_output(self):
        import subprocess

        def fake_run(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"partial")
            raise subprocess.TimeoutExpired(command, timeout=3, stderr="still running")

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("rigging.shutil.which", return_value="/fake/blender"):
                with mock.patch("rigging.subprocess.run", side_effect=fake_run):
                    with self.assertRaisesRegex(AutoRigError, "límite de tiempo"):
                        auto_rig_glb(self.source, self.destination, timeout=3)

        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
