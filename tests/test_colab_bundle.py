"""Pruebas del ZIP del cuaderno: contenido, reproducibilidad, manifiesto y detección de desfases."""

import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from support import COLAB_DIR, load_build_module

build = load_build_module()
NAME = build.BUNDLE_NAME


class PublishedBundleTests(unittest.TestCase):
    def test_published_zip_and_manifest_are_up_to_date(self):
        self.assertEqual(build.problems(), [], "ejecuta: python scripts/build_colab_zip.py")

    def test_zip_is_valid_safe_and_has_a_single_top_level_folder(self):
        with zipfile.ZipFile(build.ZIP_PATH) as archive:
            self.assertIsNone(archive.testzip())
            names = archive.namelist()
        self.assertEqual(
            sorted(names),
            sorted([NAME + "/", NAME + "/LEEME.md", NAME + "/TRELLIS_AutoRig_Colab.ipynb"]),
        )
        for name in names:  # protección frente a «zip slip»
            self.assertFalse(name.startswith("/") or ".." in Path(name).parts, name)

    def test_zip_has_the_notebook_and_the_guide_verbatim(self):
        with zipfile.ZipFile(build.ZIP_PATH) as archive:
            self.assertEqual(
                archive.read(NAME + "/TRELLIS_AutoRig_Colab.ipynb"),
                (COLAB_DIR / "TRELLIS_AutoRig_Colab.ipynb").read_bytes(),
            )
            self.assertEqual(archive.read(NAME + "/LEEME.md"), (COLAB_DIR / "LEEME.md").read_bytes())

    def test_entries_have_fixed_dates_and_portable_permissions(self):
        with zipfile.ZipFile(build.ZIP_PATH) as archive:
            for info in archive.infolist():
                self.assertEqual(info.date_time, build.BUNDLE_DATE + (0, 0, 0), info.filename)
                mode = info.external_attr >> 16
                expected = 0o40755 if info.is_dir() else 0o100644
                self.assertEqual(mode, expected, info.filename)

    def test_manifest_describes_the_published_zip(self):
        manifest = json.loads(build.MANIFEST_PATH.read_text(encoding="utf-8"))
        data = build.ZIP_PATH.read_bytes()
        self.assertEqual(manifest["file"], build.ZIP_NAME)
        self.assertEqual(manifest["bytes"], len(data))
        self.assertEqual(manifest["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(manifest["version"], build.BUNDLE_VERSION)
        self.assertEqual(manifest["tag"], "v" + build.BUNDLE_VERSION)
        self.assertEqual(manifest["colab_url"], build.colab_url())
        self.assertTrue(manifest["colab_url"].startswith("https://colab.research.google.com/github/"))
        self.assertEqual(manifest["download_url"], build.download_url())
        self.assertTrue(
            manifest["download_url"].endswith(
                "/raw/v%s/portal/downloads/%s" % (build.BUNDLE_VERSION, build.ZIP_NAME)
            )
        )
        self.assertEqual(manifest["release_url"], build.release_url())
        self.assertNotIn("release_asset", manifest)  # el ZIP no es un archivo adjunto de la release
        paths = [entry["path"] for entry in manifest["contents"]]
        self.assertEqual(paths[0], NAME + "/TRELLIS_AutoRig_Colab.ipynb")

    def test_version_is_semver_and_date_is_plausible(self):
        self.assertRegex(build.BUNDLE_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertRegex(build.bundle_date(), r"^\d{4}-\d{2}-\d{2}$")


class BuildBehaviourTests(unittest.TestCase):
    def test_build_is_reproducible(self):
        self.assertEqual(build.build_zip_bytes(), build.build_zip_bytes())

    def test_build_ignores_input_order(self):
        files = build.read_sources()
        self.assertEqual(build.build_zip_bytes(files), build.build_zip_bytes(list(reversed(files))))

    def with_paths(self, directory):
        directory = Path(directory)
        return mock.patch.multiple(
            build,
            DOWNLOADS_DIR=directory,
            ZIP_PATH=directory / build.ZIP_NAME,
            MANIFEST_PATH=directory / "manifest.json",
        )

    def test_build_then_check_round_trip_in_a_scratch_directory(self):
        with tempfile.TemporaryDirectory() as scratch, self.with_paths(scratch):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(build.main([]), 0)
                self.assertEqual(build.main(["--check"]), 0)
            self.assertEqual(build.problems(), [])
            self.assertEqual(
                (Path(scratch) / build.ZIP_NAME).read_bytes(), build.build_zip_bytes()
            )

    def test_check_detects_a_stale_zip(self):
        with tempfile.TemporaryDirectory() as scratch, self.with_paths(scratch):
            stale = build.build_zip_bytes(
                [("TRELLIS_AutoRig_Colab.ipynb", b"{}"), ("LEEME.md", b"viejo")]
            )
            build.ZIP_PATH.write_bytes(stale)
            build.MANIFEST_PATH.write_text(build.render_manifest(build.manifest_for(stale)), encoding="utf-8")
            found = build.problems()
            self.assertTrue(any("no coincide con la fuente" in line for line in found), found)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(build.main(["--check"]), 1)
            self.assertIn("no está al día", stderr.getvalue())

    def test_check_detects_a_manifest_that_does_not_match_the_zip(self):
        with tempfile.TemporaryDirectory() as scratch, self.with_paths(scratch):
            data = build.build_zip_bytes()
            build.ZIP_PATH.write_bytes(data)
            manifest = build.manifest_for(data)
            manifest["sha256"] = "0" * 64
            build.MANIFEST_PATH.write_text(build.render_manifest(manifest), encoding="utf-8")
            found = build.problems()
            self.assertTrue(any("sha256" in line for line in found), found)

    def test_check_reports_a_missing_zip(self):
        with tempfile.TemporaryDirectory() as scratch, self.with_paths(scratch):
            found = build.problems()
            self.assertTrue(any("No existe" in line for line in found), found)

    def test_notebook_pinned_to_another_version_is_rejected(self):
        with mock.patch.object(build, "BUNDLE_VERSION", "9.9.9"):
            with self.assertRaisesRegex(ValueError, "REPO_REF"):
                build.validate_notebook((COLAB_DIR / "TRELLIS_AutoRig_Colab.ipynb").read_bytes())

    def test_invalid_notebooks_are_rejected(self):
        for bad, message in (
            (b"no es json", "JSON"),
            (json.dumps({"nbformat": 3, "cells": [1]}).encode(), "nbformat 4"),
            (json.dumps({"nbformat": 4, "cells": []}).encode(), "celdas"),
            (json.dumps({"nbformat": 4, "cells": [{"cell_type": "code", "source": ["x = 1"]}]}).encode(), "REPO_REF"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build.validate_notebook(bad)

    def test_manifest_json_is_stable_text(self):
        text = build.render_manifest({"b": 1, "a": "á"})
        self.assertEqual(text, '{\n  "a": "á",\n  "b": 1\n}\n')
        self.assertTrue(re.search(r"\n$", text))


if __name__ == "__main__":
    unittest.main()
