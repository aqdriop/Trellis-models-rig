"""Pruebas del portal estático (portal/) y de netlify.toml."""

import hashlib
import json
import re
import unittest
import zipfile
from html.parser import HTMLParser

from support import PORTAL_DIR, ROOT, load_build_module

try:  # Python 3.11+; en 3.10 puede existir tomli
    import tomllib
except ImportError:  # pragma: no cover - depende del intérprete
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

build = load_build_module()


class PageParser(HTMLParser):
    """Recoge lo necesario para validar la página sin dependencias externas."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.attrs_by_id = {}
        self.references = []  # (etiqueta, atributo, valor)
        self.anchors = []  # atributos de cada <a>
        self.inline_scripts = 0
        self.style_attributes = 0
        self.event_handlers = []
        self.svgs = []
        self.h1_count = 0
        self.html_lang = None
        self.title = ""
        self.metas = {}
        self._in_title = False
        self._script_has_src = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
            self.attrs_by_id[attrs["id"]] = dict(attrs, _tag=tag)
        for name in ("href", "src"):
            if name in attrs and tag != "svg":
                self.references.append((tag, name, attrs[name]))
        if "style" in attrs:
            self.style_attributes += 1
        self.event_handlers += [name for name in attrs if name.startswith("on")]
        if tag == "a":
            self.anchors.append(attrs)
        elif tag == "html":
            self.html_lang = attrs.get("lang")
        elif tag == "h1":
            self.h1_count += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and "name" in attrs:
            self.metas[attrs["name"]] = attrs.get("content", "")
        elif tag == "script":
            self._script_has_src = "src" in attrs
            if not self._script_has_src:
                self.inline_scripts += 1
        elif tag == "svg":
            self.svgs.append(attrs)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def parse_index():
    parser = PageParser()
    parser.feed((PORTAL_DIR / "index.html").read_text(encoding="utf-8"))
    return parser


class PortalPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = parse_index()
        cls.manifest = json.loads(build.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_download_button_serves_the_published_zip(self):
        button = self.page.attrs_by_id["download-btn"]
        self.assertEqual(button["_tag"], "a")
        self.assertEqual(button["href"], "downloads/" + build.ZIP_NAME)
        self.assertIn("download", button)
        target = PORTAL_DIR / button["href"]
        self.assertTrue(target.is_file())
        data = target.read_bytes()
        self.assertTrue(zipfile.is_zipfile(target))
        self.assertEqual(hashlib.sha256(data).hexdigest(), self.manifest["sha256"])
        with zipfile.ZipFile(target) as archive:
            self.assertIn(build.BUNDLE_NAME + "/TRELLIS_AutoRig_Colab.ipynb", archive.namelist())

    def test_colab_and_release_links_match_the_manifest(self):
        self.assertEqual(self.page.attrs_by_id["colab-link"]["href"], self.manifest["colab_url"])
        self.assertEqual(self.page.attrs_by_id["release-link"]["href"], self.manifest["release_asset"])

    def test_local_references_exist(self):
        checked = 0
        for _tag, _attr, value in self.page.references:
            if re.match(r"^(https?:|mailto:|#|data:)", value):
                continue
            path = PORTAL_DIR / value.split("#", 1)[0].split("?", 1)[0]
            self.assertTrue(path.is_file(), "no existe %s" % value)
            checked += 1
        self.assertGreaterEqual(checked, 4)  # css, js, favicon y el ZIP

    def test_ids_are_unique_and_internal_references_resolve(self):
        self.assertEqual(len(self.page.ids), len(set(self.page.ids)))
        ids = set(self.page.ids)
        for _tag, attr, value in self.page.references:
            if attr == "href" and value.startswith("#") and len(value) > 1:
                self.assertIn(value[1:], ids, value)
        for element in self.page.attrs_by_id.values():
            for name in ("aria-labelledby", "aria-describedby"):
                for target in element.get(name, "").split():
                    self.assertIn(target, ids, "%s=%s" % (name, target))
        # El diagrama referencia también <title>/<desc> y el patrón del SVG.
        self.assertIn("pipe-title", ids)

    def test_page_is_compatible_with_the_strict_csp(self):
        self.assertEqual(self.page.inline_scripts, 0)
        self.assertEqual(self.page.style_attributes, 0)
        self.assertEqual(self.page.event_handlers, [])
        for tag, attr, value in self.page.references:
            if tag in ("script", "link"):
                self.assertFalse(re.match(r"^(https?:)?//", value), "recurso externo: %s" % value)

    def test_javascript_only_touches_elements_that_exist(self):
        script = (PORTAL_DIR / "app.js").read_text(encoding="utf-8")
        used = set(re.findall(r"(?:byId|setText|getElementById)\('([\w-]+)'", script))
        self.assertTrue(used)
        self.assertEqual(sorted(used - set(self.page.ids)), [])

    def test_has_the_basic_accessibility_and_seo_metadata(self):
        self.assertEqual(self.page.html_lang, "es")
        self.assertIn("TRELLIS", self.page.title)
        self.assertEqual(self.page.h1_count, 1)
        self.assertIn("width=device-width", self.page.metas["viewport"])
        self.assertGreater(len(self.page.metas["description"]), 50)
        self.assertTrue((PORTAL_DIR / "favicon.svg").is_file())
        for svg in self.page.svgs:
            self.assertTrue(
                svg.get("aria-hidden") == "true" or svg.get("role") == "img",
                "SVG sin nombre accesible ni aria-hidden: %s" % svg,
            )

    def test_links_that_open_a_new_tab_use_noopener(self):
        blank = [a for a in self.page.anchors if a.get("target") == "_blank"]
        self.assertTrue(blank)
        for anchor in blank:
            self.assertIn("noopener", anchor.get("rel", ""), anchor.get("href"))

    def test_page_names_the_repository_without_creating_another_one(self):
        repos = {
            re.match(r"https://github\.com/([\w.-]+/[\w.-]+)", a["href"]).group(1)
            for a in self.page.anchors
            if a.get("href", "").startswith("https://github.com/")
        }
        self.assertEqual(repos - {"microsoft/TRELLIS"}, {build.GITHUB_REPO})


@unittest.skipIf(tomllib is None, "requiere tomllib (Python 3.11+) o tomli")
class NetlifyConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tomllib.loads((ROOT / "netlify.toml").read_text(encoding="utf-8"))
        cls.headers = {rule["for"]: rule["values"] for rule in cls.config["headers"]}

    def test_publishes_the_portal_directory_without_a_build_step(self):
        build_section = self.config["build"]
        self.assertEqual(build_section["base"], "portal")
        self.assertEqual(build_section["publish"], ".")
        self.assertNotIn("command", build_section)
        self.assertTrue((ROOT / build_section["base"] / "index.html").is_file())

    def test_netlify_cannot_detect_dependencies_inside_the_base_directory(self):
        # Netlify instala dependencias si encuentra estos archivos en el directorio base.
        # El requirements.txt de la raíz es de la app Gradio y no debe usarse al desplegar.
        self.assertTrue((ROOT / "requirements.txt").is_file())
        for name in (
            "requirements.txt", "Pipfile", "runtime.txt", "package.json", "yarn.lock",
            "pnpm-lock.yaml", "bun.lockb", "bower.json", "composer.json", ".nvmrc",
            ".node-version", ".python-version", ".ruby-version", ".go-version", "go.mod",
        ):
            self.assertFalse((PORTAL_DIR / name).exists(), name)

    def test_applies_hardening_headers_to_every_page(self):
        values = self.headers["/*"]
        self.assertEqual(values["X-Content-Type-Options"], "nosniff")
        self.assertEqual(values["X-Frame-Options"], "DENY")
        csp = values["Content-Security-Policy"]
        for directive in ("default-src 'none'", "script-src 'self'", "style-src 'self'",
                          "connect-src 'self'", "frame-ancestors 'none'", "base-uri 'none'"):
            self.assertIn(directive, csp)
        self.assertNotIn("unsafe-inline", csp)

    def test_forces_the_zip_to_download(self):
        values = self.headers["/downloads/*.zip"]
        self.assertEqual(values["Content-Type"], "application/zip")
        self.assertTrue(values["Content-Disposition"].startswith("attachment"))


class PortalFilesTests(unittest.TestCase):
    def test_portal_has_no_stray_files(self):
        found = sorted(
            str(p.relative_to(PORTAL_DIR)) for p in PORTAL_DIR.rglob("*") if p.is_file()
        )
        self.assertEqual(
            found,
            [
                "app.js",
                "downloads/manifest.json",
                "downloads/" + build.ZIP_NAME,
                "favicon.svg",
                "index.html",
                "styles.css",
            ],
        )


if __name__ == "__main__":
    unittest.main()
