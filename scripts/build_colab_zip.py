#!/usr/bin/env python3
"""Empaqueta el cuaderno de Colab en un ZIP reproducible para el portal y las releases.

    python scripts/build_colab_zip.py           # genera el ZIP y su manifiesto
    python scripts/build_colab_zip.py --check   # comprueba que lo versionado coincide con las fuentes

Solo usa la biblioteca estándar. El ZIP es reproducible: fechas, permisos y orden de los
archivos son constantes, así que reconstruirlo sin cambiar las fuentes no genera diferencias.

Al cambiar el cuaderno o LEEME.md: sube BUNDLE_VERSION (y REPO_REF en el cuaderno, que debe ser
``v<BUNDLE_VERSION>``), ejecuta este script, versiona el resultado y publica una release nueva.
"""

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Versión del paquete -------------------------------------------------------------------
BUNDLE_VERSION = "0.1.0"
BUNDLE_DATE = (2026, 10, 4)  # año, mes, día: también fecha fija de las entradas del ZIP

# --- Nombres y rutas -----------------------------------------------------------------------
BUNDLE_NAME = "trellis-autorig-colab"
ZIP_NAME = BUNDLE_NAME + ".zip"
DOWNLOADS_DIR = REPO_ROOT / "portal" / "downloads"
ZIP_PATH = DOWNLOADS_DIR / ZIP_NAME
MANIFEST_PATH = DOWNLOADS_DIR / "manifest.json"

GITHUB_REPO = "aqdriop/Trellis-models-rig"
NOTEBOOK_REL = "colab/TRELLIS_AutoRig_Colab.ipynb"

# (ruta en el repositorio, nombre dentro del ZIP)
SOURCES = (
    (NOTEBOOK_REL, "TRELLIS_AutoRig_Colab.ipynb"),
    ("colab/LEEME.md", "LEEME.md"),
)


def _shown(path: Path) -> str:
    """Ruta relativa al repositorio si es posible (para mensajes legibles)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def version_tag() -> str:
    return "v" + BUNDLE_VERSION


def bundle_date() -> str:
    return "%04d-%02d-%02d" % BUNDLE_DATE


def download_url() -> str:
    """Descarga directa del ZIP versionado en el repositorio, fijada a la etiqueta de la versión."""
    return "https://github.com/%s/raw/%s/portal/downloads/%s" % (GITHUB_REPO, version_tag(), ZIP_NAME)


def release_url() -> str:
    return "https://github.com/%s/releases/tag/%s" % (GITHUB_REPO, version_tag())


def colab_url() -> str:
    return "https://colab.research.google.com/github/%s/blob/%s/%s" % (
        GITHUB_REPO,
        version_tag(),
        NOTEBOOK_REL,
    )


def read_sources() -> List[Tuple[str, bytes]]:
    files = []
    for rel, arcname in SOURCES:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("Falta el archivo fuente: " + rel)
        files.append((arcname, path.read_bytes()))
    return sorted(files)


def validate_notebook(data: bytes) -> None:
    """Falla (ValueError) si el cuaderno no es utilizable o no corresponde a la versión."""
    try:
        notebook = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("El cuaderno no es JSON válido: %s" % exc)
    if notebook.get("nbformat") != 4:
        raise ValueError("El cuaderno debe usar nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("El cuaderno no tiene celdas")
    code = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        code.append("".join(source) if isinstance(source, list) else source)
    match = re.search(r'^REPO_REF\s*=\s*"([^"]+)"', "\n".join(code), re.MULTILINE)
    if not match:
        raise ValueError("No se encontró la asignación REPO_REF en el cuaderno")
    if match.group(1) != version_tag():
        raise ValueError(
            'REPO_REF del cuaderno es "%s" pero la versión del paquete es "%s"'
            % (match.group(1), version_tag())
        )


def build_zip_bytes(files: Optional[List[Tuple[str, bytes]]] = None) -> bytes:
    """ZIP reproducible: una carpeta raíz, entradas ordenadas y metadatos constantes."""
    if files is None:
        files = read_sources()
    buffer = io.BytesIO()
    stamp = BUNDLE_DATE + (0, 0, 0)
    with zipfile.ZipFile(buffer, "w") as archive:
        folder = zipfile.ZipInfo(BUNDLE_NAME + "/", date_time=stamp)
        folder.create_system = 3
        folder.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(folder, b"")
        for arcname, data in sorted(files):
            info = zipfile.ZipInfo(BUNDLE_NAME + "/" + arcname, date_time=stamp)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    return buffer.getvalue()


def _members(zip_bytes: bytes) -> List[Tuple[str, int]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return [(i.filename, i.file_size) for i in archive.infolist() if not i.is_dir()]


def _contents_in_source_order(zip_bytes: bytes) -> List[Dict[str, object]]:
    """Contenido del ZIP en el orden de SOURCES (el cuaderno primero), no el alfabético."""
    sizes = dict(_members(zip_bytes))
    ordered = []
    for _rel, arcname in SOURCES:
        path = BUNDLE_NAME + "/" + arcname
        if path in sizes:
            ordered.append({"path": path, "bytes": sizes.pop(path)})
    ordered.extend({"path": path, "bytes": size} for path, size in sorted(sizes.items()))
    return ordered


def manifest_for(zip_bytes: bytes) -> Dict[str, object]:
    return {
        "name": BUNDLE_NAME,
        "file": ZIP_NAME,
        "version": BUNDLE_VERSION,
        "tag": version_tag(),
        "updated": bundle_date(),
        "bytes": len(zip_bytes),
        "sha256": hashlib.sha256(zip_bytes).hexdigest(),
        "contents": _contents_in_source_order(zip_bytes),
        "colab_url": colab_url(),
        "github": "https://github.com/" + GITHUB_REPO,
        "download_url": download_url(),
        "release_url": release_url(),
    }


def render_manifest(manifest: Dict[str, object]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def build() -> int:
    files = read_sources()
    for arcname, data in files:
        if arcname.endswith(".ipynb"):
            validate_notebook(data)
    zip_bytes = build_zip_bytes(files)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_PATH.write_bytes(zip_bytes)
    manifest = manifest_for(zip_bytes)
    MANIFEST_PATH.write_text(render_manifest(manifest), encoding="utf-8")
    print("ZIP:       %s (%d bytes)" % (_shown(ZIP_PATH), len(zip_bytes)))
    print("SHA-256:   %s" % manifest["sha256"])
    print("Manifiesto: %s" % _shown(MANIFEST_PATH))
    return 0


def problems() -> List[str]:
    """Diferencias entre el ZIP/manifiesto versionados y las fuentes (lista vacía = todo bien)."""
    found = []  # type: List[str]
    try:
        files = read_sources()
    except FileNotFoundError as exc:
        return [str(exc)]
    for arcname, data in files:
        if arcname.endswith(".ipynb"):
            try:
                validate_notebook(data)
            except ValueError as exc:
                found.append(str(exc))

    if not ZIP_PATH.is_file():
        return found + ["No existe %s: ejecuta scripts/build_colab_zip.py" % _shown(ZIP_PATH)]
    zip_bytes = ZIP_PATH.read_bytes()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            bad = archive.testzip()
            if bad:
                found.append("Entrada corrupta en el ZIP: " + bad)
            expected = {BUNDLE_NAME + "/" + arcname: data for arcname, data in files}
            actual = {i.filename: archive.read(i) for i in archive.infolist() if not i.is_dir()}
            for name in sorted(set(expected) - set(actual)):
                found.append("Falta en el ZIP: " + name)
            for name in sorted(set(actual) - set(expected)):
                found.append("Sobra en el ZIP: " + name)
            for name in sorted(set(expected) & set(actual)):
                if expected[name] != actual[name]:
                    found.append("El contenido de %s no coincide con la fuente" % name)
    except zipfile.BadZipFile as exc:
        return found + ["El ZIP no es válido: %s" % exc]

    if not MANIFEST_PATH.is_file():
        return found + ["No existe " + _shown(MANIFEST_PATH)]
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except ValueError as exc:
        return found + ["manifest.json no es JSON válido: %s" % exc]
    wanted = manifest_for(zip_bytes)  # tamaño y hash salen del ZIP versionado
    for key, value in wanted.items():
        if manifest.get(key) != value:
            found.append("manifest.json: «%s» es %r y debería ser %r" % (key, manifest.get(key), value))
    return found


def check() -> int:
    found = problems()
    if found:
        print("El paquete de Colab no está al día:", file=sys.stderr)
        for line in found:
            print("  - " + line, file=sys.stderr)
        print("Ejecuta: python scripts/build_colab_zip.py", file=sys.stderr)
        return 1
    print("OK: %s y manifest.json coinciden con las fuentes." % _shown(ZIP_PATH))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="no escribe nada: comprueba que el ZIP y el manifiesto versionados están al día",
    )
    args = parser.parse_args(argv)
    try:
        return check() if args.check else build()
    except (FileNotFoundError, ValueError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
