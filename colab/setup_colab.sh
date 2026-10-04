#!/usr/bin/env bash
# setup_colab.sh — instala TRELLIS + Blender en Google Colab (Linux x86_64 con GPU NVIDIA).
#
# Por qué un entorno aislado: el runtime actual de Colab trae Python 3.13, CUDA 13,
# numpy 2 y Gradio 6, mientras que TRELLIS está pensado para Python 3.10, PyTorch 2.4 y
# CUDA 11.8/12.x. En lugar de modificar el entorno de Colab se crea uno propio:
#   uv + Python 3.10 + torch 2.4.0 (CUDA 12.1) + toolkit CUDA 12.1 para compilar.
#
# Es idempotente: cada fase se omite si ya está hecha. Los registros completos de cada
# fase se guardan en $WORK/setup_logs/ y las últimas líneas se muestran si algo falla.
#
# Variables opcionales:
#   TRELLIS_COLAB_WORK       carpeta de trabajo (por defecto /content)
#   TRELLIS_COLAB_CUDA_HOME  destino del toolkit CUDA 12.1 (por defecto /opt/cuda-12.1)
#   MAX_JOBS                 compilaciones paralelas (por defecto 4)
#   TRELLIS_COLAB_HEARTBEAT  segundos entre mensajes de "sigue en curso" (por defecto 30)

set -Eeuo pipefail

# --------------------------------------------------------------------------- versiones
# Fijadas por reproducibilidad (los proyectos upstream no publican releases estables).
TRELLIS_COMMIT="442aa1e1afb9014e80681d3bf604e8d728a86ee7"    # microsoft/TRELLIS, 2025-11-05
MIPSPLAT_COMMIT="dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"   # autonomousvision/mip-splatting
NVDIFFRAST_REF="v0.3.3"                                       # NVlabs/nvdiffrast
TORCH_SPEC="torch==2.4.0"
TORCHVISION_SPEC="torchvision==0.19.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
XFORMERS_SPEC="xformers==0.0.27.post2"                        # compilado para torch 2.4.0
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"
BLENDER_VERSION="4.5.14"                                      # Blender 4.5 LTS
BLENDER_SHA256="9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"
BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

# --------------------------------------------------------------------------- rutas
WORK="${TRELLIS_COLAB_WORK:-/content}"
SRC="$WORK/src"
LOGS="$WORK/setup_logs"
VENV="$WORK/trellis-venv"
CUDA_HOME_DIR="${TRELLIS_COLAB_CUDA_HOME:-/opt/cuda-12.1}"
TRELLIS_DIR="$WORK/TRELLIS"
BLENDER_DIR="$WORK/blender"
ENV_FILE="$WORK/trellis_colab_env.sh"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYS_PY="$(command -v python3 || true)"    # Python del sistema (solo se usa para instalar uv)
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

HEARTBEAT_SECS="${TRELLIS_COLAB_HEARTBEAT:-30}"
ARCH=""            # arquitecturas CUDA para compilar, p. ej. "7.5"
GPU_NAME=""
GPU_MEM_MIB=""
GPU_CC=""

if [ -t 2 ]; then
  C_BOLD=$'\033[1m'; C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_OFF=$'\033[0m'
else
  C_BOLD=""; C_GREEN=""; C_RED=""; C_YELLOW=""; C_OFF=""
fi

# --------------------------------------------------------------------------- utilidades
die()  { echo "${C_RED}✗ $*${C_OFF}" >&2; exit 1; }
warn() { echo "${C_YELLOW}! $*${C_OFF}" >&2; }

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# Imprime el bloque de comentarios inicial del script.
usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

# step [--show] "Título" comando [args...]
# Ejecuta el comando con su salida en un registro, muestra un latido cada 30 s y, si falla,
# imprime las últimas líneas del registro y aborta. Con --show vuelca el registro al terminar.
step() {
  local show=0
  if [ "${1:-}" = "--show" ]; then show=1; shift; fi
  local title="$1"; shift
  local slug log t0 pid next_beat
  slug="$(printf '%s' "$title" | tr -c 'A-Za-z0-9' '_')"
  mkdir -p "$LOGS"
  log="$LOGS/${slug}.log"
  t0=$SECONDS
  next_beat=$HEARTBEAT_SECS
  echo "${C_BOLD}▶ ${title}${C_OFF}"
  ( trap - ERR; "$@" ) >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 0.2
    if [ $((SECONDS - t0)) -ge "$next_beat" ]; then
      echo "  … $((SECONDS - t0)) s"
      next_beat=$((next_beat + HEARTBEAT_SECS))
    fi
  done
  if wait "$pid"; then
    echo "  ${C_GREEN}✓${C_OFF} ${title} ($((SECONDS - t0)) s)"
    if [ "$show" -eq 1 ]; then sed 's/^/    /' "$log"; fi
  else
    echo "  ${C_RED}✗ Falló: ${title}${C_OFF}  (registro completo: $log)" >&2
    echo "  Últimas líneas:" >&2
    tail -n 40 "$log" | sed 's/^/    /' >&2
    exit 1
  fi
}

uv_cmd() { "$SYS_PY" -m uv "$@"; }
uv_pip() { uv_cmd pip install --python "$VENV/bin/python" "$@"; }
venv_has() { "$VENV/bin/python" -c "import $1" >/dev/null 2>&1; }

# clone_at URL DIR REF [recurse]  — clona (si hace falta) y deja el árbol en REF.
clone_at() {
  local url="$1" dir="$2" ref="$3" recurse="${4:-}"
  if [ ! -d "$dir/.git" ]; then
    rm -rf "$dir"
    mkdir -p "$(dirname "$dir")"
    if [ -n "$recurse" ]; then
      git clone --quiet --recurse-submodules "$url" "$dir"
    else
      git clone --quiet "$url" "$dir"
    fi
  fi
  git -C "$dir" -c advice.detachedHead=false checkout --quiet --force "$ref"
  if [ -n "$recurse" ]; then
    git -C "$dir" submodule update --quiet --init --recursive
  fi
}

# --------------------------------------------------------------------------- GPU
# parse_gpu "Tesla T4, 15360, 7.5"  -> GPU_NAME / GPU_MEM_MIB / GPU_CC
parse_gpu() {
  local name mem cc
  IFS=',' read -r name mem cc <<<"$1"
  GPU_NAME="$(trim "$name")"
  GPU_MEM_MIB="$(trim "$mem")"
  GPU_CC="$(trim "$cc")"
}

# check_gpu: valida la GPU ya parseada y fija ARCH. Termina con die() si no sirve.
check_gpu() {
  if [[ "$GPU_CC" =~ ^[0-9]+\.[0-9]+$ ]]; then
    local major="${GPU_CC%%.*}"
    if [ "$major" -ge 10 ]; then
      die "La GPU ${GPU_NAME} (capacidad ${GPU_CC}, arquitectura Blackwell) no es compatible con este cuaderno: usa PyTorch 2.4 + CUDA 12.1, que no incluye kernels para ella. Cambia el entorno de ejecución a una GPU T4, L4, A100 o H100."
    fi
    ARCH="$GPU_CC"
  else
    warn "No se pudo leer la capacidad de cómputo; se compilará para varias arquitecturas (más lento)."
    ARCH="7.5;8.0;8.6;8.9;9.0"
  fi
  if [[ "$GPU_MEM_MIB" =~ ^[0-9]+$ ]]; then
    if [ "$GPU_MEM_MIB" -lt 14000 ]; then
      warn "La GPU tiene ${GPU_MEM_MIB} MiB: TRELLIS necesita unos 16 GB, es muy probable que falte memoria."
    elif [ "$GPU_MEM_MIB" -lt 20000 ]; then
      warn "La GPU tiene ${GPU_MEM_MIB} MiB: está al límite de lo que pide TRELLIS (16 GB); si falta memoria, usa una L4 o A100."
    fi
  fi
}

preflight() {
  [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] \
    || die "Este script solo funciona en Linux x86_64 (el runtime de Colab)."
  [ -n "$SYS_PY" ] || die "No se encontró python3 en el sistema."
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "No hay GPU (no existe nvidia-smi). En Colab: Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador por hardware: GPU."
  local line
  line="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  [ -n "$line" ] || die "nvidia-smi no detecta ninguna GPU. En Colab: Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU."
  parse_gpu "$line"
  echo "GPU: ${GPU_NAME} · ${GPU_MEM_MIB} MiB · capacidad de cómputo ${GPU_CC:-desconocida}"
  check_gpu

  mkdir -p "$WORK"
  local free_kb
  free_kb="$(df -Pk "$WORK" | awk 'NR==2 {print $4}')"
  if [ -n "$free_kb" ] && [ "$free_kb" -lt $((30 * 1024 * 1024)) ]; then
    warn "Quedan menos de 30 GB libres en $WORK; la instalación necesita unos 30 GB."
  fi
}

# --------------------------------------------------------------------------- entorno
# Fichero que se "source" tanto aquí (para compilar y verificar) como al lanzar la app.
write_env_file() {
  mkdir -p "$WORK"
  cat >"$ENV_FILE" <<EOF
# Generado por colab/setup_colab.sh — entorno para compilar y ejecutar TRELLIS.
export VIRTUAL_ENV="$VENV"
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$VENV/bin:$CUDA_HOME_DIR/bin:\$PATH"
# nvcc 12.1 no admite el GCC 13 que trae Colab: se usa GCC 12 como compilador anfitrión.
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="${MAX_JOBS:-4}"
export TRELLIS_ROOT="$TRELLIS_DIR"
export BLENDER_BIN="$BLENDER_DIR/blender"
# GPUs sin FlashAttention-2 (T4): atención con xformers; spconv sin autotuning inicial.
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
# Enlace público de Gradio (la app se lanza como subproceso y no detecta Colab por sí sola).
export GRADIO_SHARE=True
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$WORK/.uv-cache"
EOF
}

# --------------------------------------------------------------------------- fases
phase_apt() {
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq || echo "aviso: apt-get update devolvió un error; se continúa con el índice existente"
  $SUDO apt-get install -y -qq --no-install-recommends \
    gcc-12 g++-12 git curl wget xz-utils ca-certificates \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libsm6 libxkbcommon0 libegl1
  gcc-12 --version | head -n 1
}

phase_cuda() {
  if [ -x "$CUDA_HOME_DIR/bin/nvcc" ]; then
    echo "Toolkit ya instalado"; "$CUDA_HOME_DIR/bin/nvcc" --version | tail -n 2; return 0
  fi
  local run="$WORK/cuda_12.1.1_linux.run"
  mkdir -p "$WORK/tmp"
  wget -q --tries=3 -O "$run" "$CUDA_RUNFILE_URL"
  # --toolkit: sin driver (el de Colab ya es más nuevo). --override: salta la comprobación de GCC.
  if ! $SUDO sh "$run" --silent --toolkit --toolkitpath="$CUDA_HOME_DIR" --no-man-page --override --tmpdir="$WORK/tmp"; then
    echo "--- /var/log/cuda-installer.log ---"
    tail -n 40 /var/log/cuda-installer.log 2>/dev/null || true
    return 1
  fi
  rm -f "$run"
  "$CUDA_HOME_DIR/bin/nvcc" --version | tail -n 2
}

phase_python() {
  if ! "$SYS_PY" -m uv --version >/dev/null 2>&1; then
    # PIP_BREAK_SYSTEM_PACKAGES: por si python3 fuese un Python "externally managed" (PEP 668).
    PIP_BREAK_SYSTEM_PACKAGES=1 "$SYS_PY" -m pip install -q --disable-pip-version-check uv
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    uv_cmd venv --python 3.10 --seed "$VENV"
  fi
  "$VENV/bin/python" -c "import sys; print('Python', sys.version.split()[0]); assert sys.version_info[:2] == (3, 10)"
}

phase_torch() {
  if "$VENV/bin/python" -c "import sys, torch; sys.exit(0 if torch.__version__.startswith('2.4.0') and torch.version.cuda == '12.1' else 1)" 2>/dev/null; then
    echo "PyTorch 2.4.0 + CUDA 12.1 ya instalado"; return 0
  fi
  uv_pip "$TORCH_SPEC" "$TORCHVISION_SPEC" --index-url "$TORCH_INDEX"
}

phase_xformers() {
  if venv_has xformers; then echo "xformers ya instalado"; return 0; fi
  uv_pip --no-deps "$XFORMERS_SPEC"
}

phase_lock() {
  uv_pip -r "$REPO_DIR/colab/requirements-colab.lock"
  uv_cmd pip check --python "$VENV/bin/python" || echo "aviso: 'uv pip check' reportó incompatibilidades (revisa este registro)"
}

phase_rasterizer() {
  if venv_has diff_gaussian_rasterization; then echo "diff-gaussian-rasterization ya instalado"; return 0; fi
  clone_at "https://github.com/autonomousvision/mip-splatting.git" "$SRC/mip-splatting" "$MIPSPLAT_COMMIT"
  nvcc --version | tail -n 2
  # --no-build-isolation: su setup.py hace `import torch`, que no existe en el entorno aislado de build.
  uv_pip --no-build-isolation "$SRC/mip-splatting/submodules/diff-gaussian-rasterization"
}

phase_nvdiffrast() {
  if venv_has nvdiffrast.torch; then echo "nvdiffrast ya instalado"; return 0; fi
  clone_at "https://github.com/NVlabs/nvdiffrast.git" "$SRC/nvdiffrast" "$NVDIFFRAST_REF"
  uv_pip --no-build-isolation "$SRC/nvdiffrast"
}

phase_trellis() {
  clone_at "https://github.com/microsoft/TRELLIS.git" "$TRELLIS_DIR" "$TRELLIS_COMMIT" recurse
  # FlexiCubes es un submódulo: si está vacío, el decodificador de mallas no funcionará.
  [ -n "$(ls -A "$TRELLIS_DIR/trellis/representations/mesh/flexicubes" 2>/dev/null)" ] \
    || { echo "El submódulo FlexiCubes está vacío"; return 1; }
  git -C "$TRELLIS_DIR" log -1 --format='TRELLIS %h (%cd)' --date=short
}

phase_blender() {
  if [ -x "$BLENDER_DIR/blender" ] && [ "$(cat "$BLENDER_DIR/.version" 2>/dev/null || true)" = "$BLENDER_VERSION" ]; then
    echo "Blender $BLENDER_VERSION ya instalado"; return 0
  fi
  local tarball="$WORK/blender-${BLENDER_VERSION}.tar.xz"
  curl -fL --retry 3 --retry-delay 5 -o "$tarball" "$BLENDER_URL"
  echo "$BLENDER_SHA256  $tarball" | sha256sum -c -
  rm -rf "$BLENDER_DIR" "$WORK/blender-${BLENDER_VERSION}-linux-x64"
  tar -xJf "$tarball" -C "$WORK"
  mv "$WORK/blender-${BLENDER_VERSION}-linux-x64" "$BLENDER_DIR"
  echo "$BLENDER_VERSION" >"$BLENDER_DIR/.version"
  rm -f "$tarball"
}

phase_verify() {
  "$VENV/bin/python" - <<'PY'
import importlib
import os
import sys
import time


def check(label, fn):
    start = time.time()
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - diagnóstico
        print(f"  FALLO  {label}: {type(exc).__name__}: {exc}")
        raise
    took = time.time() - start
    suffix = f"  [{took:.0f} s]" if took > 5 else ""
    print(f"  ok     {label}" + (f" -> {detail}" if detail else "") + suffix)


def imp(name):
    module = importlib.import_module(name)
    try:
        return str(getattr(module, "__version__", ""))
    except Exception:  # noqa: BLE001 - utils3d lanza ModuleNotFoundError (no AttributeError) en __getattr__
        return ""


import torch

check(
    "PyTorch + CUDA",
    lambda: f"{torch.__version__} / CUDA {torch.version.cuda} / {torch.cuda.get_device_name(0)}",
)
check("operación en la GPU", lambda: str(float(torch.ones(4, device="cuda").sum())))

for module in (
    "numpy", "xformers.ops", "spconv.pytorch", "diff_gaussian_rasterization", "nvdiffrast.torch",
    "utils3d", "open3d", "rembg", "trimesh", "xatlas", "pyvista", "pymeshfix", "igraph",
    "transformers", "easydict", "imageio", "gradio",
):
    check(f"import {module}", lambda m=module: imp(m))

sys.path.insert(0, os.environ["TRELLIS_ROOT"])
# Los mismos imports que hace app.py en load_trellis_runtime().
check("trellis.pipelines", lambda: str(bool(importlib.import_module("trellis.pipelines").TrellisImageTo3DPipeline)))
check("trellis.representations", lambda: str(bool(importlib.import_module("trellis.representations").Gaussian)))
check("trellis.utils", lambda: imp("trellis.utils.render_utils") + imp("trellis.utils.postprocessing_utils"))


def warm_nvdiffrast():
    # La primera vez compila el plugin CUDA de nvdiffrast (1-2 min) y lo deja en caché.
    import nvdiffrast.torch as dr

    dr.RasterizeCudaContext()
    return "plugin compilado"


check("nvdiffrast: contexto CUDA", warm_nvdiffrast)
PY
  local out
  out="$("$BLENDER_DIR/blender" --background --factory-startup \
        --python-expr "import bpy; print('BLENDER-OK', bpy.app.version_string)" 2>&1)" \
    || { echo "$out" | tail -n 20; return 1; }
  echo "$out" | grep -q "BLENDER-OK" || { echo "$out" | tail -n 20; return 1; }
  echo "  ok     Blender $(echo "$out" | sed -n 's/.*BLENDER-OK //p' | head -n 1)"
}

summary() {
  echo
  echo "${C_GREEN}${C_BOLD}✓ Instalación completada.${C_OFF}"
  echo "  Entorno: $ENV_FILE"
  echo "  Siguiente paso: ejecuta la celda «3 · Iniciar la aplicación»."
}

main() {
  case "${1:-}" in
    -h|--help) usage; return 0 ;;
  esac
  preflight
  write_env_file
  # shellcheck disable=SC1090
  source "$ENV_FILE"

  step "Paquetes del sistema (GCC 12 y librerías de Blender)" phase_apt
  step "Toolkit CUDA 12.1 (nvcc; descarga de ~4 GB)" phase_cuda
  step "Python 3.10 aislado con uv" phase_python
  step "PyTorch 2.4.0 + CUDA 12.1" phase_torch
  step "xformers" phase_xformers
  step "Dependencias de Python (lockfile)" phase_lock
  step "Compilar diff-gaussian-rasterization" phase_rasterizer
  step "Instalar nvdiffrast" phase_nvdiffrast
  step "Descargar TRELLIS" phase_trellis
  step "Descargar Blender ${BLENDER_VERSION}" phase_blender
  step --show "Verificar la instalación (compila el plugin de nvdiffrast)" phase_verify
  summary
}

# Solo se ejecuta si se invoca directamente; con `source` quedan disponibles las funciones.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  trap 'echo "${C_RED}✗ setup_colab.sh terminó con error (código $?) cerca de la línea $LINENO${C_OFF}" >&2' ERR
  main "$@"
fi
