# TRELLIS + Auto-Rigging

Capa de integración para añadir auto-rigging al flujo de generación de imagen a 3D de [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS). Este repositorio no copia los pesos ni el código completo de TRELLIS: `app.py` reutiliza su pipeline instalado y añade la exportación con rig.

## Flujo

En el código upstream revisé `app.py`: `pipeline.run(..., formats=["gaussian", "mesh"])` devuelve esas dos representaciones, y `extract_glb` las entrega a `trellis.utils.postprocessing_utils.to_glb(...)` antes de llamar a `glb.export(...)`. La interfaz estándar produce GLB (no OBJ); el rigging de esta integración opera sobre ese GLB para conservar la textura y los materiales. Esta app conserva la ruta oficial de texturizado/exportación. Al marcar **Enable Auto-Rigging**, ejecuta Blender en segundo plano para:

1. importar el GLB texturizado generado por TRELLIS;
2. crear una armadura humanoide aproximada ajustada a la caja envolvente de la malla;
3. asociar la malla al esqueleto con pesos automáticos de Blender (y usar un cálculo de pesos de respaldo si Blender no puede resolverlos);
4. exportar un segundo GLB con armature, skin y materiales, que se muestra y se descarga desde la interfaz.

Con la casilla desmarcada se descarga el GLB normal, sin cambios. El archivo original se conserva incluso cuando se solicita rigging.

> El rig es una plantilla geométrica, no un detector semántico de articulaciones ni un modelo aprendido como UniRig. Está pensado como punto de partida para personajes aproximadamente verticales; los objetos no humanoides y las formas muy estilizadas pueden necesitar ajuste manual.

## Requisitos

- Una instalación funcional de TRELLIS y sus dependencias/CUDA. Sigue primero las instrucciones oficiales de [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) (el setup de TRELLIS gestiona PyTorch y sus extensiones CUDA; no se instala un `torch` genérico desde este repositorio).
- Python del entorno TRELLIS y los paquetes de `requirements.txt`.
- Blender 3.6 o posterior disponible como ejecutable. Para usar el auto-rig, `blender` debe estar en `PATH` o hay que definir `BLENDER_BIN` con la ruta completa al ejecutable. Blender no es una dependencia instalable con pip.
- GPU NVIDIA compatible con el modelo TRELLIS seleccionado.

## Instalación y ejecución

1. Prepara el entorno siguiendo la guía oficial de TRELLIS. Clona TRELLIS con sus submódulos y usa su `setup.sh` para instalar las dependencias específicas de GPU.
2. Desde ese mismo entorno, instala las dependencias de esta capa:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Indica dónde está el checkout oficial de TRELLIS y arranca la app:

   ```bash
   TRELLIS_ROOT=/ruta/a/TRELLIS python app.py
   ```

   Si ejecutas el comando desde la raíz del checkout oficial, también puedes omitir `TRELLIS_ROOT`.

   En Windows PowerShell:

   ```powershell
   $env:TRELLIS_ROOT = "C:\ruta\a\TRELLIS"
   $env:BLENDER_BIN = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
   python app.py
   ```

   El servidor escucha por defecto en `0.0.0.0:7860`. Se puede cambiar con `GRADIO_SERVER_NAME` y `GRADIO_SERVER_PORT`.

4. Sube una imagen, genera el modelo y pulsa **Extract GLB**. Marca **Enable Auto-Rigging** antes de extraer si quieres el GLB riggeado. Blender se ejecuta únicamente durante esa exportación.

## Uso del script de rigging sin la interfaz

Con Blender instalado también puede invocarse el paso de rigging desde la línea de comandos:

```bash
python -c "from rigging import auto_rig_glb; print(auto_rig_glb('modelo.glb', 'modelo_rigged.glb'))"
```

El script de Blender utilizado por el wrapper está en `scripts/auto_rig_glb.py`. El wrapper informa explícitamente si Blender falta, agota el tiempo de espera o no puede generar el archivo; no devuelve silenciosamente un GLB sin rig cuando la casilla está activada.

## Colab y portal de descarga

Para ejecutar todo en Google Colab, sin instalar nada en tu equipo, el repositorio incluye:

- **Cuaderno de Colab**: [`colab/TRELLIS_AutoRig_Colab.ipynb`](colab/TRELLIS_AutoRig_Colab.ipynb) ([ábrelo en Colab](https://colab.research.google.com/github/aqdriop/Trellis-models-rig/blob/v0.1.0/colab/TRELLIS_AutoRig_Colab.ipynb)). Clona este repositorio, ejecuta [`colab/setup_colab.sh`](colab/setup_colab.sh) y lanza `app.py` con un enlace público de Gradio.
- **ZIP del cuaderno**: [`portal/downloads/trellis-autorig-colab.zip`](portal/downloads/trellis-autorig-colab.zip) (cuaderno + `LEEME.md`), publicado también como archivo de la [release](https://github.com/aqdriop/Trellis-models-rig/releases/latest): [descarga directa](https://github.com/aqdriop/Trellis-models-rig/releases/latest/download/trellis-autorig-colab.zip).
- **Portal web** para Netlify: sitio estático en [`portal/`](portal/) con el botón de descarga del ZIP, configurado en [`netlify.toml`](netlify.toml).

El runtime actual de Colab (Python 3.13, CUDA 13, numpy 2, Gradio 6) no es compatible con TRELLIS ni con `requirements.txt`, así que `setup_colab.sh` no toca ese entorno: crea uno aparte con Python 3.10, PyTorch 2.4.0 + CUDA 12.1 (con un toolkit CUDA 12.1 propio para compilar), `diff-gaussian-rasterization` y `nvdiffrast` compilados para la GPU detectada, TRELLIS en un commit fijado y Blender 4.5 LTS verificado con SHA-256. Las dependencias de Python están fijadas en [`colab/requirements-colab.lock`](colab/requirements-colab.lock), generado desde [`colab/requirements-colab.in`](colab/requirements-colab.in) con `uv pip compile` (el comando está en la cabecera del lock). Requisitos: una GPU NVIDIA con unos 16 GB (una T4 va justa; L4, A100 o H100 mejor). Las GPU Blackwell («G4») no son compatibles con PyTorch 2.4 / CUDA 12.1 y el script se detiene al detectarlas.

> `setup_colab.sh` omite `kaolin`, `diffoctreerast` y `vox2seq` del `setup.sh` oficial: ningún import del camino imagen → GLB los carga (solo se usan para renderizar radiance fields y entrenar). Tampoco usa `setup.sh --demo`: instala `gradio==4.44.1`, que con las dependencias actuales ni siquiera importa.

### Publicar el portal en Netlify

1. En Netlify: **Add new project → Import an existing project**, elige este repositorio y la rama `main`.
2. No cambies nada: `netlify.toml` fija `base = "portal"` y `publish = "."`, y no hay comando de build. El `base` es importante: Netlify instala el `requirements.txt` que encuentre en el directorio base y, sin él, intentaría instalar el de la raíz (Gradio, numpy…), que es de la app y no del sitio.
3. Vista previa local, sin Netlify: `python -m http.server --directory portal 8080`.

### Actualizar el cuaderno y publicar una versión nueva

1. Edita `colab/TRELLIS_AutoRig_Colab.ipynb` y/o `colab/LEEME.md`.
2. Sube `BUNDLE_VERSION` y `BUNDLE_DATE` en `scripts/build_colab_zip.py`; pon `REPO_REF = "v<versión>"` en la celda 1 del cuaderno y actualiza esa versión en las URL «Abrir en Colab» (cuaderno, `portal/index.html` y este README). Las pruebas avisan si algo no coincide.
3. Regenera el ZIP y su manifiesto con `python scripts/build_colab_zip.py` (`--check` comprueba que lo versionado está al día).
4. Haz commit y publica la release con el ZIP como archivo adjunto:

   ```bash
   gh release create v<versión> portal/downloads/trellis-autorig-colab.zip \
     --title "Cuaderno de Colab v<versión>" --notes "..."
   ```

   Para corregir el archivo de una release existente: `gh release upload v<versión> portal/downloads/trellis-autorig-colab.zip --clobber`.

## Pruebas

Las pruebas no requieren CUDA, Blender, red ni Colab:

```bash
python -m unittest discover -s tests -v
```

Cubren el wrapper de rigging, la construcción de la interfaz Gradio (se omite si Gradio no está instalado), el cuaderno (incluida la ejecución de sus celdas contra un Colab simulado), `setup_colab.sh`, el lockfile, el ZIP con su manifiesto y el portal con `netlify.toml`. Algunas se omiten si faltan `bash`, `git` o `tomllib`/`tomli`.
