# TRELLIS + Auto-Rigging · Cuaderno de Google Colab

Este ZIP contiene el cuaderno de Colab que convierte una imagen en un modelo 3D con
[Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) y, opcionalmente, exporta un **GLB con
esqueleto (armature) y pesos de skinning** generados con Blender.

| Archivo | Para qué sirve |
| --- | --- |
| `TRELLIS_AutoRig_Colab.ipynb` | El cuaderno de Colab. |
| `LEEME.md` | Esta guía. |

No hace falta nada más: el cuaderno descarga desde GitHub la app Gradio, los scripts de instalación,
TRELLIS y Blender.

## Cómo usarlo

1. Descomprime el ZIP.
2. Entra en <https://colab.research.google.com> → **Archivo → Subir cuaderno** y elige
   `TRELLIS_AutoRig_Colab.ipynb`. (También puedes subirlo a Google Drive y abrirlo desde allí con
   *Google Colaboratory*.)
3. **Entorno de ejecución → Cambiar tipo de entorno de ejecución →** acelerador por hardware **GPU**.
4. **Entorno de ejecución → Ejecutar todas**. La primera vez puede tardar entre 15 y 25 minutos.
5. Cuando el paso 3 muestre un enlace `https://….gradio.live`, ábrelo: sube una imagen, pulsa
   **Generate 3D**, marca **Enable Auto-Rigging**, pulsa **Extract GLB** y descarga el resultado con
   **Download GLB**.

## Requisitos y límites

- **GPU NVIDIA con unos 16 GB**: una T4 (15 GB) va justa; una L4, A100 o H100 va mejor. Las GPU
  Blackwell (runtime «G4» de Colab) no son compatibles con PyTorch 2.4 / CUDA 12.1.
- El cuaderno **no modifica el entorno de Colab**: crea uno aparte (Python 3.10, PyTorch 2.4,
  CUDA 12.1), porque el actual de Colab (Python 3.13, CUDA 13, numpy 2) no es compatible con TRELLIS.
- Colab es efímero: si la sesión caduca hay que repetir la instalación, y los modelos generados se
  pierden al cerrarla. Descarga el GLB antes de parar.
- **El rig es una plantilla humanoide aproximada**, no un detector de articulaciones: pensado como
  punto de partida para personajes más o menos verticales.
- El enlace público de Gradio lo puede usar cualquiera que lo tenga mientras la app esté en marcha.

## Si algo falla

La instalación se apoya en versiones fijadas (PyTorch 2.4.0, CUDA 12.1, Blender 4.5 LTS, commits de
TRELLIS y de las extensiones), pero Colab cambia su entorno con frecuencia. Si una fase falla, el
cuaderno muestra las últimas líneas del registro; el registro completo está en
`/content/setup_logs/`. Abre un *issue* en el repositorio adjuntándolo.

## Más información

- Repositorio: <https://github.com/aqdriop/Trellis-models-rig>
- TRELLIS: <https://github.com/microsoft/TRELLIS>
- Blender: <https://www.blender.org>

Por defecto el cuaderno usa la versión del repositorio indicada en `REPO_REF` (paso 1). Cámbiala a
`main` para probar el código más reciente.
