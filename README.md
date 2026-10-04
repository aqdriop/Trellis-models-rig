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

## Pruebas

Las pruebas del wrapper no requieren CUDA ni Blender:

```bash
python -m unittest discover -s tests -v
```
