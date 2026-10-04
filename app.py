"""Gradio frontend for Microsoft TRELLIS with optional Blender auto-rigging.

This app expects the official TRELLIS checkout and its CUDA dependencies to be
installed separately. Set TRELLIS_ROOT if the checkout is not the current
working directory.
"""

# No añadir `from __future__ import annotations` en este módulo: Gradio resuelve la
# anotación `request: gr.Request` con typing.get_type_hints() y `gr` es una variable
# local de build_demo(), así que con anotaciones diferidas la interfaz no se construye
# (NameError: name 'gr' is not defined). Lo cubre tests/test_app_ui.py.

import argparse
import hashlib
import os
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from rigging import AutoRigError, auto_rig_glb


MAX_SEED = 2**31 - 1
TMP_DIR = Path(
    os.environ.get(
        "TRELLIS_RIG_TMP_DIR",
        str(Path(tempfile.gettempdir()) / "trellis-models-rig"),
    )
)


def load_trellis_runtime(trellis_root: Optional[str] = None) -> Dict[str, Any]:
    """Load TRELLIS lazily so the rigging wrapper remains usable on its own."""
    root_value = trellis_root or os.environ.get("TRELLIS_ROOT")
    if root_value:
        root = Path(root_value).expanduser().resolve()
        if not (root / "trellis").is_dir():
            raise RuntimeError(
                f"TRELLIS_ROOT no contiene el paquete `trellis`: {root}. "
                "Indica la raíz del checkout oficial de Microsoft TRELLIS."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    else:
        # Running this integration from the official checkout is also supported.
        working_root = Path.cwd().resolve()
        if (working_root / "trellis").is_dir() and str(working_root) not in sys.path:
            sys.path.insert(0, str(working_root))

    try:
        import gradio as gr
        import imageio
        import numpy as np
        import torch
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.representations import Gaussian
        from trellis.utils import postprocessing_utils, render_utils
    except ImportError as exc:
        raise RuntimeError(
            "No se pudieron importar las dependencias de TRELLIS. Instala primero "
            "TRELLIS y sus extensiones/CUDA según su setup.sh, instala requirements.txt "
            "de este repositorio y define TRELLIS_ROOT si el checkout está en otra ruta."
        ) from exc

    model_name = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
    pipeline.cuda()
    return {
        "gr": gr,
        "imageio": imageio,
        "np": np,
        "torch": torch,
        "Gaussian": Gaussian,
        "pipeline": pipeline,
        "postprocessing_utils": postprocessing_utils,
        "render_utils": render_utils,
    }


def _session_work_dir(request: Any) -> Path:
    session_hash = getattr(request, "session_hash", None) or uuid.uuid4().hex
    safe_session = hashlib.sha256(str(session_hash).encode("utf-8")).hexdigest()[:20]
    work_dir = TMP_DIR / safe_session / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _pack_state(gaussian: Any, mesh: Any, work_dir: Path) -> Dict[str, Any]:
    """Keep the TRELLIS output in Gradio state as CPU arrays, like upstream app.py."""
    return {
        "gaussian": {
            **gaussian.init_params,
            "_xyz": gaussian._xyz.detach().cpu().numpy(),
            "_features_dc": gaussian._features_dc.detach().cpu().numpy(),
            "_scaling": gaussian._scaling.detach().cpu().numpy(),
            "_rotation": gaussian._rotation.detach().cpu().numpy(),
            "_opacity": gaussian._opacity.detach().cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.detach().cpu().numpy(),
            "faces": mesh.faces.detach().cpu().numpy(),
        },
        "work_dir": str(work_dir),
    }


def _unpack_state(state: Dict[str, Any], runtime: Dict[str, Any]) -> Tuple[Any, Any]:
    torch = runtime["torch"]
    gaussian_state = state["gaussian"]
    gaussian = runtime["Gaussian"](
        aabb=gaussian_state["aabb"],
        sh_degree=gaussian_state["sh_degree"],
        mininum_kernel_size=gaussian_state["mininum_kernel_size"],
        scaling_bias=gaussian_state["scaling_bias"],
        opacity_bias=gaussian_state["opacity_bias"],
        scaling_activation=gaussian_state["scaling_activation"],
    )
    gaussian._xyz = torch.tensor(gaussian_state["_xyz"], device="cuda")
    gaussian._features_dc = torch.tensor(gaussian_state["_features_dc"], device="cuda")
    gaussian._scaling = torch.tensor(gaussian_state["_scaling"], device="cuda")
    gaussian._rotation = torch.tensor(gaussian_state["_rotation"], device="cuda")
    gaussian._opacity = torch.tensor(gaussian_state["_opacity"], device="cuda")

    mesh_state = state["mesh"]
    mesh = SimpleNamespace(
        vertices=torch.tensor(mesh_state["vertices"], device="cuda"),
        faces=torch.tensor(mesh_state["faces"], device="cuda"),
    )
    return gaussian, mesh


def build_demo(runtime: Dict[str, Any]):
    """Build the Gradio UI and bind its callbacks to a loaded TRELLIS pipeline."""
    gr = runtime["gr"]
    np = runtime["np"]
    torch = runtime["torch"]
    pipeline = runtime["pipeline"]
    render_utils = runtime["render_utils"]
    postprocessing_utils = runtime["postprocessing_utils"]
    imageio = runtime["imageio"]

    def get_seed(randomize_seed: bool, seed: int) -> int:
        return int(np.random.randint(0, MAX_SEED)) if randomize_seed else int(seed)

    def image_to_3d(
        image: Any,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        request: gr.Request,
    ):
        if image is None:
            raise gr.Error("Sube una imagen antes de generar el modelo.")

        work_dir = _session_work_dir(request)
        try:
            processed_image = pipeline.preprocess_image(image)
            outputs = pipeline.run(
                processed_image,
                seed=int(seed),
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": int(ss_sampling_steps),
                    "cfg_strength": float(ss_guidance_strength),
                },
                slat_sampler_params={
                    "steps": int(slat_sampling_steps),
                    "cfg_strength": float(slat_guidance_strength),
                },
            )

            gaussian = outputs["gaussian"][0]
            mesh = outputs["mesh"][0]
            gaussian_video = render_utils.render_video(gaussian, num_frames=120)["color"]
            mesh_video = render_utils.render_video(mesh, num_frames=120)["normal"]
            preview = [
                np.concatenate((gaussian_video[index], mesh_video[index]), axis=1)
                for index in range(len(gaussian_video))
            ]
            video_path = work_dir / "preview.mp4"
            imageio.mimsave(str(video_path), preview, fps=15)
            state = _pack_state(gaussian, mesh, work_dir)
            return state, str(video_path)
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if isinstance(exc, gr.Error):
                raise
            raise gr.Error(f"No se pudo generar el modelo TRELLIS: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_glb(
        state: Optional[Dict[str, Any]],
        mesh_simplify: float,
        texture_size: int,
        enable_auto_rigging: bool,
    ):
        if not state:
            raise gr.Error("Genera un modelo antes de exportar el GLB.")

        work_dir = Path(state["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        source_path = work_dir / "sample.glb"
        try:
            gaussian, mesh = _unpack_state(state, runtime)
            glb = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=float(mesh_simplify),
                texture_size=int(texture_size),
                verbose=False,
            )
            glb.export(str(source_path))

            if enable_auto_rigging:
                output_path = work_dir / "sample_rigged.glb"
                try:
                    auto_rig_glb(source_path, output_path)
                except AutoRigError as exc:
                    raise gr.Error(f"No se pudo completar el auto-rigging: {exc}") from exc
            else:
                output_path = source_path

            return str(output_path), str(output_path)
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(f"No se pudo exportar el modelo a GLB: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with gr.Blocks(delete_cache=(3600, 3600), title="TRELLIS · Auto-Rigging") as demo:
        gr.Markdown(
            """
            ## Imagen a 3D con [TRELLIS](https://github.com/microsoft/TRELLIS)

            Sube una imagen y genera el modelo. Al extraer el GLB puedes activar
            **Enable Auto-Rigging** para añadir una armature y pesos de skinning
            mediante Blender en modo headless.
            """
        )
        with gr.Row():
            with gr.Column():
                image_prompt = gr.Image(
                    label="Imagen de entrada",
                    image_mode="RGBA",
                    type="pil",
                    height=300,
                )
                with gr.Accordion(label="Ajustes de generación", open=False):
                    seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    randomize_seed = gr.Checkbox(label="Aleatorizar seed", value=True)
                    gr.Markdown("Etapa 1: generación de estructura dispersa")
                    with gr.Row():
                        ss_guidance_strength = gr.Slider(
                            0.0, 10.0, label="Guidance strength", value=7.5, step=0.1
                        )
                        ss_sampling_steps = gr.Slider(
                            1, 50, label="Sampling steps", value=12, step=1
                        )
                    gr.Markdown("Etapa 2: generación de latentes estructurados")
                    with gr.Row():
                        slat_guidance_strength = gr.Slider(
                            0.0, 10.0, label="Guidance strength", value=3.0, step=0.1
                        )
                        slat_sampling_steps = gr.Slider(
                            1, 50, label="Sampling steps", value=12, step=1
                        )

                generate_btn = gr.Button("Generate 3D")
                with gr.Accordion(label="Ajustes de exportación GLB", open=True):
                    mesh_simplify = gr.Slider(
                        0.90, 0.98, label="Simplify", value=0.95, step=0.01
                    )
                    texture_size = gr.Slider(
                        512, 2048, label="Texture size", value=1024, step=512
                    )
                    enable_auto_rigging = gr.Checkbox(
                        label="Enable Auto-Rigging",
                        value=False,
                        info=(
                            "Añade una armature humanoide aproximada con pesos de Blender. "
                            "Requiere Blender 3.6+ instalado."
                        ),
                    )
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)

            with gr.Column():
                video_output = gr.Video(
                    label="Vista previa generada",
                    autoplay=True,
                    loop=True,
                    height=300,
                )
                model_output = gr.Model3D(label="GLB exportado", height=360)
                download_glb = gr.DownloadButton(
                    label="Download GLB",
                    interactive=False,
                )

        model_state = gr.State()
        generate_btn.click(
            get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed],
        ).then(
            image_to_3d,
            inputs=[
                image_prompt,
                seed,
                ss_guidance_strength,
                ss_sampling_steps,
                slat_guidance_strength,
                slat_sampling_steps,
            ],
            outputs=[model_state, video_output],
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[extract_glb_btn],
        )

        video_output.clear(
            lambda: gr.update(interactive=False),
            outputs=[extract_glb_btn],
        )
        extract_glb_btn.click(
            extract_glb,
            inputs=[model_state, mesh_simplify, texture_size, enable_auto_rigging],
            outputs=[model_output, download_glb],
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[download_glb],
        )
        model_output.clear(
            lambda: gr.update(interactive=False),
            outputs=[download_glb],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="TRELLIS demo con auto-rigging opcional")
    parser.add_argument(
        "--trellis-root",
        default=None,
        help="Raíz del checkout de Microsoft TRELLIS (o usa TRELLIS_ROOT).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        help="Dirección de escucha de Gradio (por defecto: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        help="Puerto de Gradio (por defecto: 7860).",
    )
    args = parser.parse_args()

    try:
        runtime = load_trellis_runtime(args.trellis_root)
    except (ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    demo = build_demo(runtime)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
