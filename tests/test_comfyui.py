"""ComfyUI prompt-graph extraction, including modern (Flux/SD3) node types."""

from __future__ import annotations

import json
from collections import OrderedDict

from conftest import png_bytes


def extract(pi, prompt: dict, workflow: dict | None = None) -> dict[str, str]:
    return pi.MetadataComfyUI(
        json.dumps(prompt),
        json.dumps(workflow) if workflow is not None else None,
    ).params


def field(params: dict[str, str], class_type: str) -> str:
    """The single params entry whose key names `class_type`."""
    matches = [v for k, v in params.items() if k.startswith(f"{class_type}.")]
    assert len(matches) == 1, f"expected one {class_type} entry, got {len(matches)}"
    return matches[0]


def test_classic_ksampler_graph(pi):
    params = extract(
        pi,
        {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 25,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            },
        },
    )
    sampler = field(params, "KSampler")
    assert "[seed]: 42" in sampler
    assert "[steps]: 25" in sampler
    assert "[cfg]: 7.5" in sampler
    assert "[sampler_name]: dpmpp_2m" in sampler
    # `denoise` was previously missing from the KSampler handler.
    assert "[denoise]: 1.0" in sampler
    assert "sd_xl_base_1.0.safetensors" in field(params, "CheckpointLoaderSimple")


def test_integer_valued_float_widget_is_kept(pi):
    """ComfyUI serialises `cfg: 8` as an int; it must not be dropped."""
    params = extract(
        pi,
        {"3": {"class_type": "KSampler", "inputs": {"cfg": 8, "steps": 20}}},
    )
    sampler = field(params, "KSampler")
    assert "[cfg]: 8" in sampler
    assert "[steps]: 20" in sampler


def test_flux_graph_captures_unet_clip_lora_and_guidance(pi, flux_comfy_prompt):
    params = extract(pi, flux_comfy_prompt)
    assert "flux1-dev.safetensors" in field(params, "UNETLoader")
    assert "fp8_e4m3fn" in field(params, "UNETLoader")

    dual = field(params, "DualCLIPLoader")
    assert "t5xxl_fp16.safetensors" in dual
    assert "clip_l.safetensors" in dual
    assert "[type]: flux" in dual

    lora = field(params, "LoraLoaderModelOnly")
    assert "detail_slider.safetensors" in lora
    assert "[strength_model]: 0.8" in lora

    # guidance is an int in the fixture, exercising the numeric-type handling.
    assert "[guidance]: 3" in field(params, "FluxGuidance")
    assert "219670278747233" in field(params, "RandomNoise")
    assert "[sampler_name]: euler" in field(params, "KSamplerSelect")

    scheduler = field(params, "BasicScheduler")
    assert "[scheduler]: simple" in scheduler
    assert "[steps]: 20" in scheduler

    latent = field(params, "EmptySD3LatentImage")
    assert "[width]: 1024" in latent
    assert "[height]: 1024" in latent


def test_node_titles_are_included_when_enabled(pi, flux_comfy_prompt):
    pi.CFG.comfyui_show_titles = True
    params = extract(pi, flux_comfy_prompt)
    assert any("Positive Prompt" in k for k in params)


def test_node_titles_are_omitted_when_disabled(pi, flux_comfy_prompt):
    pi.CFG.comfyui_show_titles = False
    params = extract(pi, flux_comfy_prompt)
    assert not any("Positive Prompt" in k for k in params)


def test_sd3_triple_clip_and_text_encode(pi):
    params = extract(
        pi,
        {
            "1": {
                "class_type": "TripleCLIPLoader",
                "inputs": {
                    "clip_name1": "clip_g.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "clip_name3": "t5xxl_fp8.safetensors",
                },
            },
            "2": {
                "class_type": "CLIPTextEncodeSD3",
                "inputs": {
                    "clip_l": "a lighthouse",
                    "clip_g": "a lighthouse, dramatic",
                    "t5xxl": "a lighthouse at dusk",
                    "empty_padding": "none",
                },
            },
        },
    )
    triple = field(params, "TripleCLIPLoader")
    assert "clip_g.safetensors" in triple
    assert "t5xxl_fp8.safetensors" in triple
    encode = field(params, "CLIPTextEncodeSD3")
    assert "a lighthouse at dusk" in encode


def test_wan_video_latent_dimensions(pi):
    params = extract(
        pi,
        {
            "1": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "length": 81,
                    "batch_size": 1,
                },
            },
        },
    )
    wan = field(params, "WanImageToVideo")
    assert "[length]: 81" in wan
    assert "[width]: 832" in wan


def test_gguf_loader(pi):
    params = extract(
        pi,
        {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "flux-Q4.gguf"},
            }
        },
    )
    assert "flux-Q4.gguf" in field(params, "UnetLoaderGGUF")


def test_unknown_node_types_are_skipped(pi):
    params = extract(
        pi,
        {"1": {"class_type": "SomeCustomNodeNobodyHasEverHeardOf", "inputs": {"x": 1}}},
    )
    assert params == {}


def test_newlines_in_prompts_are_flattened_when_enabled(pi):
    pi.CFG.comfyui_replace_field_newlines = True
    params = extract(
        pi,
        {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "line one\nline two"},
            }
        },
    )
    assert "line one line two" in field(params, "CLIPTextEncode")


def test_widget_values_override_from_workflow(pi):
    """A ShowText node's real value lives in the workflow's widgets_values."""
    pi.CFG.comfyui_extract_widget_values = True
    params = extract(
        pi,
        {"9": {"class_type": "ShowText|pysssss", "inputs": {"text": ["8", 0]}}},
        {
            "nodes": [
                {
                    "id": 9,
                    "inputs": [{"name": "text", "widget": {"name": "text"}}],
                    "widgets_values": [["the resolved caption"]],
                },
            ],
        },
    )
    assert "the resolved caption" in field(params, "ShowText|pysssss")


def test_detected_from_png_prompt_chunk(pi, flux_comfy_prompt):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(
        0,
        png_bytes(prompt=json.dumps(flux_comfy_prompt)),
        metadata,
    )
    assert list(metadata) == [0]
    assert metadata[0].NAME == "ComfyUI"
    assert metadata[0].EXTENSION == "json"
