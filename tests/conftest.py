"""Shared fixtures: synthetic images carrying generation metadata.

Fixtures are generated rather than committed so the expected values live next to
the assertions and no binary blobs enter the repo.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin

REPO_ROOT = Path(__file__).resolve().parent.parent

# Discord's documented ceiling for the combined title, description, field names,
# field values, footer text and author name across an embed.
DISCORD_EMBED_TOTAL_LIMIT = 6000


@pytest.fixture(scope="session")
def pi():
    """The PromptInspector module, imported from the repo root by path."""
    if "PromptInspector" in sys.modules:
        return sys.modules["PromptInspector"]
    spec = importlib.util.spec_from_file_location(
        "PromptInspector",
        REPO_ROOT / "PromptInspector.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["PromptInspector"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _default_config(pi):
    """Every test starts from the in-code defaults, not a user's config.toml."""
    pi.CFG.set_defaults()
    return pi.CFG


A1111_PARAMETERS = (
    "a cat sitting on a windowsill, masterpiece\n"
    "Negative prompt: blurry, low quality\n"
    "Steps: 28, Sampler: DPM++ 2M, CFG scale: 6.5, Seed: 987654321, "
    "Size: 832x1216, Model: someModel_v3, VAE: sdxl_vae.safetensors, "
    "Denoising strength: 0.45, Version: v1.10.1"
)


def png_bytes(**text_chunks: str) -> bytes:
    img = Image.new("RGB", (8, 8), "navy")
    info = PngImagePlugin.PngInfo()
    for key, value in text_chunks.items():
        info.add_text(key, value)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def exif_user_comment_bytes(
    text: str,
    fmt: str = "JPEG",
    encoding: str = "utf-16-be",
) -> bytes:
    """An image with `text` in Exif UserComment, the way A1111 writes JPEG/WebP."""
    img = Image.new("RGB", (8, 8), "darkgreen")
    exif = img.getexif()
    prefix = b"UNICODE\x00" if encoding.startswith("utf-16") else b"ASCII\x00\x00\x00"
    exif.get_ifd(0x8769)[0x9286] = prefix + text.encode(encoding)
    buf = io.BytesIO()
    img.save(buf, format=fmt, exif=exif)
    return buf.getvalue()


def exif_comfy_bytes(prompt: dict, workflow: dict | None = None) -> bytes:
    """An image with ComfyUI prompt/workflow in Exif, the way ComfyUI writes WebP."""
    img = Image.new("RGB", (8, 8), "maroon")
    exif = img.getexif()
    exif[0x010F] = "Prompt:" + json.dumps(prompt)
    if workflow is not None:
        exif[0x010E] = "Workflow:" + json.dumps(workflow)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", exif=exif)
    return buf.getvalue()


@pytest.fixture
def a1111_png() -> bytes:
    return png_bytes(parameters=A1111_PARAMETERS)


@pytest.fixture
def flux_comfy_prompt() -> dict:
    """A representative Flux prompt graph, as ComfyUI serialises it."""
    return {
        "6": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
            "inputs": {"text": "a fox in a snowy forest", "clip": ["11", 0]},
        },
        "11": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp16.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
        },
        "12": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-dev.safetensors",
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "13": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "detail_slider.safetensors",
                "strength_model": 0.8,
                "model": ["12", 0],
            },
        },
        "26": {
            # cfg deliberately serialised as an int, which is what ComfyUI does
            # for whole numbers and what a bare isinstance(v, float) would drop.
            "class_type": "FluxGuidance",
            "inputs": {"guidance": 3, "conditioning": ["6", 0]},
        },
        "25": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 219670278747233},
        },
        "16": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "17": {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": "simple",
                "steps": 20,
                "denoise": 1,
                "model": ["13", 0],
            },
        },
        "27": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
    }


class FakeAuthor:
    """The subset of Member that Metadata.get_embed touches."""

    def __init__(self, name: str = "someuser"):
        self._name = name
        self.color = 0x3498DB
        self.display_avatar = "https://example.invalid/avatar.png"

    def __str__(self) -> str:
        return self._name


@pytest.fixture
def msg_ctx():
    return SimpleNamespace(author=FakeAuthor())


@pytest.fixture
def attachment():
    return SimpleNamespace(url="https://cdn.example.invalid/image.png")


def make_view(pi, **kwargs):
    """Build an InspectAttachmentView.

    py-cord 2.8's ui.Item.__init__ calls asyncio.get_running_loop(), so a View
    cannot be constructed outside a running loop. In the bot it is always built
    inside an async handler.
    """

    async def build():
        return pi.InspectAttachmentView(**kwargs)

    return asyncio.run(build())
