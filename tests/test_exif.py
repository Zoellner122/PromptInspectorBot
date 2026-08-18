"""Metadata carried in Exif rather than PNG text chunks (JPEG/WebP)."""

from __future__ import annotations

import io
import json
from collections import OrderedDict

import pytest
from conftest import (
    A1111_PARAMETERS,
    exif_comfy_bytes,
    exif_user_comment_bytes,
)
from PIL import Image


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_a1111_parameters_read_from_exif_user_comment(pi, fmt):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(
        0,
        exif_user_comment_bytes(A1111_PARAMETERS, fmt=fmt),
        metadata,
    )
    assert list(metadata) == [0], f"no metadata found in {fmt}"
    assert metadata[0].NAME == "A1111"
    assert metadata[0].params["Steps"] == "28"
    assert metadata[0].params["Prompt"].startswith("a cat sitting on a windowsill")


@pytest.mark.parametrize("encoding", ["utf-16-be", "utf-16-le", "ascii"])
def test_user_comment_encodings(pi, encoding):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(
        0,
        exif_user_comment_bytes(A1111_PARAMETERS, encoding=encoding),
        metadata,
    )
    assert list(metadata) == [0], f"failed to decode {encoding}"
    assert metadata[0].params["Sampler"] == "DPM++ 2M"


def test_comfyui_prompt_read_from_exif(pi, flux_comfy_prompt):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(
        0,
        exif_comfy_bytes(flux_comfy_prompt),
        metadata,
    )
    assert list(metadata) == [0]
    assert metadata[0].NAME == "ComfyUI"
    assert "flux1-dev.safetensors" in json.dumps(metadata[0].params)


def test_png_text_chunks_take_precedence_over_exif(pi):
    """A PNG with both should not pay for an Exif round-trip."""
    from conftest import png_bytes

    data = png_bytes(parameters=A1111_PARAMETERS)
    with Image.open(io.BytesIO(data)) as img:
        chunks = pi.get_image_text_chunks(img)
    assert chunks["parameters"] == A1111_PARAMETERS


def test_image_with_exif_but_no_generation_data_is_ignored(pi):
    img = Image.new("RGB", (8, 8), "gray")
    exif = img.getexif()
    exif[0x010F] = "Canon"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    metadata = OrderedDict()
    pi.populate_attachment_metadata(0, buf.getvalue(), metadata)
    assert metadata == {}


def test_image_with_no_exif_at_all_is_ignored(pi):
    img = Image.new("RGB", (8, 8), "gray")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    metadata = OrderedDict()
    pi.populate_attachment_metadata(0, buf.getvalue(), metadata)
    assert metadata == {}


class TestDecodeUserComment:
    def test_utf16_be_with_prefix(self, pi):
        raw = b"UNICODE\x00" + "hello world".encode("utf-16-be")
        assert pi.decode_user_comment(raw) == "hello world"

    def test_utf16_le_with_prefix(self, pi):
        raw = b"UNICODE\x00" + "hello world".encode("utf-16-le")
        assert pi.decode_user_comment(raw) == "hello world"

    def test_utf16_with_bom(self, pi):
        raw = b"UNICODE\x00" + "hello world".encode("utf-16")
        assert pi.decode_user_comment(raw) == "hello world"

    def test_ascii_prefix(self, pi):
        raw = b"ASCII\x00\x00\x00hello world"
        assert pi.decode_user_comment(raw) == "hello world"

    def test_zero_prefix_treated_as_utf8(self, pi):
        raw = b"\x00" * 8 + "héllo".encode()
        assert pi.decode_user_comment(raw) == "héllo"

    def test_no_recognised_prefix(self, pi):
        assert pi.decode_user_comment(b"plain bytes, no prefix") == (
            "plain bytes, no prefix"
        )

    def test_trailing_nulls_stripped(self, pi):
        raw = b"ASCII\x00\x00\x00hello\x00\x00"
        assert pi.decode_user_comment(raw) == "hello"

    def test_passthrough_str(self, pi):
        assert pi.decode_user_comment("already text") == "already text"

    def test_none_and_empty(self, pi):
        assert pi.decode_user_comment(None) is None
        assert pi.decode_user_comment("") is None

    def test_non_bytes_non_str(self, pi):
        assert pi.decode_user_comment(1234) is None

    def test_non_ascii_prompt_survives_utf16(self, pi):
        text = "一匹の猫, masterpiece"
        raw = b"UNICODE\x00" + text.encode("utf-16-be")
        assert pi.decode_user_comment(raw) == text
