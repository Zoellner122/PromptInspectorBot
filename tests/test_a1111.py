"""A1111 parameter-string parsing."""

from __future__ import annotations

from collections import OrderedDict

import pytest
from conftest import A1111_PARAMETERS, png_bytes


def test_parses_prompt_negative_and_params(pi):
    params = pi.MetadataA1111(A1111_PARAMETERS).params
    assert params["Prompt"] == "a cat sitting on a windowsill, masterpiece"
    assert params["Negative Prompt"] == "blurry, low quality"
    assert params["Steps"] == "28"
    assert params["Sampler"] == "DPM++ 2M"
    assert params["CFG scale"] == "6.5"
    assert params["Seed"] == "987654321"
    assert params["Size"] == "832x1216"
    assert params["Model"] == "someModel_v3"
    assert params["Denoising strength"] == "0.45"
    assert params["Version"] == "v1.10.1"


def test_prompt_order_is_preserved(pi):
    params = pi.MetadataA1111(A1111_PARAMETERS).params
    assert list(params)[:2] == ["Prompt", "Negative Prompt"]


def test_missing_negative_prompt_is_omitted(pi):
    params = pi.MetadataA1111("just a prompt\nSteps: 10, Seed: 1").params
    assert params["Prompt"] == "just a prompt"
    assert "Negative Prompt" not in params
    assert params["Steps"] == "10"


def test_missing_steps_key_raises(pi):
    with pytest.raises(ValueError, match="missing Steps key"):
        pi.MetadataA1111("no recognisable metadata here")


def test_long_values_are_truncated_to_the_configured_limit(pi):
    pi.CFG.a1111_prompt_size_limit = 20
    long_prompt = "x" * 100
    params = pi.MetadataA1111(f"{long_prompt}\nSteps: 10").params
    assert params["Prompt"] == "x" * 20 + "..."


def test_detected_from_png_text_chunk(pi):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(0, png_bytes(parameters=A1111_PARAMETERS), metadata)
    assert list(metadata) == [0]
    assert metadata[0].NAME == "A1111"
    assert metadata[0].params["Steps"] == "28"


def test_image_without_metadata_is_ignored(pi):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(0, png_bytes(), metadata)
    assert metadata == {}


def test_unrelated_text_chunk_is_ignored(pi):
    metadata = OrderedDict()
    pi.populate_attachment_metadata(0, png_bytes(Comment="hello"), metadata)
    assert metadata == {}
