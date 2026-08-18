"""Embed construction, including Discord's hard size limits.

Discord rejects the whole message with a 400 if an embed's combined character
count exceeds 6000 or if any field value is empty, so these are correctness
constraints rather than cosmetic ones.
"""

from __future__ import annotations

import json

from conftest import A1111_PARAMETERS, DISCORD_EMBED_TOTAL_LIMIT


def embed_len(embed) -> int:
    """Mirror of Discord's accounting, matching discord.Embed.__len__."""
    total = len(embed.title or "") + len(embed.description or "")
    total += len(getattr(embed.footer, "text", "") or "")
    total += len(getattr(embed.author, "name", "") or "")
    for f in embed.fields:
        total += len(f.name or "") + len(f.value or "")
    return total


class DictMetadata:
    """Build a Metadata subclass over a fixed params dict."""

    @staticmethod
    def make(pi, params: dict[str, str]):
        class _M(pi.Metadata):
            def get_params_from_string(self, s):  # noqa: ARG002
                return dict(params)

        return _M("")


def test_a1111_embed_within_limits(pi, msg_ctx, attachment):
    embed = pi.MetadataA1111(A1111_PARAMETERS).get_embed(msg_ctx, attachment=attachment)
    assert len(embed.fields) <= pi.CFG.message_embed_limit
    assert embed_len(embed) <= DISCORD_EMBED_TOTAL_LIMIT
    assert embed.image.url == attachment.url


def test_many_long_comfy_text_nodes_stay_within_total_limit(pi, msg_ctx):
    """The case that previously produced a >26000 character embed."""
    graph = {
        str(i): {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": f"Prompt {i}"},
            "inputs": {"text": "a very detailed description " * 40},
        }
        for i in range(30)
    }
    embed = pi.MetadataComfyUI(json.dumps(graph), None).get_embed(msg_ctx)
    assert embed_len(embed) <= DISCORD_EMBED_TOTAL_LIMIT
    assert embed.fields, "should still surface as many fields as fit"


def test_a1111_with_prompt_cap_disabled_stays_within_total_limit(pi, msg_ctx):
    pi.CFG.a1111_prompt_size_limit = 10**6
    tags = ", ".join(f"tag{i} intricate detailed" for i in range(500))
    params = f"{tags}\nNegative prompt: {tags}\nSteps: 30, Sampler: X, Seed: 1"
    embed = pi.MetadataA1111(params).get_embed(msg_ctx)
    assert embed_len(embed) <= DISCORD_EMBED_TOTAL_LIMIT


def test_no_field_value_exceeds_the_per_field_cap(pi, msg_ctx):
    md = DictMetadata.make(pi, {"Big": "y" * 5000})
    embed = md.get_embed(msg_ctx)
    assert all(len(f.value) <= 1024 for f in embed.fields)


def test_truncated_values_are_marked_with_an_ellipsis(pi, msg_ctx):
    md = DictMetadata.make(pi, {"Big": "y" * 5000})
    embed = md.get_embed(msg_ctx)
    assert embed.fields[0].value.endswith("...")


def test_empty_field_values_are_dropped(pi, msg_ctx):
    md = DictMetadata.make(pi, {"Empty": "", "Good": "value"})
    embed = md.get_embed(msg_ctx)
    assert [f.name for f in embed.fields] == ["Good"]
    assert all(f.value for f in embed.fields)


def test_field_count_limit_is_honoured(pi, msg_ctx):
    pi.CFG.message_embed_limit = 3
    md = DictMetadata.make(pi, dict.fromkeys("abcdefgh", "v"))
    assert len(md.get_embed(msg_ctx).fields) == 3


def test_prioritised_fields_come_first_in_the_given_order(pi, msg_ctx):
    md = DictMetadata.make(pi, {"a": "1", "b": "2", "c": "3", "d": "4"})
    embed = md.get_embed(msg_ctx, prioritize_fields=("c", "a"))
    assert [f.name for f in embed.fields] == ["c", "a", "b", "d"]


def test_absent_prioritised_field_is_skipped(pi, msg_ctx):
    md = DictMetadata.make(pi, {"a": "1", "b": "2"})
    embed = md.get_embed(msg_ctx, prioritize_fields=("nope", "b"))
    assert [f.name for f in embed.fields] == ["b", "a"]


def test_inline_only_for_short_non_prompt_values(pi, msg_ctx):
    md = DictMetadata.make(pi, {"Seed": "123", "Prompt": "a cat"})
    flags = {f.name: f.inline for f in md.get_embed(msg_ctx).fields}
    assert flags == {"Seed": True, "Prompt": False}


def test_comfyui_never_inlines(pi, msg_ctx):
    graph = {"1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}}
    embed = pi.MetadataComfyUI(json.dumps(graph), None).get_embed(msg_ctx)
    assert all(f.inline is False for f in embed.fields)


def test_footer_names_the_poster(pi, msg_ctx):
    embed = pi.MetadataA1111(A1111_PARAMETERS).get_embed(msg_ctx)
    assert embed.footer.text == f"Posted by {msg_ctx.author}"


class TestFitFieldValue:
    def test_short_value_untouched(self, pi):
        assert pi.Metadata.fit_field_value("hello", 1000) == "hello"

    def test_truncates_to_budget(self, pi):
        out = pi.Metadata.fit_field_value("y" * 500, 100)
        assert len(out) == 100
        assert out.endswith("...")

    def test_truncates_to_per_field_cap(self, pi):
        out = pi.Metadata.fit_field_value("y" * 5000, 10**6)
        assert len(out) == 1024

    def test_returns_none_when_budget_exhausted(self, pi):
        assert pi.Metadata.fit_field_value("hello", 4) is None

    def test_returns_none_on_negative_budget(self, pi):
        assert pi.Metadata.fit_field_value("hello", -50) is None
