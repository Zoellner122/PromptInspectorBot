"""Formatting of the "Full Parameters" response.

Discord's code blocks are fragile: the content must not contain a fence of its
own, the closing fence needs its own line, and the whole message has to fit in
2000 characters. Anything that cannot satisfy those goes out as a file instead.
"""

from __future__ import annotations

import json

from conftest import A1111_PARAMETERS, make_view


def view(pi, extension="txt", text=""):
    return make_view(pi, text_metadata=text, content_extension=extension)


class TestComfyDisplayText:
    def test_compact_json_is_indented(self, pi, flux_comfy_prompt):
        compact = json.dumps(flux_comfy_prompt)
        assert "\n" not in compact, "fixture should start out minified"
        out = pi.MetadataComfyUI(compact, None).get_display_text()
        assert out.count("\n") > 20
        assert json.loads(out) == flux_comfy_prompt

    def test_no_line_is_absurdly_long(self, pi, flux_comfy_prompt):
        out = pi.MetadataComfyUI(json.dumps(flux_comfy_prompt), None).get_display_text()
        # The prompt text itself is the longest single value in the fixture.
        assert max(len(line) for line in out.splitlines()) < 120

    def test_non_ascii_is_not_escaped(self, pi):
        graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "一匹の猫"}}}
        out = pi.MetadataComfyUI(json.dumps(graph), None).get_display_text()
        assert "一匹の猫" in out
        assert "\\u" not in out

    def test_falls_back_to_raw_when_unparseable(self, pi, flux_comfy_prompt):
        md = pi.MetadataComfyUI(json.dumps(flux_comfy_prompt), None)
        md.text_metadata = "{not valid json"
        assert md.get_display_text() == "{not valid json"


class TestA1111DisplayText:
    def test_is_the_raw_parameter_string(self, pi):
        md = pi.MetadataA1111(A1111_PARAMETERS)
        assert md.get_display_text() == A1111_PARAMETERS


class TestFormatCodeBlock:
    def test_wraps_with_language_tag(self, pi):
        block = view(pi, "txt").format_code_block("a cat\nSteps: 20")
        assert block.startswith("```plaintext\n")

    def test_json_extension_uses_json_tag(self, pi):
        block = view(pi, "json").format_code_block("{}")
        assert block.startswith("```json\n")

    def test_unknown_extension_falls_back_to_plaintext(self, pi):
        block = view(pi, "xyz").format_code_block("data")
        assert block.startswith("```plaintext\n")

    def test_closing_fence_is_on_its_own_line(self, pi):
        block = view(pi, "txt").format_code_block("a cat")
        assert block.endswith("\n```")

    def test_trailing_backtick_does_not_merge_into_the_fence(self, pi):
        block = view(pi, "txt").format_code_block("prompt ending in a tick `")
        assert block.endswith("`\n```")
        assert block.count("```") == 2

    def test_fence_count_stays_even(self, pi):
        block = view(pi, "txt").format_code_block("a cat, masterpiece")
        assert block.count("```") == 2

    def test_content_with_a_fence_is_rejected(self, pi):
        assert view(pi, "txt").format_code_block("a cat ``` Steps: 20") is None

    def test_overlong_content_is_rejected(self, pi):
        assert view(pi, "txt").format_code_block("x" * 2500) is None

    def test_content_just_inside_the_limit_is_accepted(self, pi):
        v = view(pi, "json")
        overhead = len("```json\n") + len("\n```")
        block = v.format_code_block("x" * (v.DISCORD_MESSAGE_LIMIT - overhead))
        assert block is not None
        assert len(block) == v.DISCORD_MESSAGE_LIMIT

    def test_content_one_over_the_limit_is_rejected(self, pi):
        v = view(pi, "json")
        overhead = len("```json\n") + len("\n```")
        assert (
            v.format_code_block("x" * (v.DISCORD_MESSAGE_LIMIT - overhead + 1)) is None
        )


def test_view_receives_the_display_text_not_the_raw_json(
    pi, msg_ctx, flux_comfy_prompt
):
    """get_embed_view must hand the pretty text to the button."""
    import asyncio

    md = pi.MetadataComfyUI(json.dumps(flux_comfy_prompt), None)

    async def build():
        return md.get_embed_view(msg_ctx)

    _embed, v = asyncio.run(build())
    assert "\n" in v.text_metadata
    assert json.loads(v.text_metadata) == flux_comfy_prompt


class FakeResponse:
    def __init__(self):
        self.edits = []

    async def edit_message(self, **kwargs):
        self.edits.append(kwargs)


class FakeFollowup:
    def __init__(self):
        self.sends = []

    async def send(self, content=None, **kwargs):
        self.sends.append((content, kwargs))


class FakeInteraction:
    def __init__(self):
        self.response = FakeResponse()
        self.followup = FakeFollowup()


def press_button(pi, text, extension):
    """Invoke the Full Parameters button and return (view, interaction)."""
    import asyncio

    async def run():
        # Built inline rather than via make_view: we are already inside the loop
        # the View constructor needs.
        v = pi.InspectAttachmentView(
            text_metadata=text,
            content_extension=extension,
        )
        interaction = FakeInteraction()
        await v.children[0].callback(interaction)
        return v, interaction

    return asyncio.run(run())


def test_button_is_disabled_after_pressing(pi):
    v, interaction = press_button(pi, "a cat\nSteps: 20", "txt")
    assert v.children[0].disabled is True
    assert interaction.response.edits, "the view should be edited to show it disabled"


def test_small_graph_is_sent_as_an_indented_code_block(pi, flux_comfy_prompt):
    pretty = pi.MetadataComfyUI(json.dumps(flux_comfy_prompt), None).get_display_text()
    _v, interaction = press_button(pi, pretty, "json")
    content, kwargs = interaction.followup.sends[0]
    assert "file" not in kwargs
    assert content.startswith("```json\n")
    assert content.endswith("\n```")
    assert content.count("```") == 2
    assert max(len(line) for line in content.splitlines()) < 120


def test_large_graph_is_sent_as_an_indented_file(pi):
    big = {
        str(i): {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a very detailed description " * 5},
        }
        for i in range(30)
    }
    pretty = pi.MetadataComfyUI(json.dumps(big), None).get_display_text()
    _v, interaction = press_button(pi, pretty, "json")
    content, kwargs = interaction.followup.sends[0]
    assert content is None
    f = kwargs["file"]
    assert f.filename == "parameters.json"
    f.reset()
    body = f.fp.read()
    assert json.loads(body) == big
    assert body.lstrip().startswith("{\n"), "the attached file should be indented too"


def test_content_with_a_fence_is_sent_as_a_file_unmangled(pi):
    raw = "a cat ``` Steps: 20"
    _v, interaction = press_button(pi, raw, "txt")
    content, kwargs = interaction.followup.sends[0]
    assert content is None
    f = kwargs["file"]
    f.reset()
    assert f.fp.read() == raw, "the text must not be altered to fit a code block"


def test_empty_metadata_reports_instead_of_sending_an_empty_block(pi):
    _v, interaction = press_button(pi, "", "txt")
    content, kwargs = interaction.followup.sends[0]
    assert content == "No metadata to send!"
    assert "file" not in kwargs
