"""The Discord-facing handlers.

These are driven with stand-in context/message objects rather than a live
gateway connection. The point is the branching: what gets responded to, whether
a response is ephemeral, and what happens when a lookup fails.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import A1111_PARAMETERS, FakeAuthor, png_bytes


class FakeAttachment:
    def __init__(self, filename="image.png", data=b"", size=None):
        self.filename = filename
        self.url = f"https://cdn.example.invalid/{filename}"
        self._data = data
        self.size = size if size is not None else max(len(data), 1)
        self.reads = 0

    async def read(self):
        self.reads += 1
        return self._data


class FakeDM:
    def __init__(self, fail_with=None):
        self.sends = []
        self._fail_with = fail_with

    async def send(self, **kwargs):
        if self._fail_with is not None:
            raise self._fail_with
        self.sends.append(kwargs)


class FakeCtxAuthor(FakeAuthor):
    def __init__(self, dm=None, *, bot=False):
        super().__init__()
        self.id = 4242
        self.bot = bot
        self.dm = dm if dm is not None else FakeDM()

    async def create_dm(self):
        return self.dm


class FakeCtx:
    """Stands in for ApplicationContext."""

    def __init__(self, author=None):
        self.author = author or FakeCtxAuthor()
        self.defers = []
        self.responses = []

    async def defer(self, **kwargs):
        self.defers.append(kwargs)

    async def respond(self, *args, **kwargs):
        self.responses.append((args, kwargs))


class FakeChannel:
    def __init__(self, message=None, raise_on_fetch=None):
        self.id = 999
        self._message = message
        self._raise = raise_on_fetch

    async def fetch_message(self, _mid):
        if self._raise is not None:
            raise self._raise
        return self._message


class FakeMessage:
    def __init__(self, attachments=(), channel_id=999, author=None):
        self.attachments = list(attachments)
        self.channel = type("Ch", (), {"id": channel_id})()
        self.author = author or FakeCtxAuthor()
        self.reactions_added = []

    async def add_reaction(self, emoji):
        self.reactions_added.append(emoji)


class FakeReactionEvent:
    def __init__(self, emoji="🔎", channel_id=999, member=None):
        self.emoji = type("E", (), {"name": emoji})()
        self.channel_id = channel_id
        self.message_id = 555
        self.user_id = 4242
        self.member = member


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def a1111_attachment():
    return FakeAttachment("gen.png", png_bytes(parameters=A1111_PARAMETERS))


# --------------------------------------------------------------------------
# message_command_view_prompt: the ephemeral leak regression
# --------------------------------------------------------------------------


def test_view_prompt_responds_only_ephemerally(pi, a1111_attachment):
    """Regression: the first response used to go out publicly.

    collect_attachments defers, so every ctx.respond() is a followup, and
    followups are public unless ephemeral is passed explicitly.
    """
    ctx = FakeCtx()
    message = FakeMessage([a1111_attachment])
    run(pi.message_command_view_prompt.callback(ctx, message))
    assert ctx.responses, "should have responded with the metadata"
    for args, kwargs in ctx.responses:
        assert kwargs.get("ephemeral") is True, f"public response: {args} {kwargs}"


def test_view_prompt_is_ephemeral_for_every_image(pi):
    attachments = [
        FakeAttachment(f"gen{i}.png", png_bytes(parameters=A1111_PARAMETERS))
        for i in range(3)
    ]
    ctx = FakeCtx()
    run(pi.message_command_view_prompt.callback(ctx, FakeMessage(attachments)))
    assert len(ctx.responses) == 3
    assert all(kw.get("ephemeral") is True for _a, kw in ctx.responses)


def test_view_prompt_defers_ephemerally(pi, a1111_attachment):
    ctx = FakeCtx()
    run(pi.message_command_view_prompt.callback(ctx, FakeMessage([a1111_attachment])))
    assert ctx.defers == [{"ephemeral": True}]


def test_view_prompt_reports_when_no_images_match(pi):
    ctx = FakeCtx()
    message = FakeMessage([FakeAttachment("clip.mp4", b"x")])
    run(pi.message_command_view_prompt.callback(ctx, message))
    (args, kwargs) = ctx.responses[0]
    assert "no matching images" in args[0]
    assert kwargs["ephemeral"] is True


def test_view_prompt_reports_when_images_have_no_metadata(pi):
    ctx = FakeCtx()
    message = FakeMessage([FakeAttachment("plain.png", png_bytes())])
    run(pi.message_command_view_prompt.callback(ctx, message))
    (args, kwargs) = ctx.responses[0]
    assert "no image generation data" in args[0]
    assert kwargs["ephemeral"] is True


# --------------------------------------------------------------------------
# message_command_view_prompt_dm
# --------------------------------------------------------------------------


def test_view_prompt_dm_sends_to_the_authors_dm(pi, a1111_attachment):
    ctx = FakeCtx()
    run(
        pi.message_command_view_prompt_dm.callback(ctx, FakeMessage([a1111_attachment]))
    )
    assert len(ctx.author.dm.sends) == 1
    assert "embed" in ctx.author.dm.sends[0]
    assert ctx.responses[-1][0][0] == "DM sent!"


def test_view_prompt_dm_reports_closed_dms(pi, a1111_attachment):
    from discord import Forbidden

    failing = FakeDM(fail_with=Forbidden.__new__(Forbidden))
    ctx = FakeCtx(author=FakeCtxAuthor(dm=failing))
    run(
        pi.message_command_view_prompt_dm.callback(ctx, FakeMessage([a1111_attachment]))
    )
    assert any("Couldn't DM" in a[0] for a, _k in ctx.responses if a)


# --------------------------------------------------------------------------
# on_message
# --------------------------------------------------------------------------


def test_on_message_reacts_when_metadata_is_found(pi, a1111_attachment):
    pi.CFG.monitored_channel_ids = {999}
    message = FakeMessage([a1111_attachment])
    run(pi.on_message(message))
    assert "🔎" in message.reactions_added


def test_on_message_ignores_unmonitored_channels(pi, a1111_attachment):
    pi.CFG.monitored_channel_ids = {111}
    message = FakeMessage([a1111_attachment], channel_id=999)
    run(pi.on_message(message))
    assert message.reactions_added == []
    assert a1111_attachment.reads == 0, "must not download from unwatched channels"


def test_on_message_ignores_bots(pi, a1111_attachment):
    pi.CFG.monitored_channel_ids = {999}
    message = FakeMessage([a1111_attachment], author=FakeCtxAuthor(bot=True))
    run(pi.on_message(message))
    assert message.reactions_added == []


def test_on_message_without_metadata_adds_no_magnifier(pi):
    pi.CFG.monitored_channel_ids = {999}
    message = FakeMessage([FakeAttachment("plain.png", png_bytes())])
    run(pi.on_message(message))
    assert "🔎" not in message.reactions_added


def test_on_message_can_flag_images_without_metadata(pi):
    pi.CFG.monitored_channel_ids = {999}
    pi.CFG.react_on_no_metadata = True
    message = FakeMessage([FakeAttachment("plain.png", png_bytes())])
    run(pi.on_message(message))
    assert "⛔" in message.reactions_added


def test_on_message_skips_unsupported_extensions(pi):
    pi.CFG.monitored_channel_ids = {999}
    att = FakeAttachment("clip.mp4", b"x")
    run(pi.on_message(FakeMessage([att])))
    assert att.reads == 0


def test_on_message_skips_oversized_attachments(pi):
    pi.CFG.monitored_channel_ids = {999}
    pi.CFG.scan_limit_bytes = 10
    att = FakeAttachment("big.png", png_bytes(parameters=A1111_PARAMETERS), size=5000)
    run(pi.on_message(FakeMessage([att])))
    assert att.reads == 0


# --------------------------------------------------------------------------
# on_raw_reaction_add
# --------------------------------------------------------------------------


def test_reaction_with_a_different_emoji_is_ignored(pi, monkeypatch):
    pi.CFG.monitored_channel_ids = {999}
    called = False

    async def spy(_ctx):
        nonlocal called
        called = True

    monkeypatch.setattr(pi, "fetch_reaction_message", spy)
    run(pi.on_raw_reaction_add(FakeReactionEvent(emoji="❤")))
    assert called is False


def test_reaction_in_unmonitored_channel_is_ignored(pi, monkeypatch):
    pi.CFG.monitored_channel_ids = {111}
    called = False

    async def spy(_ctx):
        nonlocal called
        called = True

    monkeypatch.setattr(pi, "fetch_reaction_message", spy)
    run(pi.on_raw_reaction_add(FakeReactionEvent(channel_id=999)))
    assert called is False


def test_bots_own_reaction_is_ignored(pi, monkeypatch):
    pi.CFG.monitored_channel_ids = {999}
    called = False

    async def spy(_ctx):
        nonlocal called
        called = True

    monkeypatch.setattr(pi, "fetch_reaction_message", spy)
    bot_member = FakeCtxAuthor(bot=True)
    run(pi.on_raw_reaction_add(FakeReactionEvent(member=bot_member)))
    assert called is False


def test_reaction_dms_the_reacting_user(pi, monkeypatch, a1111_attachment):
    pi.CFG.monitored_channel_ids = {999}
    message = FakeMessage([a1111_attachment])
    dm = FakeDM()

    async def fake_fetch(_ctx):
        return message

    async def fake_fetch_user(_uid):
        return FakeCtxAuthor(dm=dm)

    monkeypatch.setattr(pi, "fetch_reaction_message", fake_fetch)
    monkeypatch.setattr(pi.client, "fetch_user", fake_fetch_user)
    run(pi.on_raw_reaction_add(FakeReactionEvent()))
    assert len(dm.sends) == 1
    assert "🔎" in message.reactions_added


def test_reaction_with_unresolvable_message_stops_quietly(pi, monkeypatch):
    pi.CFG.monitored_channel_ids = {999}

    async def fake_fetch(_ctx):
        return None

    monkeypatch.setattr(pi, "fetch_reaction_message", fake_fetch)
    run(pi.on_raw_reaction_add(FakeReactionEvent()))  # must not raise


# --------------------------------------------------------------------------
# fetch_reaction_message
# --------------------------------------------------------------------------


def test_fetch_returns_none_for_uncached_and_unfetchable_channel(pi, monkeypatch):
    from discord import InvalidData

    monkeypatch.setattr(pi.client, "get_channel", lambda _cid: None)

    async def boom(_cid):
        raise InvalidData

    monkeypatch.setattr(pi.client, "fetch_channel", boom)
    assert run(pi.fetch_reaction_message(FakeReactionEvent())) is None


def test_fetch_returns_none_for_non_messageable_channel(pi, monkeypatch):
    monkeypatch.setattr(pi.client, "get_channel", lambda _cid: object())
    assert run(pi.fetch_reaction_message(FakeReactionEvent())) is None


def test_fetch_returns_none_when_the_message_is_gone(pi, monkeypatch):
    from discord import NotFound
    from discord.abc import Messageable

    class Ch(Messageable):
        async def _get_channel(self):
            return self

        async def fetch_message(self, _mid):
            raise NotFound.__new__(NotFound)

    monkeypatch.setattr(pi.client, "get_channel", lambda _cid: Ch())
    assert run(pi.fetch_reaction_message(FakeReactionEvent())) is None
