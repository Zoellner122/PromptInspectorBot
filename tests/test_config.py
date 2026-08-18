"""Config loading and attachment filtering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def write_config(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def test_defaults_are_applied(pi):
    assert pi.CFG.scan_limit_bytes == 10 * 1024**2
    assert pi.CFG.monitored_channel_ids == set()
    assert pi.CFG.log_level == "INFO"
    assert "png" in pi.CFG.scan_extensions


def test_example_config_matches_code_defaults_for_scan_limit(pi):
    """config.example.toml must not disagree with the in-code default."""
    import tomllib
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "config.example.toml"
    with example.open("rb") as fp:
        cfg = tomllib.load(fp)
    assert cfg["SCAN_LIMIT_BYTES"] == pi.CFG.scan_limit_bytes


def test_load_overrides_defaults(pi, tmp_path):
    path = write_config(
        tmp_path,
        """
MONITORED_CHANNEL_IDS = [111, 222, 222]
SCAN_LIMIT_BYTES = 4096
LOG_LEVEL = "DEBUG"
COMFYUI_SHOW_TITLES = false
""",
    )
    pi.CFG.load(path)
    assert pi.CFG.monitored_channel_ids == {111, 222}
    assert pi.CFG.scan_limit_bytes == 4096
    assert pi.CFG.log_level == "DEBUG"
    assert pi.CFG.comfyui_show_titles is False


def test_load_leaves_unspecified_fields_at_default(pi, tmp_path):
    path = write_config(tmp_path, "MONITORED_CHANNEL_IDS = [1]\n")
    pi.CFG.load(path)
    assert pi.CFG.scan_limit_bytes == 10 * 1024**2
    assert pi.CFG.attach_file_size_threshold == 1980


def test_load_accepts_str_path(pi, tmp_path):
    path = write_config(tmp_path, "SCAN_LIMIT_BYTES = 123\n")
    pi.CFG.load(str(path))
    assert pi.CFG.scan_limit_bytes == 123


def test_example_config_is_loadable(pi):
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "config.example.toml"
    pi.CFG.load(example)
    assert pi.CFG.monitored_channel_ids == {1234}


def fake_message(*attachments):
    return SimpleNamespace(
        attachments=[
            SimpleNamespace(filename=name, size=size) for name, size in attachments
        ],
    )


@pytest.mark.parametrize(
    "filename",
    ["a.png", "A.PNG", "b.jpg", "c.jpeg", "d.webp", "e.WebP"],
)
def test_supported_extensions_are_scanned(pi, filename):
    message = fake_message((filename, 1000))
    assert len(pi.scannable_attachments(message)) == 1


@pytest.mark.parametrize("filename", ["a.gif", "b.txt", "c.mp4", "d.png.txt", "noext"])
def test_unsupported_extensions_are_skipped(pi, filename):
    message = fake_message((filename, 1000))
    assert pi.scannable_attachments(message) == []


def test_oversized_attachments_are_skipped(pi):
    pi.CFG.scan_limit_bytes = 500
    message = fake_message(("small.png", 499), ("big.png", 501))
    assert [a.filename for a in pi.scannable_attachments(message)] == ["small.png"]


def test_extension_list_is_configurable(pi):
    pi.CFG.scan_extensions = ("png",)
    message = fake_message(("a.png", 10), ("b.webp", 10))
    assert [a.filename for a in pi.scannable_attachments(message)] == ["a.png"]


def test_extensions_may_be_written_with_leading_dot(pi):
    pi.CFG.scan_extensions = (".PNG",)
    message = fake_message(("a.png", 10))
    assert len(pi.scannable_attachments(message)) == 1


def test_setup_logging_rejects_bad_level(pi):
    pi.CFG.log_level = "NOPE"
    with pytest.raises(ValueError, match="Invalid log_level"):
        pi.setup_logging()


def test_dump_reports_unreadable_files_cleanly(pi, tmp_path, capsys):
    """`--dump` on a non-image should explain itself, not raise a traceback."""
    bad = tmp_path / "not-an-image.txt"
    bad.write_text("definitely not a png")
    with pytest.raises(SystemExit) as exc:
        pi.handle_check(bad)
    assert exc.value.code == 1
    assert "Not an image file" in capsys.readouterr().out


def test_dump_reports_images_without_metadata(pi, tmp_path, capsys):
    from conftest import png_bytes

    img = tmp_path / "plain.png"
    img.write_bytes(png_bytes())
    pi.handle_check(img)
    assert "No metadata" in capsys.readouterr().out
