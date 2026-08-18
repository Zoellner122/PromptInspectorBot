Forked from https://github.com/sALTaccount/PromptInspectorBot

# Prompt Inspector 🔎

[![CI](https://github.com/Zoellner122/PromptInspectorBot/actions/workflows/ci.yml/badge.svg)](https://github.com/Zoellner122/PromptInspectorBot/actions/workflows/ci.yml)

Inspect prompts 🔎 from images uploaded to discord

## Functionality

This Discord bot reacts to any image with generation metadata from Automatic1111's WebUI and ComfyUI.
If generation metadata is detected, a magnifying glass react is added to the image. If the user
clicks the magnifying glass, they are sent a DM with the image generation information.

Metadata is read from PNG text chunks and, for JPEG/WebP uploads, from Exif — A1111 writes its
parameter string to `UserComment`, ComfyUI writes prompt/workflow JSON to `Make`/`ImageDescription`.
Which extensions get inspected is controlled by `SCAN_EXTENSIONS`.

Two context menu apps are provided.
One sends an ephemeral response in the channel.
The other activates the emoji reaction (🔎) DM based response (useful for channels the bot isn't watching).

## Requirements

Python 3.11 or newer (3.11–3.14 are covered by CI).

## Setup

1. Clone the repository
2. Enter the directory
3. Create a venv with `python3 -m venv ./venv`
4. Activate it with `source ./venv/bin/activate` (`.\venv\Scripts\activate` on Windows)
5. Install the dependencies with `pip install -r requirements.txt`
6. Create a Discord bot and invite it to your server
7. Enable the `Message Content Intent` in the Discord developer portal
8. Enable the `Server Members Intent` in the Discord developer portal
9. Create a file named ".env" in the root directory of the project
10. Set `BOT_TOKEN=<your discord bot token>` in the .env file
11. Copy the `config.example.toml` to `config.toml`
12. Add the channel IDs for channels you want the bot to watch, and set the settings you want in the `config.toml` file
13. Run the bot with `python3 PromptInspector.py`

Alternatively, `pip install .` installs the bot and provides a `prompt-inspector` command
equivalent to `python3 PromptInspector.py`.

## Usage

```
prompt-inspector [-c CONFIG] [-d FILE]
```

- `-c`, `--config` — path to the configuration file (default `config.toml`)
- `-d`, `--dump` — print the generation metadata for a local image and exit, without
  connecting to Discord. Useful for checking whether a format is supported.

## Running as a service

`prompt_inspector.service` is a systemd unit for running the bot as a daemon. Adjust the
`User`, `Group` and path settings to match your deployment, then:

```sh
sudo cp prompt_inspector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prompt_inspector
```

## Development

```sh
pip install -e ".[dev]"
pytest              # run the test suite
ruff check .        # lint
ruff format .       # format
```

Tests generate their own image fixtures, so no binary files are committed.

## Examples
![Example 1](images/2023-03-09_00-14.png)
![Example 2](images/2023-03-09_00-14_1.png)
