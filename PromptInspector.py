from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
import tomllib
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from discord import (
    ApplicationContext,
    Attachment,
    ButtonStyle,
    Embed,
    File,
    HTTPException,
    Intents,
    InvalidData,
    Message,
    RawReactionActionEvent,
)
from discord.abc import Messageable
from discord.ext import commands
from discord.ui import View, button
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Sequence

load_dotenv()

log = logging.getLogger("PromptInspector")


class __f:  # noqa: N801
    def __init__(self, fmt, /, *args: Sequence[Any], **kwargs: dict[Any, Any]):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


class Config:
    class _EmptyValue:
        pass

    FIELDS = (
        ("monitored_channel_ids", set()),
        ("scan_limit_bytes", 10 * 1024**2),  # Default 10 MB
        ("scan_extensions", ("png", "jpg", "jpeg", "webp")),
        (
            "a1111_important_fields",
            (
                "Prompt",
                "Negative Prompt",
                "Steps",
                "Sampler",
                "CFG scale",
                "Seed",
                "Size",
                "Model",
                "VAE",
                "Denoising strength",
                "Hires upscale",
                "Hires steps",
                "Hires upscaler",
                "Version",
            ),
        ),
        ("a1111_prompt_size_limit", 1000),
        ("comfyui_show_titles", True),
        ("comfyui_extract_widget_values", True),
        ("comfyui_prioritize_nodes_with_text", True),
        ("comfyui_replace_field_newlines", True),
        ("message_embed_limit", 25),
        ("attach_file_size_threshold", 1980),
        ("react_on_no_metadata", False),
        ("log_color", False),
        ("log_level", "INFO"),
    )

    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        for k, v in self.FIELDS:
            setattr(self, k, v)

    def load(self, filepath: Path | str = "config.toml"):
        empty = self._EmptyValue
        with Path(filepath).open("rb") as fp:
            cfg = tomllib.load(fp)
        for k, _v in self.FIELDS:
            cfgval = cfg.get(k.upper(), empty)
            if cfgval is not empty:
                if k == "monitored_channel_ids":
                    cfgval = set(cfgval)
                setattr(self, k, cfgval)


CFG = Config()

intents = Intents.default() | Intents.message_content | Intents.members
client = commands.Bot(intents=intents)


class InspectAttachmentView(View):
    TXTBLOCK_TYPES = MappingProxyType(
        {"txt": "plaintext", "json": "json", "yaml": "yaml"},
    )
    CODE_FENCE = "```"
    # Discord rejects a message whose content exceeds 2000 characters, and the
    # fence and language tag count toward that.
    DISCORD_MESSAGE_LIMIT = 2000

    def __init__(
        self,
        timeout=3600,
        text_metadata: str | None = None,
        content_type="text/plain",
        content_extension="txt",
        **kwargs: dict[str, Any],
    ):
        super().__init__(timeout=timeout, disable_on_timeout=True)
        if text_metadata is not None:
            text_metadata = text_metadata.strip()
        self.text_metadata = text_metadata
        self.content_type = content_type
        self.content_extension = content_extension
        self.kwargs = kwargs

    def format_code_block(self, text: str) -> str | None:
        """Wrap `text` in a Discord code block, or None if it needs to be a file.

        Returns None when the text cannot be represented inline: either it
        contains a fence of its own, which would terminate the block early and
        spill the rest as normal markdown, or the wrapped result would exceed
        Discord's message length limit.
        """
        if self.CODE_FENCE in text:
            return None
        lang = self.TXTBLOCK_TYPES.get(self.content_extension, "plaintext")
        # The closing fence needs its own line, otherwise text ending in a
        # backtick merges into it and the block never closes.
        block = f"{self.CODE_FENCE}{lang}\n{text}\n{self.CODE_FENCE}"
        if len(block) > self.DISCORD_MESSAGE_LIMIT:
            return None
        return block

    @button(label="Full Parameters", style=ButtonStyle.green)
    async def details(self, button, interaction):
        button.disabled = True
        await interaction.response.edit_message(view=self)
        if not self.text_metadata:
            await interaction.followup.send("No metadata to send!", **self.kwargs)
            return
        if len(self.text_metadata) <= CFG.attach_file_size_threshold:
            block = self.format_code_block(self.text_metadata)
            if block is not None:
                await interaction.followup.send(block, **self.kwargs)
                return
        # Too long, or not safe to inline: send the untouched text as a file.
        with io.StringIO() as f:
            f.write(self.text_metadata)
            f.seek(0)
            await interaction.followup.send(
                file=File(f, f"parameters.{self.content_extension}"),
                **self.kwargs,
            )


class Metadata:
    NAME = "Unknown"
    CONTENT_TYPE = "text/plain"
    EXTENSION = "txt"
    ALLOW_INLINE_EMBEDS = True

    # Discord protocol limits, not preferences: an embed whose title, footer and
    # field names/values together exceed 6000 characters is rejected outright
    # with a 400, and an empty field value is rejected too. `message_embed_limit`
    # caps the field *count*, which on its own is not enough: 25 fields of 1024
    # characters is four times over the total budget.
    EMBED_TOTAL_LIMIT = 6000
    EMBED_FIELD_VALUE_LIMIT = 1024
    # Below this there is no point adding a field, only an ellipsis would fit.
    EMBED_MIN_FIELD_VALUE = 16
    ELLIPSIS = "..."

    def __init__(self, s):
        self.text_metadata = s
        self.params = self.get_params_from_string(s)

    def get_params_from_string(self, *args: Sequence[Any], **kwargs: dict[Any, Any]):
        raise NotImplementedError

    def get_display_text(self) -> str:
        """Return the full metadata as it should be shown to the user."""
        return self.text_metadata

    def get_embed_view(self, msg_ctx: Message, attachment=None, ephemeral=False):
        embed = self.get_embed(msg_ctx, attachment=attachment)
        kwargs = {"ephemeral": True} if ephemeral else {}
        view = InspectAttachmentView(
            text_metadata=self.get_display_text(),
            content_type=self.CONTENT_TYPE,
            content_extension=self.EXTENSION,
            **kwargs,
        )
        return embed, view

    @classmethod
    def fit_field_value(cls, value: str, budget: int) -> str | None:
        """Trim `value` to the per-field cap and the embed's remaining budget.

        Returns None when not even a truncated value would be worth adding.
        """
        limit = min(cls.EMBED_FIELD_VALUE_LIMIT, budget)
        if limit < cls.EMBED_MIN_FIELD_VALUE:
            return None
        if len(value) <= limit:
            return value
        return value[: limit - len(cls.ELLIPSIS)] + cls.ELLIPSIS

    def get_embed(
        self,
        msg_ctx: Message,
        attachment=None,
        prioritize_fields: tuple[str, ...] = (),
    ):
        embed_dict = self.params | {}
        title = f"{self.NAME} Parameters"
        footer = f"Posted by {msg_ctx.author}"
        embed = Embed(title=title, color=msg_ctx.author.color)
        # Title and footer count toward the same 6000 character budget as the
        # fields, so reserve them up front.
        budget = self.EMBED_TOTAL_LIMIT - len(title) - len(footer)
        # Prioritised keys first, in the order given, then whatever is left in
        # its original order.
        ordered = [(k, embed_dict.pop(k)) for k in prioritize_fields if k in embed_dict]
        ordered += list(embed_dict.items())
        count = 0
        for key, value in ordered:
            if count >= CFG.message_embed_limit:
                break
            if not value:
                # Discord rejects an embed containing an empty field value.
                continue
            value_str = self.fit_field_value(value, budget - len(key))
            if value_str is None:
                # Out of budget. Everything after this would not fit either.
                break
            embed.add_field(
                name=key,
                value=value_str,
                inline=self.ALLOW_INLINE_EMBEDS
                and "Prompt" not in key
                and len(value) < 32,
            )
            budget -= len(key) + len(value_str)
            count += 1
        embed.set_footer(text=footer, icon_url=msg_ctx.author.display_avatar)
        if attachment is not None:
            embed.set_image(url=attachment.url)
        return embed


class MetadataA1111(Metadata):
    NAME = "A1111"
    CONTENT_TYPE = "text/plain"
    EXTENSION = "txt"
    # The parameter block starts at this key, and get_params_from_string splits
    # on it. Detection must test for exactly the same string, otherwise a near
    # miss like "Steps:20" is detected as A1111 and then fails to parse.
    STEPS_KEY = "Steps: "

    def get_embed(self, msg_ctx: Message, attachment=None):
        return super().get_embed(
            msg_ctx,
            attachment=attachment,
            prioritize_fields=CFG.a1111_important_fields,
        )

    def get_params_from_string(self, param_str: str) -> OrderedDict[str, str]:
        # TODO: try to adapt A1111's own metadata parsing code: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cf2772fab0af5573da775e7437e6acdca424f26e/modules/generation_parameters_copypaste.py#L211
        max_prompt = CFG.a1111_prompt_size_limit
        output_dict = OrderedDict()
        parts = param_str.split(self.STEPS_KEY, 1)
        if len(parts) != 2:
            raise ValueError("Can't parse A1111 metadata: missing Steps key")
        prompts = parts[0]
        params = self.STEPS_KEY + parts[1]
        neg_parts = (
            prompts.split("Negative prompt: ", 1)
            if "Negative prompt: " in prompts
            else ()
        )
        if neg_parts:
            output_dict["Prompt"] = neg_parts[0].strip()
            output_dict["Negative Prompt"] = neg_parts[1].strip()
        else:
            output_dict["Prompt"] = prompts.strip()
        param_list = params.split(", ")
        for param in param_list:
            kv = param.split(": ", 1)
            if len(kv) == 2:
                output_dict[kv[0].strip()] = kv[1].strip()
        for k, v in output_dict.items():
            if len(v) > max_prompt:
                output_dict[k] = v[:max_prompt] + "..."
        return output_dict


class MetadataComfyUI(Metadata):
    NAME = "ComfyUI"
    CONTENT_TYPE = "application/json"
    EXTENSION = "json"
    ALLOW_INLINE_EMBEDS = False

    # ComfyUI serialises workflows as JSON, so a float-valued widget such as
    # `cfg: 8` round-trips as an int and would be silently dropped by a bare
    # isinstance(val, float) check. Use this for any numeric widget.
    NUMBER = (int, float)

    # This is just a read-only dict.
    COMFY_HANDLERS = MappingProxyType(
        {
            # Handler format: input name, input type, [optional widget name]
            # When widget name is present and conforms to the expected type its value will
            # replace the node input value.
            #
            # --- Core checkpoint / model / VAE loaders ---
            "checkpointloadersimple": (("ckpt_name", str),),
            "checkpointloader": (("ckpt_name", str),),
            "imageonlycheckpointloader": (("ckpt_name", str),),
            "unclipcheckpointloader": (("ckpt_name", str),),
            "checkpointloadernf4": (("ckpt_name", str),),
            "vaeloader": (("vae_name", str),),
            "unetloader": (("unet_name", str), ("weight_dtype", str)),
            "diffusersloader": (("model_path", str),),
            # GGUF quantised loaders (city96/ComfyUI-GGUF)
            "unetloadergguf": (("unet_name", str),),
            "unetloadergguf/advanced": (("unet_name", str), ("dequant_dtype", str)),
            "cliploadergguf": (("clip_name", str), ("type", str)),
            "dualcliploadergguf": (
                ("clip_name1", str),
                ("clip_name2", str),
                ("type", str),
            ),
            "triplecliploadergguf": (
                ("clip_name1", str),
                ("clip_name2", str),
                ("clip_name3", str),
            ),
            # --- CLIP / text encoder loaders ---
            "cliploader": (("clip_name", str), ("type", str)),
            "dualcliploader": (("clip_name1", str), ("clip_name2", str), ("type", str)),
            "triplecliploader": (
                ("clip_name1", str),
                ("clip_name2", str),
                ("clip_name3", str),
            ),
            "quadruplecliploader": (
                ("clip_name1", str),
                ("clip_name2", str),
                ("clip_name3", str),
                ("clip_name4", str),
            ),
            "clipvisionloader": (("clip_name", str),),
            "clipsetlastlayer": (("stop_at_clip_layer", int),),
            # --- LoRA / hypernetwork / style ---
            "loraloader": (
                ("lora_name", str),
                ("strength_model", NUMBER),
                ("strength_clip", NUMBER),
            ),
            "loraloadermodelonly": (("lora_name", str), ("strength_model", NUMBER)),
            "loraloader|pysssss": (
                ("lora_name", str),
                ("strength_model", NUMBER),
                ("strength_clip", NUMBER),
            ),
            "lora loader": (  # WAS Node Suite
                ("lora_name", str),
                ("lora_model_strength", NUMBER),
                ("lora_clip_strength", NUMBER),
            ),
            "hypernetworkloader": (("hypernetwork_name", str), ("strength", NUMBER)),
            "stylemodelloader": (("style_model_name", str),),
            "gligenloader": (("gligen_name", str),),
            # --- Text encoding ---
            "cliptextencode": (("text", str),),
            "cliptextencodesdxl": (("text_l", str), ("text_g", str)),
            "cliptextencodesdxlrefiner": (("text", str), ("ascore", NUMBER)),
            "cliptextencodeflux": (
                ("clip_l", str),
                ("t5xxl", str),
                ("guidance", NUMBER),
            ),
            "cliptextencodesd3": (
                ("clip_l", str),
                ("clip_g", str),
                ("t5xxl", str),
                ("empty_padding", str),
            ),
            "cliptextencodelumina2": (("system_prompt", str), ("user_prompt", str)),
            "cliptextencodehunyuandit": (("bert", str), ("mt5xl", str)),
            "textencodeqwenimageedit": (("prompt", str),),
            "cliptextencodeperpweight": (("text", str),),
            "bnk_cliptextencodeadvanced": (("text", str),),
            "bnk_cliptextencodesdxladvanced": (("text", str),),
            "editableclipencode": (("text", str),),
            "text multiline": (("text", str),),
            "promptcontrolsimple": (("positive", str), ("negative", str)),
            # --- Guidance / model sampling patches ---
            "fluxguidance": (("guidance", NUMBER),),
            "modelsamplingflux": (
                ("max_shift", NUMBER),
                ("base_shift", NUMBER),
                ("width", int),
                ("height", int),
            ),
            "modelsamplingsd3": (("shift", NUMBER),),
            "modelsamplingauraflow": (("shift", NUMBER),),
            "modelsamplingdiscrete": (("sampling", str),),
            "rescalecfg": (("multiplier", NUMBER),),
            "perturbedattentionguidance": (("scale", NUMBER),),
            "selfattentionguidance": (("scale", NUMBER), ("blur_sigma", NUMBER)),
            "freeu": (
                ("b1", NUMBER),
                ("b2", NUMBER),
                ("s1", NUMBER),
                ("s2", NUMBER),
            ),
            "freeu_v2": (
                ("b1", NUMBER),
                ("b2", NUMBER),
                ("s1", NUMBER),
                ("s2", NUMBER),
            ),
            "cfgguider": (("cfg", NUMBER),),
            # --- Latents ---
            "emptylatentimage": (
                ("width", int),
                ("height", int),
                ("batch_size", int),
            ),
            "emptysd3latentimage": (
                ("width", int),
                ("height", int),
                ("batch_size", int),
            ),
            "emptyhunyuanlatentvideo": (
                ("width", int),
                ("height", int),
                ("length", int),
                ("batch_size", int),
            ),
            "emptymochilatentvideo": (
                ("width", int),
                ("height", int),
                ("length", int),
                ("batch_size", int),
            ),
            "emptyltxvlatentvideo": (
                ("width", int),
                ("height", int),
                ("length", int),
                ("batch_size", int),
            ),
            "wanimagetovideo": (
                ("width", int),
                ("height", int),
                ("length", int),
                ("batch_size", int),
            ),
            "latentupscale": (
                ("upscale_method", str),
                ("width", int),
                ("height", int),
                ("crop", str),
            ),
            "latentupscaleby": (("upscale_method", str), ("scale_by", NUMBER)),
            # --- Image scaling / upscaling ---
            "imagescale": (
                ("upscale_method", str),
                ("width", int),
                ("height", int),
                ("crop", str),
            ),
            "imagescaleby": (("upscale_method", str), ("scale_by", NUMBER)),
            "upscalemodelloader": (("model_name", str),),
            # --- ControlNet ---
            "controlnetloader": (("control_net_name", str),),
            "diffcontrolnetloader": (("control_net_name", str),),
            "controlnetapply": (("strength", NUMBER),),
            "controlnetapplyadvanced": (
                ("strength", NUMBER),
                ("start_percent", NUMBER),
                ("end_percent", NUMBER),
            ),
            # --- Samplers: classic ---
            "ksampler": (
                ("seed", int),
                ("steps", int),
                ("cfg", NUMBER),
                ("sampler_name", str),
                ("scheduler", str),
                ("denoise", NUMBER),
            ),
            "ksampleradvanced": (
                ("add_noise", str),
                ("noise_seed", int),
                ("steps", int),
                ("start_at_step", int),
                ("end_at_step", int),
                ("cfg", NUMBER),
                ("sampler_name", str),
                ("scheduler", str),
            ),
            # --- Samplers: custom sampling chain ---
            "samplercustom": (
                ("noise_seed", int),
                ("cfg", NUMBER),
            ),
            "randomnoise": (("noise_seed", int),),
            "ksamplerselect": (("sampler_name", str),),
            "basicscheduler": (
                ("scheduler", str),
                ("steps", int),
                ("denoise", NUMBER),
            ),
            "sdturboscheduler": (("steps", int), ("denoise", NUMBER)),
            "karrasscheduler": (
                ("steps", int),
                ("sigma_max", NUMBER),
                ("sigma_min", NUMBER),
                ("rho", NUMBER),
            ),
            "exponentialscheduler": (
                ("steps", int),
                ("sigma_max", NUMBER),
                ("sigma_min", NUMBER),
            ),
            "polyexponentialscheduler": (
                ("steps", int),
                ("sigma_max", NUMBER),
                ("sigma_min", NUMBER),
                ("rho", NUMBER),
            ),
            "betasamplingscheduler": (
                ("steps", int),
                ("alpha", NUMBER),
                ("beta", NUMBER),
            ),
            "samplereulerancestral": (("eta", NUMBER), ("s_noise", NUMBER)),
            "samplerdpmpp_2m_sde": (
                ("eta", NUMBER),
                ("s_noise", NUMBER),
                ("solver_type", str),
            ),
            "samplerdpmpp_sde": (
                ("eta", NUMBER),
                ("s_noise", NUMBER),
                ("r", NUMBER),
            ),
            # --- Samplers: restart sampling extension ---
            "ksampler with restarts (simple)": (
                ("seed", int),
                ("steps", int),
                ("cfg", NUMBER),
                ("sampler_name", str),
                ("scheduler", str),
            ),
            "ksampler with restarts": (
                ("seed", int),
                ("steps", int),
                ("cfg", NUMBER),
                ("sampler_name", str),
                ("scheduler", str),
                ("restart_scheduler", str),
            ),
            "ksampler with restarts (advanced)": (
                ("noise_seed", int),
                ("steps", int),
                ("cfg", NUMBER),
                ("sampler_name", str),
                ("scheduler", str),
                ("restart_scheduler", str),
            ),
            "ksampler with restarts (custom)": (
                ("noise_seed", int),
                ("steps", int),
                ("cfg", NUMBER),
                ("scheduler", str),
                ("restart_scheduler", str),
            ),
            # --- Third-party bundles ---
            "efficient_loader": (
                ("ckpt_name", str),
                ("vae_name", str),
                ("clip_skip", int),
                ("clip_positive", str),
                ("clip_negative", str),
                ("empty_latent_width", int),
                ("empty_latent_height", int),
            ),
            "ksampler (efficient)": (
                ("seed", int),
                ("steps", int),
                ("cfg", NUMBER),
                ("sampler_name", str),
                ("scheduler", str),
                ("denoise", NUMBER),
            ),
            "checkpointloader|pysssss": (("ckpt_name", str),),
            "checkpoint loader (simple)": (("ckpt_name", str),),  # WAS Node Suite
            "ttn pipeloader": (
                ("ckpt_name", str),
                ("vae_name", str),
                ("clip_skip", int),
                ("positive", str),
                ("negative", str),
                ("empty_latent_width", int),
                ("empty_latent_height", int),
                ("seed", int),
            ),
            "ttn pipeloadersdxl": (
                ("ckpt_name", str),
                ("vae_name", str),
                ("clip_skip", int),
                ("positive", str),
                ("negative", str),
                ("empty_latent_width", int),
                ("empty_latent_height", int),
                ("seed", int),
            ),
            "showtext|pysssss": (("text", str, "text"),),
        },
    )

    def __init__(self, prompt: str, workflow: str | None):
        self.text_metadata = prompt
        self.params = self.get_params_from_string(prompt, workflow)

    def get_display_text(self) -> str:
        r"""Re-indent ComfyUI's compact JSON.

        ComfyUI stores the prompt graph as a single minified line, which in a
        Discord code block renders as one very long horizontally-scrolling row.
        ensure_ascii=False keeps non-Latin prompts legible rather than escaping
        them to \uXXXX sequences.
        """
        try:
            return json.dumps(
                json.loads(self.text_metadata),
                indent=2,
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return self.text_metadata

    def get_embed(self, msg_ctx: Message, attachment=None):
        return super().get_embed(
            msg_ctx,
            attachment=attachment,
            prioritize_fields=tuple(k for k, v in self.params.items() if "text" in v)
            if CFG.comfyui_prioritize_nodes_with_text
            else (),
        )

    def get_params_from_string(
        self,
        param_str: str,
        workflow_str: str | None,
    ) -> OrderedDict[str, str]:
        promptdata = json.loads(param_str)
        workflowdata = {}
        if workflow_str:
            with contextlib.suppress(Exception):
                workflowdata = json.loads(workflow_str)
        comfymeta = self.extract_comfy_metadata(promptdata, workflowdata)
        params = OrderedDict()
        nl = "\n"
        for k, v in comfymeta.items():
            vs = ((ik, str(iv)) for ik, iv in v.items())
            params[k] = "\n".join(
                f"[{ik}]:{f' {iv}' if len(iv) < 32 else f'{nl}{iv}{nl}'}"
                for ik, iv in vs
            ).strip()
        return params

    @staticmethod
    def set_comfy_input(result, name, key, inputs, typ=str) -> None:
        val = inputs.get(key)
        if val is None or not isinstance(val, typ):
            return
        if typ is str:
            if CFG.comfyui_replace_field_newlines:
                val = val.replace("\r", " ").replace("\n", " ")
            val = val.strip()
        vals = result.get(name)
        if vals is None:
            result[name] = {key: val}
        else:
            vals[key] = val

    @classmethod
    def set_widget_value(
        cls,
        workflowdata,
        inputs,
        node_id,
        input_name,
        required_type,
        widget_name,
    ):
        wf_node = workflowdata.get(node_id, {})
        widget_idx = -1
        for idx, wf_input in enumerate(wf_node.get("inputs", ())):
            if wf_input.get("name") != input_name:
                continue
            cur_widget_name = wf_input.get("widget", {}).get("name")
            if cur_widget_name != widget_name:
                continue
            widget_idx = idx
            break
        widget_values = wf_node.get("widgets_values", ())
        if widget_idx != -1 and widget_idx < len(widget_values):
            widget_value = None
            with contextlib.suppress(IndexError):
                widget_value = widget_values[widget_idx][0]
            if isinstance(widget_value, required_type):
                inputs[input_name] = widget_value

    @classmethod
    def extract_comfy_metadata(cls, promptdata, workflowdata, result=None):
        workflowdata = {str(v["id"]): v for v in workflowdata.get("nodes", ())}
        handlers = cls.COMFY_HANDLERS
        if result is None:
            result = OrderedDict()
        for k, v in promptdata.items():
            inputs = (v.get("inputs") or {}).copy()
            typ = v.get("class_type", "").strip()
            handler = handlers.get(typ.lower())
            if not inputs or not handler:
                continue
            for input_name, required_type, *rest in handler:
                if rest and CFG.comfyui_extract_widget_values:
                    cls.set_widget_value(
                        workflowdata,
                        inputs,
                        k,
                        input_name,
                        required_type,
                        rest[0],
                    )
                if CFG.comfyui_show_titles:
                    title = v.get("_meta", {}).get("title", None)
                    if isinstance(title, str):
                        title = title.strip()
                    if title == typ or not title:
                        title = None
                else:
                    title = None
                name = (
                    f"{typ}.{k} - {title.strip()}"
                    if title is not None
                    else f"{typ}.{k}"
                )
                cls.set_comfy_input(result, name, input_name, inputs, required_type)
        return result


# Exif tag ids used to smuggle generation metadata into non-PNG containers.
EXIF_IFD = 0x8769
EXIF_USER_COMMENT = 0x9286
EXIF_IMAGE_DESCRIPTION = 0x010E
EXIF_MAKE = 0x010F


def decode_exif_text(payload: bytes, encoding: str) -> str | None:
    try:
        return payload.decode(encoding).rstrip("\x00")
    except (UnicodeDecodeError, LookupError):
        return None


def score_plausible_text(text: str) -> int:
    """Rough plausibility score. Prompts are overwhelmingly printable ASCII."""
    return sum(1 for ch in text if " " <= ch <= "~" or ch in "\r\n\t")


def decode_user_comment(raw: bytes | str | None) -> str | None:  # noqa: PLR0911
    """Decode an Exif UserComment value into text.

    UserComment carries an 8-byte character-code prefix (Exif 2.3 section 4.6.5).
    A1111 writes it via piexif as UTF-16, but the endianness in the wild is not
    consistent and mis-decoding does not raise, so pick whichever direction
    yields more plausible text.
    """
    if isinstance(raw, str):
        return raw or None
    if not isinstance(raw, bytes):
        return None
    prefix, payload = raw[:8], raw[8:]
    if prefix == b"UNICODE\x00":
        if payload[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return decode_exif_text(payload, "utf-16")
        best = None
        for encoding in ("utf-16-be", "utf-16-le"):
            text = decode_exif_text(payload, encoding)
            if text is None:
                continue
            if best is None or score_plausible_text(text) > score_plausible_text(best):
                best = text
        return best
    if prefix == b"ASCII\x00\x00\x00":
        return decode_exif_text(payload, "utf-8") or decode_exif_text(
            payload, "latin-1"
        )
    if prefix == b"\x00" * 8:
        return decode_exif_text(payload, "utf-8")
    # No recognised prefix, treat the whole value as text.
    return decode_exif_text(raw, "utf-8") or decode_exif_text(raw, "latin-1")


def get_image_text_chunks(img: Image.Image) -> dict[str, str]:
    """Collect generation metadata from PNG text chunks or, failing that, Exif.

    Keys match the names A1111 and ComfyUI use for their PNG text chunks
    ("parameters", "prompt", "workflow") whatever the container format, so
    callers can treat every format the same way.
    """
    chunks = {k: v for k, v in (img.info or {}).items() if isinstance(v, str)}
    if "parameters" in chunks or "prompt" in chunks:
        return chunks
    # JPEG/WebP/AVIF have no text chunks. A1111 puts the parameter string in
    # Exif UserComment; ComfyUI puts "Prompt:"/"Workflow:" JSON in Make and
    # ImageDescription.
    exif = None
    with contextlib.suppress(Exception):
        exif = img.getexif()
    if not exif:
        return chunks
    ifd = {}
    with contextlib.suppress(Exception):
        ifd = exif.get_ifd(EXIF_IFD)
    comment = decode_user_comment(ifd.get(EXIF_USER_COMMENT))
    if comment and MetadataA1111.STEPS_KEY in comment:
        chunks.setdefault("parameters", comment)
    for tag, tag_prefix, key in (
        (EXIF_MAKE, "Prompt:", "prompt"),
        (EXIF_IMAGE_DESCRIPTION, "Workflow:", "workflow"),
    ):
        value = exif.get(tag)
        if isinstance(value, str) and value.startswith(tag_prefix):
            chunks.setdefault(key, value[len(tag_prefix) :])
    return chunks


def parse_metadata(cls: type[Metadata], *args: str | None) -> Metadata | None:
    """Build a Metadata subclass, returning None if the payload is malformed.

    Detection is necessarily a heuristic, so a file can look like a format and
    still fail to parse. json.JSONDecodeError subclasses ValueError, so this
    covers both a truncated ComfyUI graph and an unparseable A1111 block.
    """
    try:
        return cls(*args)
    except ValueError as error:
        log.warning(
            __f(
                "Ignoring unparseable {name} metadata: {error}",
                name=cls.NAME,
                error=error,
            ),
        )
        return None


def populate_attachment_metadata(
    i: int,
    image_data: bytes,
    metadata: OrderedDict,
):
    with Image.open(io.BytesIO(image_data)) as img:
        ii = get_image_text_chunks(img)
        if not ii:
            return
        md = None
        if MetadataA1111.STEPS_KEY in ii.get("parameters", ""):
            # Has Steps in paramaters field, looks like A1111 format
            md = parse_metadata(MetadataA1111, ii["parameters"])
        elif ii.get("prompt", "").lstrip().startswith('{"'):
            # (Apparent) JSON data in prompt field, looks like ComfyUI format
            md = parse_metadata(MetadataComfyUI, ii["prompt"], ii.get("workflow"))
        #
        # NovelAI NYI
        # elif ii.get("Software") == "NovelAI" and "Description" in ii:
        #     info = ii["Description"] + ii.get("Comment", "")
        if md is not None and md.params:
            metadata[i] = md


async def read_attachment_metadata(
    i: int,
    attachment: Attachment,
    metadata: OrderedDict,
):
    """Read one attachment's metadata, so callers can download in bulk."""
    try:
        image_data = await attachment.read()
        populate_attachment_metadata(i, image_data, metadata)
    except Exception as error:
        errname = type(error).__name__
        log.exception(__f("Error: {errname}", errname=errname), exc_info=error)


def scannable_attachments(message: Message) -> list[Attachment]:
    """Attachments worth downloading: supported extension, under the size limit."""
    extensions = tuple(f".{e.lower().lstrip('.')}" for e in CFG.scan_extensions)
    return [
        a
        for a in message.attachments
        if a.filename.lower().endswith(extensions) and a.size < CFG.scan_limit_bytes
    ]


async def collect_attachments(
    ctx: ApplicationContext,
    message: Message,
    respond=True,
):
    if respond:
        await ctx.defer(ephemeral=True)
    attachments = scannable_attachments(message)
    if not attachments:
        if respond:
            await ctx.respond("This post contains no matching images.", ephemeral=True)
        return None, None
    metadata = OrderedDict()
    tasks = [
        read_attachment_metadata(i, attachment, metadata)
        for i, attachment in enumerate(attachments)
    ]
    await asyncio.gather(*tasks)
    if not metadata:
        if respond:
            await ctx.respond(
                "This post contains no image generation data.",
                ephemeral=True,
            )
        return None, None
    return metadata, attachments


async def update_reactions(message: Message, count: int):
    with contextlib.suppress(HTTPException):
        if count > 0:
            await message.add_reaction("🔎")
        elif CFG.react_on_no_metadata:
            await message.add_reaction("⛔")


async def add_heartboard(message: Message):
    with contextlib.suppress(HTTPException):
        await message.add_reaction("❤")


async def fetch_reaction_message(ctx: RawReactionActionEvent) -> Message | None:
    """Resolve the message a raw reaction refers to.

    Both the channel lookup and the message fetch can fail: get_channel only
    consults the cache, and this is the *raw* handler precisely because nothing
    here is guaranteed to be cached.
    """
    channel = client.get_channel(ctx.channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(ctx.channel_id)
        except (HTTPException, InvalidData):
            channel = None
    if not isinstance(channel, Messageable):
        log.warning(
            __f("Reaction in unreachable channel {cid}", cid=ctx.channel_id),
        )
        return None
    try:
        return await channel.fetch_message(ctx.message_id)
    except HTTPException as error:
        log.warning(
            __f(
                "Could not fetch message {mid}: {error}",
                mid=ctx.message_id,
                error=error,
            ),
        )
        return None


@client.event
async def on_ready():
    log.info(__f("Logged in as {user}!", user=client.user))


@client.event
async def on_message(message: Message):
    if (
        not (message.channel.id in CFG.monitored_channel_ids and message.attachments)
        or message.author.bot
    ):
        return
    await add_heartboard(message)
    attachments = scannable_attachments(message)
    if not attachments:
        return
    log.info(__f("MESSAGE: {0!r}", message))
    count = 0
    for i, attachment in enumerate(
        attachments
    ):  # download one at a time as usually the first image is already ai-generated
        metadata = OrderedDict()
        await read_attachment_metadata(i, attachment, metadata)
        if metadata:
            count += 1
            break
    await update_reactions(message, count)


@client.event
async def on_raw_reaction_add(ctx: RawReactionActionEvent):
    """Send image metadata in the reacted post to the reacting user's DMs."""
    if (
        ctx.emoji.name != "🔎"
        or ctx.channel_id not in CFG.monitored_channel_ids
        or getattr(ctx.member, "bot", False)
    ):
        return
    message = await fetch_reaction_message(ctx)
    if message is None:
        return
    log.info(__f("REACTION: {0!r}", ctx))
    metadata, attachments = await collect_attachments(ctx, message, respond=False)
    count = 0
    if metadata:
        user_dm = await (await client.fetch_user(ctx.user_id)).create_dm()
        for attachment, md in ((attachments[i], data) for i, data in metadata.items()):
            embed, view = md.get_embed_view(message, attachment)
            await user_dm.send(embed=embed, view=view, mention_author=False)
            count += 1
    await update_reactions(message, count)


@client.message_command(name="View Prompt")
async def message_command_view_prompt(ctx: ApplicationContext, message: Message):
    """Get raw list of parameters for every image in this post."""
    log.info(
        __f("APP: View: ctx={ctx!r}, message={message!r}", ctx=ctx, message=message),
    )
    metadata, attachments = await collect_attachments(ctx, message)
    if not metadata:
        return
    for attachment, md in ((attachments[i], data) for i, data in metadata.items()):
        embed, view = md.get_embed_view(message, attachment, ephemeral=True)
        # `ephemeral` must be passed on every call: collect_attachments already
        # deferred, so each respond() goes out as a followup, and followups
        # default to public.
        await ctx.respond(embed=embed, view=view, ephemeral=True)


@client.message_command(name="View Prompt (Get a DM)")
async def message_command_view_prompt_dm(ctx: ApplicationContext, message: Message):
    """Get raw list of parameters for every image in this post."""
    log.info(
        __f("APP: ViewDM: ctx={ctx!r}, message={message!r}", ctx=ctx, message=message),
    )
    metadata, attachments = await collect_attachments(ctx, message)
    if not metadata:
        return
    user_dm = await ctx.author.create_dm()
    for attachment, md in ((attachments[i], data) for i, data in metadata.items()):
        embed, view = md.get_embed_view(message, attachment)
        try:
            await user_dm.send(embed=embed, view=view, mention_author=False)
        except Exception as error:
            errname = type(error).__name__
            log.exception(__f("Error: {errname}", errname=errname), exc_info=error)
            await ctx.respond(
                "Couldn't DM. Please check that your DMs from non-friends are enabled for this server.",
                ephemeral=True,
                delete_after=60,
            )
            return
    await ctx.respond("DM sent!", ephemeral=True, delete_after=60)


def handle_check(filename: Path):
    with filename.open("rb") as fp:
        file_data = fp.read()
    metadata = OrderedDict()
    try:
        populate_attachment_metadata(0, file_data, metadata)
    except UnidentifiedImageError:
        print(f"* Not an image file Pillow can read: {filename}")
        sys.exit(1)
    if not metadata:
        print("* No metadata")
        return
    md = metadata[0]
    print(f"* Dumping {md.NAME} parameters")
    for k, v in md.params.items():
        print(f"\n{k.strip()}:\n{v.strip()}")


class ColorLogFormatter(logging.Formatter):
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {  # noqa: RUF012
        logging.DEBUG: GREY,
        logging.INFO: GREY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(
        self,
        use_color: bool,
        fmt="{asctime} {levelname:>8}: {message}",
        datefmt="%Y%m%d.%H%M%S",
    ):
        super().__init__(
            fmt=fmt,
            datefmt=datefmt,
            style="{",
        )
        if use_color:
            self.formatters = {
                ll: ColorExceptionLogFormatter(
                    fmt=f"{self.LEVEL_COLORS[ll]}{fmt}{self.RESET}",
                    datefmt=datefmt,
                    style="{",
                )
                for ll in (
                    logging.DEBUG,
                    logging.INFO,
                    logging.WARNING,
                    logging.ERROR,
                    logging.CRITICAL,
                )
            }
        self.use_color = use_color

    def format(self, record):
        if not self.use_color:
            return super().format(record)
        return self.formatters[record.levelno].format(record)


class ColorExceptionLogFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super().formatException(exc_info)
        return f"{ColorLogFormatter.BOLD_RED}{result}{ColorLogFormatter.RESET}"

    def formatStack(self, stack_info):
        return f"{ColorLogFormatter.BOLD_RED}{stack_info}{ColorLogFormatter.RESET}"


def setup_logging():
    log_level = logging.getLevelNamesMapping().get(CFG.log_level.upper())
    if log_level is None:
        raise ValueError("Invalid log_level in configuration")
    log.setLevel(log_level)
    ch = logging.StreamHandler()  # Simple console logging
    ch.setLevel(log_level)
    ch.setFormatter(ColorLogFormatter(CFG.log_color))
    log.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(description="Prompt inspector bot")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Configuration file",
    )
    parser.add_argument(
        "-d",
        "--dump",
        type=Path,
        help="Check metadata for the specified file",
    )
    args = parser.parse_args()
    CFG.load(args.config)
    setup_logging()
    if args.dump:
        handle_check(args.dump)
        return
    # Otherwise run the bot
    if not CFG.monitored_channel_ids:
        log.error("No channels to monitor!")
        sys.exit(1)
    bot_token = os.environ.get("BOT_TOKEN")
    if bot_token is None:
        log.error("BOT_TOKEN environment variable missing!")
        sys.exit(1)
    client.run(bot_token)


if __name__ == "__main__":
    main()
