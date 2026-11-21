"""User-facing settings, persisted as JSON next to the rest of the app config."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .util import ClipForgeError, config_dir, default_output_dir

CONFIG_FILE = config_dir() / "config.json"

CAPTION_FONTS = ("montserrat", "bebas", "poppins", "bangers")
CAPTION_ANIMATIONS = ("word_pop", "typewriter", "karaoke")
CAPTION_POSITIONS = ("lower", "center", "upper")


@dataclass
class Config:
    # ---- what to make ---------------------------------------------------- #
    clip_count: int = 5
    min_duration: float = 15.0
    max_duration: float = 60.0

    # ---- output ---------------------------------------------------------- #
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crf: int = 20
    preset: str = "veryfast"

    # blur  : full frame in the middle, blurred copy filling the sides
    # crop  : fill the frame (best for talking heads)
    # pad   : letterbox on black
    layout: str = "blur"

    # center | face | static
    reframe: str = "face"

    # ---- captions -------------------------------------------------------- #
    captions: bool = True
    # Preset key used by captions.py: montserrat | bebas | poppins | bangers.
    # Any other value falls back to `font` below.
    caption_font: str = "montserrat"
    # Fallback font family name, used only when caption_font is not a preset.
    font: str = "Arial"
    font_size: int = 76
    caption_color: str = "FFFFFF"   # RGB hex
    highlight_color: str = "00E5FF"  # RGB hex
    caption_max_words: int = 4
    caption_max_chars: int = 18
    caption_position: str = "lower"  # lower | center | upper
    # word_pop | typewriter | karaoke
    caption_animation: str = "word_pop"

    # ---- speech to text -------------------------------------------------- #
    # auto | local | api | file
    transcribe_backend: str = "auto"
    whisper_model: str = "small"
    whisper_language: str = ""      # "" = auto detect
    whisper_device: str = "auto"    # auto | cpu | cuda
    whisper_compute_type: str = "auto"

    # ---- highlight picking ----------------------------------------------- #
    # "" = built-in heuristic scorer. Set a base URL for LLM mode.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    llm_json_mode: bool = True
    llm_timeout: int = 300

    # ---- output location ------------------------------------------------- #
    output_dir: str = field(default_factory=lambda: str(default_output_dir()))

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ClipForgeError if a choice field holds an unknown value."""
        checks = {
            "caption_font": (self.caption_font.lower(), CAPTION_FONTS, True),
            "caption_animation": (self.caption_animation.lower(), CAPTION_ANIMATIONS, False),
            "caption_position": (self.caption_position.lower(), CAPTION_POSITIONS, False),
        }
        for name, (value, allowed, optional) in checks.items():
            if optional and value not in allowed:
                # caption_font may be a custom family name, so unknown values are allowed.
                continue
            if value not in allowed:
                raise ClipForgeError(
                    f"Invalid {name} {getattr(self, name)!r}; expected one of {', '.join(allowed)}"
                )

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def copy(self) -> "Config":
        return Config(**asdict(self))

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_FILE
        cfg = cls()
        if not path.exists():
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise ClipForgeError(f"Could not read settings from {path}: {err}") from err
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
        cfg.validate()
        return cfg


# Environment variables win over the config file, so CI/scripts can override
# without touching the user's saved settings.
ENV_MAP = {
    "CLIPFORGE_LLM_BASE": "llm_base_url",
    "CLIPFORGE_LLM_KEY": "llm_api_key",
    "CLIPFORGE_LLM_MODEL": "llm_model",
    "CLIPFORGE_WHISPER_MODEL": "whisper_model",
    "CLIPFORGE_OUTPUT": "output_dir",
    "CLIPFORGE_TRANSCRIBE": "transcribe_backend",
    "CLIPFORGE_CAPTION_FONT": "caption_font",
    "CLIPFORGE_CAPTION_ANIMATION": "caption_animation",
}


def load_config() -> Config:
    cfg = Config.load()
    for env_name, attr in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value:
            setattr(cfg, attr, value)
    cfg.validate()
    return cfg