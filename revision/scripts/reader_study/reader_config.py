"""Reader identities for the reader-study analysis, loaded from a local, git-ignored file.

The released code carries no reader logins. Analysing a real export needs
``revision/config/readers.local.json`` (same format as
``revision/config/readers.example.json``); ``PROSPER_READERS_CONFIG`` may point to
another file. Without either, the example with fake logins is used, which is enough
for the unit tests but matches no real reader.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
LOCAL = CONFIG_DIR / "readers.local.json"
EXAMPLE = CONFIG_DIR / "readers.example.json"


def config_path() -> Path:
    env = os.environ.get("PROSPER_READERS_CONFIG")
    if env:
        return Path(env)
    return LOCAL if LOCAL.exists() else EXAMPLE


def load_reader_config() -> dict:
    path = config_path()
    if path == EXAMPLE:
        print(f"reader_config: {LOCAL.name} not found; using the example (fake) reader logins",
              file=sys.stderr)
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("fig8_reader_order", "expertise_tier", "excluded_readers", "curator"):
        if key not in cfg:
            raise KeyError(f"{path}: missing key {key!r}")
    return cfg
