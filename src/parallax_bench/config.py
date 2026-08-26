"""``systems.toml`` -> validated configuration.

Ablations are expressed here as multiple configured instances of the same
adapter (e.g. ``edge-ant-rerank`` vs ``edge-ant-norerank``), never as extra
parameters in the adapter protocol.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, field_validator

from parallax_bench.adapters.base import RagSystem, load_adapter

_ENV_RE = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)\}")


def _expand_env(value):
    """Expand ``${VAR}`` in string values, recursively through dicts/lists.

    Unset variables are a hard error — a benchmark run against a half-configured
    system must fail loudly, not measure something else.
    """
    if isinstance(value, str):

        def sub(m: re.Match) -> str:
            name = m.group("name")
            if name not in os.environ:
                raise KeyError(f"environment variable {name} referenced in systems.toml is not set")
            return os.environ[name]

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


_SECRET_KEY_RE = re.compile(r"(secret|token|password|api_key|apikey|jwt)", re.IGNORECASE)
_ENDPOINT_KEY_RE = re.compile(r"(url|endpoint|host)", re.IGNORECASE)


def redacted_config(config: dict) -> dict:
    """Resolved configuration as stored in a run's ``config.json``.

    Secrets are replaced by ``"<redacted>"`` — never dropped, it must remain
    visible that they were there.  Production endpoints are replaced by the
    stable alias ``"<sut-endpoint>"``.  Everything else is recorded verbatim.
    """
    resolved = _expand_env(config)

    def mask(key: str, value):
        if isinstance(value, dict):
            return {k: mask(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [mask(key, v) for v in value]
        if _SECRET_KEY_RE.search(key):
            return "<redacted>"
        if _ENDPOINT_KEY_RE.search(key) and isinstance(value, str):
            return "<sut-endpoint>"
        return value

    return {k: mask(k, v) for k, v in resolved.items()}


class SystemConfig(BaseModel):
    id: str
    adapter: str  # "pkg.module:ClassName"
    config: dict = {}

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", v):
            raise ValueError(f"system id must be kebab-case, got {v!r}")
        return v

    @field_validator("adapter")
    @classmethod
    def _adapter_shape(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError(f"adapter must look like 'pkg.module:ClassName', got {v!r}")
        return v

    def instantiate(self) -> RagSystem:
        return load_adapter(self.adapter, _expand_env(self.config))


BUILTIN_SYSTEMS: dict[str, SystemConfig] = {
    # available without any systems.toml so the quickstart works out of the box
    "baseline-local": SystemConfig(
        id="baseline-local",
        adapter="parallax_bench.adapters.baseline_local:BaselineLocalSystem",
        config={},
    ),
}


def load_systems(path: Path | None = None) -> dict[str, SystemConfig]:
    systems = dict(BUILTIN_SYSTEMS)
    if path is None:
        default = Path.cwd() / "systems.toml"
        path = default if default.is_file() else None
    if path is not None:
        # a single-file adapter living next to systems.toml must be importable
        # without packaging or PYTHONPATH gymnastics
        toml_dir = str(Path(path).resolve().parent)
        if toml_dir not in sys.path:
            sys.path.insert(0, toml_dir)
        with Path(path).open("rb") as fh:
            raw = tomllib.load(fh)
        for entry in raw.get("systems", []):
            cfg = SystemConfig.model_validate(entry)
            systems[cfg.id] = cfg
    return systems


def get_system(system_id: str, path: Path | None = None) -> SystemConfig:
    systems = load_systems(path)
    if system_id not in systems:
        raise KeyError(
            f"unknown system {system_id!r}; known: {', '.join(sorted(systems))} "
            f"(define new ones in systems.toml)"
        )
    return systems[system_id]
