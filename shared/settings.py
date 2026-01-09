import os
from pathlib import Path
from functools import lru_cache
from typing import List, Optional

import yaml
from pydantic import HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InterfaceSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    retry_after_cap_ms: int = 120_000
    profile_queue_max: int = 10
    text2img_queue_max: int = 10
    regen_attempts: int = 2
    profile_sla_ms: int = 60_000
    text2img_sla_ms: int = 8_000
    llm_service_url: HttpUrl = "https://z8xu5cpyevi44q-8002.proxy.runpod.net"
    selfie_feature_worker_url: HttpUrl = "https://aknn9iryuo4tr1-8003.proxy.runpod.net"
    profile_worker_url: HttpUrl = "https://je5fxki9i9cekh-8003.proxy.runpod.net"
    text2img_worker_urls: List[HttpUrl] = ["https://tx5t471v3aljjd-8004.proxy.runpod.net"]

    model_config = SettingsConfigDict(env_prefix="INTERFACE_", extra="ignore")

    @field_validator("regen_attempts")
    def validate_regen_attempts(cls, v: int) -> int:  # noqa: N805
        if v < 1:
            raise ValueError("regen_attempts must be >= 1")
        return v


class LLMSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8002

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class ProfileWorkerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8003

    model_config = SettingsConfigDict(env_prefix="PROFILE_WORKER_", extra="ignore")


class Text2ImgWorkerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8004

    model_config = SettingsConfigDict(env_prefix="TEXT2IMG_WORKER_", extra="ignore")


class AppConfig:
    def __init__(self) -> None:
        # YAML config path is optional.
        self.config_path = os.getenv("APP_CONFIG_PATH")

    def _load_yaml(self) -> dict:
        candidate_paths: List[Path] = []
        if self.config_path:
            candidate_paths.append(Path(self.config_path))
        candidate_paths.append(Path("config/config.yaml"))

        for p in candidate_paths:
            try:
                if p.exists() and p.is_file():
                    with p.open("r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                        if isinstance(data, dict):
                            return data
            except Exception:
                # If config is malformed we fail closed by ignoring it here.
                # Services can still be configured via env vars.
                return {}
        return {}

    def _env_override(self, key: str) -> Optional[str]:
        return os.getenv(key)

    def _build_interface_kwargs(self) -> dict:
        data = self._load_yaml().get("interface", {})
        if not isinstance(data, dict):
            data = {}

        # Env overrides (explicit) so config file is the primary source.
        mapping = {
            "host": "INTERFACE_HOST",
            "port": "INTERFACE_PORT",
            "retry_after_cap_ms": "INTERFACE_RETRY_AFTER_CAP_MS",
            "profile_queue_max": "INTERFACE_PROFILE_QUEUE_MAX",
            "text2img_queue_max": "INTERFACE_TEXT2IMG_QUEUE_MAX",
            "regen_attempts": "INTERFACE_REGEN_ATTEMPTS",
            "profile_sla_ms": "INTERFACE_PROFILE_SLA_MS",
            "text2img_sla_ms": "INTERFACE_TEXT2IMG_SLA_MS",
            "llm_service_url": "INTERFACE_LLM_SERVICE_URL",
            "profile_worker_url": "INTERFACE_PROFILE_WORKER_URL",
            # list handled separately
        }

        for field, env_key in mapping.items():
            val = self._env_override(env_key)
            if val is None:
                continue
            # type-coerce simple ints
            if field in {"port", "retry_after_cap_ms", "profile_queue_max", "text2img_queue_max", "regen_attempts", "profile_sla_ms", "text2img_sla_ms"}:
                data[field] = int(val)
            else:
                data[field] = val

        # text2img_worker_urls can be set in YAML or env as comma-separated
        urls_env = self._env_override("INTERFACE_TEXT2IMG_WORKER_URLS")
        if urls_env:
            data["text2img_worker_urls"] = [u.strip() for u in urls_env.split(",") if u.strip()]

        return data

    @lru_cache(maxsize=1)
    def interface_settings(self) -> InterfaceSettings:
        kwargs = self._build_interface_kwargs()
        return InterfaceSettings(**kwargs)  # type: ignore[arg-type]

    @lru_cache(maxsize=1)
    def llm_settings(self) -> LLMSettings:
        return LLMSettings()  # type: ignore[arg-type]

    @lru_cache(maxsize=1)
    def profile_worker_settings(self) -> ProfileWorkerSettings:
        return ProfileWorkerSettings()  # type: ignore[arg-type]

    @lru_cache(maxsize=1)
    def text2img_worker_settings(self) -> Text2ImgWorkerSettings:
        return Text2ImgWorkerSettings()  # type: ignore[arg-type]


config = AppConfig()
