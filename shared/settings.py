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
    chat1to1_queue_max: int = 10
    shorts_queue_max: int = 10
    scenes_queue_max: int = 10
    regen_attempts: int = 2
    profile_sla_ms: int = 90_000
    text2img_sla_ms: int = 8_000
    # Image generation can take tens of seconds on GPU. Keep below typical edge proxy
    # timeouts by default, but allow tuning via env/config.
    chat1to1_sla_ms: int = 90_000
    shorts_sla_ms: int = 90_000
    scenes_sla_ms: int = 90_000

    # Default generation params (tune for speed/quality tradeoff).
    # Target: ~8–12s warm latency on decent GPUs with good quality.
    # 512x512 + 9 steps provides good quality without blur (7 was better but still slightly soft).
    # Recommended style: "3d_cartoon" for best results at this resolution.
    chat1to1_height: int = 512
    chat1to1_width: int = 512
    chat1to1_num_inference_steps: int = 9
    shorts_height: int = 512
    shorts_width: int = 512
    shorts_num_inference_steps: int = 9
    scenes_height: int = 512
    scenes_width: int = 512
    scenes_num_inference_steps: int = 9
    llm_service_url: HttpUrl = "https://abj0jt7cd4hgdy-8002.proxy.runpod.net"
    selfie_feature_worker_urls: List[HttpUrl] = ["https://97dsnjce4yxe96-8003.proxy.runpod.net"]
    profile_worker_urls: List[HttpUrl] = ["https://x690sjq9dtevw4-8003.proxy.runpod.net"]
    text2img_worker_urls: List[HttpUrl] = ["https://tx5t471v3aljjd-8004.proxy.runpod.net"]
    chat1to1_worker_urls: List[HttpUrl] = ["https://6eubyihk4kt8l0-8005.proxy.runpod.net"]
    shorts_worker_urls: List[HttpUrl] = ["https://klu0524bz1nx1i-8006.proxy.runpod.net"]
    scenes_worker_urls: List[HttpUrl] = ["https://5r1nfrz20lc715-8007.proxy.runpod.net"]

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


class Chat1to1WorkerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8005

    model_config = SettingsConfigDict(env_prefix="CHAT1TO1_WORKER_", extra="ignore")


class ShortsWorkerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8006

    model_config = SettingsConfigDict(env_prefix="SHORTS_WORKER_", extra="ignore")


class ScenesWorkerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8007

    model_config = SettingsConfigDict(env_prefix="SCENES_WORKER_", extra="ignore")


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
            "chat1to1_queue_max": "INTERFACE_CHAT1TO1_QUEUE_MAX",
            "shorts_queue_max": "INTERFACE_SHORTS_QUEUE_MAX",
            "scenes_queue_max": "INTERFACE_SCENES_QUEUE_MAX",
            "regen_attempts": "INTERFACE_REGEN_ATTEMPTS",
            "profile_sla_ms": "INTERFACE_PROFILE_SLA_MS",
            "text2img_sla_ms": "INTERFACE_TEXT2IMG_SLA_MS",
            "chat1to1_sla_ms": "INTERFACE_CHAT1TO1_SLA_MS",
            "shorts_sla_ms": "INTERFACE_SHORTS_SLA_MS",
            "scenes_sla_ms": "INTERFACE_SCENES_SLA_MS",

            "chat1to1_height": "INTERFACE_CHAT1TO1_HEIGHT",
            "chat1to1_width": "INTERFACE_CHAT1TO1_WIDTH",
            "chat1to1_num_inference_steps": "INTERFACE_CHAT1TO1_NUM_INFERENCE_STEPS",
            "shorts_height": "INTERFACE_SHORTS_HEIGHT",
            "shorts_width": "INTERFACE_SHORTS_WIDTH",
            "shorts_num_inference_steps": "INTERFACE_SHORTS_NUM_INFERENCE_STEPS",
            "scenes_height": "INTERFACE_SCENES_HEIGHT",
            "scenes_width": "INTERFACE_SCENES_WIDTH",
            "scenes_num_inference_steps": "INTERFACE_SCENES_NUM_INFERENCE_STEPS",
            "llm_service_url": "INTERFACE_LLM_SERVICE_URL",
            "profile_worker_url": "INTERFACE_PROFILE_WORKER_URL",
            # list handled separately
        }

        for field, env_key in mapping.items():
            val = self._env_override(env_key)
            if val is None:
                continue
            # type-coerce simple ints
            if field in {
                "port",
                "retry_after_cap_ms",
                "profile_queue_max",
                "text2img_queue_max",
                "chat1to1_queue_max",
                "shorts_queue_max",
                "scenes_queue_max",
                "regen_attempts",
                "profile_sla_ms",
                "text2img_sla_ms",
                "chat1to1_sla_ms",
                "shorts_sla_ms",
                "scenes_sla_ms",
                "chat1to1_height",
                "chat1to1_width",
                "chat1to1_num_inference_steps",
                "shorts_height",
                "shorts_width",
                "shorts_num_inference_steps",
                "scenes_height",
                "scenes_width",
                "scenes_num_inference_steps",
            }:
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

    @lru_cache(maxsize=1)
    def chat1to1_worker_settings(self) -> Chat1to1WorkerSettings:
        return Chat1to1WorkerSettings()  # type: ignore[arg-type]

    @lru_cache(maxsize=1)
    def shorts_worker_settings(self) -> ShortsWorkerSettings:
        return ShortsWorkerSettings()  # type: ignore[arg-type]

    @lru_cache(maxsize=1)
    def scenes_worker_settings(self) -> ScenesWorkerSettings:
        return ScenesWorkerSettings()  # type: ignore[arg-type]


config = AppConfig()
