import json
import logging
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.__dict__.get("extra_fields"):
            payload.update(record.__dict__["extra_fields"])
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
