from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def json(self) -> Any:
        return httpx.Response(
            status_code=self.status_code,
            headers=self.headers,
            content=self.content,
        ).json()


async def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_s: float,
    headers: Optional[Dict[str, str]] = None,
) -> HttpResponse:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=payload, headers=headers)
        # Remove content-encoding header to prevent double-decompression
        headers_dict = dict(resp.headers)
        headers_dict.pop('content-encoding', None)
        headers_dict.pop('Content-Encoding', None)
        return HttpResponse(status_code=resp.status_code, headers=headers_dict, content=resp.content)


async def post_multipart_file(
    url: str,
    *,
    field_name: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
    timeout_s: float,
    data: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> HttpResponse:
    files = {field_name: (filename, file_bytes, content_type)}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, files=files, data=data, headers=headers)
        # Remove content-encoding header to prevent double-decompression
        headers_dict = dict(resp.headers)
        headers_dict.pop('content-encoding', None)
        headers_dict.pop('Content-Encoding', None)
        return HttpResponse(status_code=resp.status_code, headers=headers_dict, content=resp.content)
