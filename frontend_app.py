"""Static hosting for the prebuilt Nuxt frontend."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from starlette.responses import FileResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles


FRONTEND_MEDIA_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".ico": "image/x-icon",
    ".js": "text/javascript",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mjs": "text/javascript",
    ".map": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".wasm": "application/wasm",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}

# Build a registry-independent fallback for future frontend asset extensions.
_STANDARD_MEDIA_TYPES = mimetypes.MimeTypes(filenames=())


def frontend_media_type(path: str | Path) -> str:
    """Return a deterministic media type without consulting Windows registry."""
    suffix = Path(path).suffix.lower()
    if suffix in FRONTEND_MEDIA_TYPES:
        return FRONTEND_MEDIA_TYPES[suffix]
    return _STANDARD_MEDIA_TYPES.guess_type(str(path))[0] or "application/octet-stream"


def _content_type_header(path: str) -> str:
    media_type = frontend_media_type(path)
    return f"{media_type}; charset=utf-8" if media_type.startswith("text/") else media_type


def _is_client_route(path: str) -> bool:
    """The generated build leaves only config detail pages to the client router."""
    parts = path.strip("/").split("/")
    return (
        len(parts) == 2
        and parts[0] == "config"
        and parts[1] not in ("", ".", "..")
        and "\x00" not in parts[1]
        and not Path(parts[1]).suffix
    )


class FrontendApp:
    """ASGI frontend host composed around Starlette's path-safe StaticFiles."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.static = StaticFiles(directory=self.directory, html=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return
        if scope["type"] != "http":
            return

        request_path = scope.get("path", "")
        normalized_path = request_path.lstrip("/")
        if normalized_path == "api" or normalized_path.startswith("api/"):
            await PlainTextResponse("Not Found", status_code=404)(scope, receive, send)
            return

        extension = Path(request_path).suffix
        fallback_sent = False

        async def send_response(message):
            nonlocal fallback_sent
            if message["type"] == "http.response.start":
                status = message["status"]
                if status == 404 and _is_client_route(request_path):
                    fallback = self.directory / "200.html"
                    if fallback.is_file():
                        fallback_sent = True
                        await FileResponse(fallback, media_type="text/html")(
                            scope, receive, send
                        )
                        return

                if status in (200, 206, 304) and extension:
                    headers = [
                        (name, value)
                        for name, value in message["headers"]
                        if name.lower() != b"content-type"
                    ]
                    headers.append(
                        (b"content-type", _content_type_header(request_path).encode())
                    )
                    message = {**message, "headers": headers}

            if not fallback_sent:
                await send(message)

        try:
            await self.static(scope, receive, send_response)
        except (ValueError, OSError):
            if not fallback_sent:
                await PlainTextResponse("Not Found", status_code=404)(scope, receive, send)
