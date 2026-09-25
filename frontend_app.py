"""Static hosting for the prebuilt Nuxt frontend."""

from __future__ import annotations

import mimetypes
import stat
from pathlib import Path

import anyio
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse
from starlette.staticfiles import NotModifiedResponse, StaticFiles


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


class FrontendStaticFiles(StaticFiles):
    """Serve frontend files and narrowly scoped extensionless SPA routes."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return
        if scope["type"] != "http":
            return
        await super().__call__(scope, receive, send)

    def file_response(self, full_path, stat_result, scope, status_code=200):
        request_headers = Headers(scope=scope)
        response = FileResponse(
            full_path,
            status_code=status_code,
            stat_result=stat_result,
            media_type=frontend_media_type(full_path),
        )
        if self.is_not_modified(response.headers, request_headers):
            return NotModifiedResponse(response.headers)
        return response

    async def get_response(self, path: str, scope):
        """Use StaticFiles for file safety, then fallback only for page routes."""
        normalized_path = path.lstrip("/")
        scope_path = scope.get("path", "").lstrip("/")
        if (
            normalized_path == "api"
            or normalized_path.startswith("api/")
            or scope_path == "api"
            or scope_path.startswith("api/")
        ):
            raise HTTPException(status_code=404)

        try:
            response = await super().get_response(path, scope)
        except (ValueError, OSError) as exc:
            # Starlette normally converts malformed filesystem paths to 404;
            # keep the same contract for platform-specific path errors.
            raise HTTPException(status_code=404) from exc

        if response.status_code != 404 or Path(path).suffix:
            return response

        # Nuxt emits 200.html for client-side routes. This branch never uses
        # user input as a filesystem path, so StaticFiles retains path safety.
        fallback_name = "200.html" if self.html else "index.html"
        full_path, stat_result = await anyio.to_thread.run_sync(
            self.lookup_path, fallback_name
        )
        if stat_result is not None and stat.S_ISREG(stat_result.st_mode):
            return self.file_response(full_path, stat_result, scope)
        raise HTTPException(status_code=404)
