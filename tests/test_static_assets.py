import mimetypes
import tempfile
from pathlib import Path
from unittest import TestCase

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from api_neko.frontend_app import FRONTEND_MEDIA_TYPES, FrontendApp, frontend_media_type


class StaticAssetTests(TestCase):
    def test_frontend_media_types_are_explicit(self):
        for suffix, media_type in FRONTEND_MEDIA_TYPES.items():
            with self.subTest(suffix=suffix):
                self.assertEqual(frontend_media_type("asset" + suffix), media_type)

    def test_unknown_media_types_use_registry_independent_standard_table(self):
        self.assertEqual(frontend_media_type("image.svg"), "image/svg+xml")
        self.assertEqual(frontend_media_type("font.woff2"), "font/woff2")
        self.assertEqual(frontend_media_type("unknown.nekoasset"), "application/octet-stream")

    def test_static_mount_serves_assets_and_client_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.js").write_text("console.log('ok');", encoding="utf-8")
            (root / "index.html").write_text("<h1>root</h1>", encoding="utf-8")
            (root / "200.html").write_text("<h1>spa</h1>", encoding="utf-8")
            (root / "404.html").write_text("<h1>not found</h1>", encoding="utf-8")
            (root / "manifest.json").write_text('{"id":"test"}', encoding="utf-8")

            app = Starlette(
                routes=[
                    Route("/api/v3/health", lambda request: JSONResponse({"ok": True}))
                ]
            )
            app.mount("/", FrontendApp(directory=root), name="frontend")
            with TestClient(app, raise_server_exceptions=False) as client:
                self.assertEqual(
                    client.get("/app.js").headers["content-type"],
                    "text/javascript; charset=utf-8",
                )
                self.assertEqual(
                    client.get("/manifest.json").headers["content-type"],
                    "application/json",
                )
                self.assertEqual(client.get("/api/v3/health").status_code, 200)
                self.assertEqual(client.get("/config/new").status_code, 200)
                self.assertEqual(client.get("/config/example").status_code, 200)
                self.assertEqual(client.get("/unknown-page").status_code, 404)
                self.assertEqual(client.get("/missing.js").status_code, 404)
                self.assertEqual(client.get("/favicon.ico").status_code, 404)
                self.assertEqual(client.get("/api/v3/unknown").status_code, 404)
                self.assertEqual(client.get("/api/unknown").status_code, 404)
                for path in ("/..%2Fserver.py", "/%2e%2e/server.py", "/a%00b"):
                    with self.subTest(path=path):
                        self.assertEqual(client.get(path).status_code, 404)

    def test_static_response_is_not_affected_by_global_registry_mapping(self):
        previous = mimetypes.types_map.get(".js")
        mimetypes.types_map[".js"] = "text/plain"
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                (root / "app.js").write_text("console.log('ok');", encoding="utf-8")
                app = Starlette()
                app.mount("/", FrontendApp(directory=root), name="frontend")
                with TestClient(app) as client:
                    self.assertEqual(
                        client.get("/app.js").headers["content-type"],
                        "text/javascript; charset=utf-8",
                    )
        finally:
            if previous is None:
                mimetypes.types_map.pop(".js", None)
            else:
                mimetypes.types_map[".js"] = previous
