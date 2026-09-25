import tempfile
from pathlib import Path
from unittest import TestCase

from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.testclient import TestClient

from api_neko.server import FrontendStaticFiles, _frontend_media_type, _resolve_frontend_path


class StaticAssetTests(TestCase):
    def test_frontend_media_types_are_explicit(self):
        expected = {
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".json": "application/json",
            ".txt": "text/plain",
            ".ico": "image/x-icon",
        }
        for suffix, media_type in expected.items():
            with self.subTest(suffix=suffix):
                self.assertEqual(_frontend_media_type("asset" + suffix), media_type)

    def test_static_mount_uses_explicit_media_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.js").write_text("console.log('ok');", encoding="utf-8")
            (root / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
            (root / "manifest.json").write_text('{"id":"test"}', encoding="utf-8")

            app = Starlette()
            app.mount("/", FrontendStaticFiles(directory=root), name="static")
            with TestClient(app) as client:
                self.assertEqual(client.get("/app.js").headers["content-type"], "text/javascript; charset=utf-8")
                self.assertEqual(client.get("/index.html").headers["content-type"], "text/html; charset=utf-8")
                self.assertEqual(client.get("/manifest.json").headers["content-type"], "application/json")

    def test_frontend_path_rejects_escape(self):
        with self.assertRaises(HTTPException) as caught:
            _resolve_frontend_path("../server.py")
        self.assertEqual(caught.exception.status_code, 404)
