"""Optional browser acceptance test for the distributed voice editor."""
import importlib.util
import os
import re
from pathlib import Path
import socket
import tempfile
import threading
import time
from unittest import TestCase, skipUnless
from unittest.mock import patch

from fastapi import FastAPI
import uvicorn

from api_neko import config
from api_neko.frontend_app import FrontendApp
from api_neko.routers import config as config_router
from api_neko.routers.health import router as health_router


@skipUnless(os.environ.get("GSV_BROWSER_TEST") == "1" and importlib.util.find_spec("playwright"),
            "set GSV_BROWSER_TEST=1 with Playwright installed")
class VoiceEditorBrowserTests(TestCase):
    def test_v5_selection_save_and_reload(self):
        from playwright.sync_api import sync_playwright, expect

        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voices = root / "voices"
            voices.mkdir()
            reference = root / "reference.wav"
            reference.write_bytes(b"temporary reference; this test does not run inference")
            pretrained = root / "GPT_SoVITS/pretrained_models"
            v5 = pretrained / "gsv-v5-pretrained"
            v5.mkdir(parents=True)
            (pretrained / "s1v3.ckpt").write_bytes(b"test GPT metadata")
            for version in ("v5dev", "v5turbo"):
                (v5 / f"s2G{version}.pth").write_bytes(b"test SoVITS metadata")
            app = FastAPI()
            app.state.backend_name = "gsv"
            app.include_router(config_router.router)
            app.include_router(health_router)
            app.mount("/", FrontendApp(repository / "frontend/.output/public"))

            with patch.object(config, "VOICES_DIR", voices), \
                    patch.object(config, "PROJECT_ROOT", root), \
                    patch.object(config, "SETTINGS_PATH", root / "settings.toml"), \
                    patch.object(config, "_default_data_cache", {}), \
                    patch.object(config_router, "_PROJECT_ROOT", root), \
                    patch("api_neko.backends.gsv_backend.sovits_model_version", side_effect=ImportError):
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
                thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
                thread.start()
                print(f"Temporary browser test server: http://127.0.0.1:{port}; automatically stopped after test")
                try:
                    deadline = time.monotonic() + 10
                    while not server.started and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(server.started)
                    with sync_playwright() as playwright:
                        options = {"headless": True, "args": ["--disable-gpu"]}
                        executable = os.environ.get("GSV_BROWSER_EXECUTABLE")
                        if executable:
                            options["executable_path"] = executable
                        browser = playwright.chromium.launch(**options)
                        try:
                            context = browser.new_context(viewport={"width": 1440, "height": 1100}, color_scheme="dark")
                            page = context.new_page()
                            errors = []
                            page.on("pageerror", lambda error: errors.append(str(error)))
                            base = f"http://127.0.0.1:{port}"
                            page.goto(base + "/config/new")
                            editor = page.get_by_test_id("voice-editor")
                            expect(editor).to_be_visible(timeout=30000)
                            version = page.get_by_test_id("model-version")
                            expect(version).to_be_visible(timeout=30000)
                            values = version.locator("option").evaluate_all("options => options.map(option => option.value)")
                            self.assertIn("v5dev", values)
                            self.assertIn("v5turbo", values)

                            version.select_option("v5turbo")
                            expect(page.get_by_test_id("sovits-weights")).to_have_value(
                                re.compile(r"s2Gv5turbo\.pth$"))
                            self.assertTrue(page.get_by_test_id("gpt-weights").input_value().endswith("s1v3.ckpt"))
                            expect(page.get_by_test_id("sample_steps-auto")).to_be_checked()
                            expect(page.get_by_test_id("sample_steps-default")).to_contain_text("4")
                            expect(page.get_by_test_id("cfg_rate-default")).to_contain_text("0")
                            page.get_by_test_id("voice-id").fill("browser-turbo")
                            page.get_by_test_id("voice-name").fill("V5 Turbo test")
                            page.get_by_test_id("reference-path").fill(str(reference))
                            page.get_by_test_id("reference-transcript").fill("Public test transcript")
                            page.get_by_test_id("use_flash_attention").uncheck()
                            page.get_by_test_id("save-voice").click()
                            expect(page).to_have_url(base + "/config")

                            page.goto(base + "/config/browser-turbo")
                            expect(page.get_by_test_id("model-version")).to_have_value("v5turbo", timeout=30000)
                            expect(page.get_by_test_id("sample_steps-auto")).to_be_checked()
                            expect(page.get_by_test_id("cfg_rate-auto")).to_be_checked()
                            expect(page.get_by_test_id("use_flash_attention")).not_to_be_checked()
                            saved = config.load_voice("browser-turbo")
                            self.assertIsNone(saved.params.sample_steps)
                            self.assertIsNone(saved.params.cfg_rate)
                            self.assertIs(saved.params.use_flash_attention, False)

                            page.goto(base + "/config/new")
                            expect(page.get_by_test_id("model-version")).to_be_visible()
                            page.get_by_test_id("model-version").select_option("v5dev")
                            expect(page.get_by_test_id("sovits-weights")).to_have_value(
                                re.compile(r"s2Gv5dev\.pth$"))
                            expect(page.get_by_test_id("sample_steps-default")).to_contain_text("32")
                            expect(page.get_by_test_id("cfg_rate-default")).to_contain_text("1.3")
                            page.get_by_test_id("sample_steps-auto").uncheck()
                            page.get_by_test_id("sample_steps-control").locator('input[type="number"]').fill("12")
                            page.get_by_test_id("cfg_rate-auto").uncheck()
                            page.get_by_test_id("cfg_rate-control").locator('input[type="number"]').fill("0.9")
                            page.get_by_test_id("voice-id").fill("browser-dev")
                            page.get_by_test_id("voice-name").fill("V5 Dev test")
                            page.get_by_test_id("reference-path").fill(str(reference))
                            page.get_by_test_id("reference-transcript").fill("Public test transcript")
                            page.get_by_test_id("save-voice").click()
                            expect(page).to_have_url(base + "/config")
                            page.goto(base + "/config/browser-dev")
                            expect(page.get_by_test_id("model-version")).to_have_value("v5dev", timeout=30000)
                            expect(page.get_by_test_id("sample_steps-auto")).not_to_be_checked()
                            expect(page.get_by_test_id("sample_steps-control").locator('input[type="number"]')).to_have_value("12")
                            expect(page.get_by_test_id("cfg_rate-control").locator('input[type="number"]')).to_have_value("0.9")
                            saved = config.load_voice("browser-dev")
                            self.assertEqual(saved.params.sample_steps, 12)
                            self.assertEqual(saved.params.cfg_rate, 0.9)
                            self.assertEqual(errors, [])
                            screenshot = os.environ.get("GSV_BROWSER_SCREENSHOT")
                            if screenshot:
                                page.screenshot(path=screenshot, full_page=True)
                            context.close()
                        finally:
                            browser.close()
                finally:
                    server.should_exit = True
                    thread.join(10)
                    listener.close()
                    self.assertFalse(thread.is_alive())
