import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from api_neko.backends.gsv_backend import sovits_model_version
from api_neko.routers import config as router


class ModelCatalogTests(TestCase):
    def test_host_weight_reader_handles_header_based_finetunes(self):
        reader = Mock(return_value=("v2", "v5turbo", True))
        with patch.dict("sys.modules", {"GPT_SoVITS.process_ckpt": SimpleNamespace(
                get_sovits_version_from_path_fast=reader)}):
            self.assertEqual(sovits_model_version("custom-voice.pth"), "v5turbo")
        reader.assert_called_once_with("custom-voice.pth")

    def test_v5_bases_share_the_official_gpt_without_listing_the_vocoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrained = root / "GPT_SoVITS/pretrained_models"
            folder = pretrained / "gsv-v5-pretrained"
            folder.mkdir(parents=True)
            (pretrained / "s1v3.ckpt").write_bytes(b"gpt")
            for name in ("s2Gv5dev.pth", "s2Gv5turbo.pth", "vocoder.pth"):
                (folder / name).write_bytes(b"weights")
            with patch.object(router, "_PROJECT_ROOT", root), \
                    patch("api_neko.backends.gsv_backend.sovits_model_version", side_effect=ImportError):
                models = router._scan_gsv_models()
            v5 = {entry["version"]: entry for entry in models if entry["version"].startswith("v5")}
            self.assertEqual(set(v5), {"v5dev", "v5turbo"})
            for entry in v5.values():
                self.assertEqual(entry["gpt"]["filename"], "s1v3.ckpt")
                self.assertTrue(entry["sovits"]["filename"].startswith("s2Gv5"))
            self.assertNotIn("vocoder.pth", str(models))
            with patch.object(router, "_PROJECT_ROOT", root), \
                    patch("api_neko.backends.gsv_backend.sovits_model_version", side_effect=ImportError):
                gpt = router._scan_weights(router._GPT_WEIGHT_ROOTS, {".ckpt"}, "v5turbo")
                sovits = router._scan_weights(router._SOVITS_WEIGHT_ROOTS, {".pth"}, "v5turbo")
            self.assertEqual([Path(path).name for path in gpt], ["s1v3.ckpt"])
            self.assertEqual([Path(path).name for path in sovits], ["s2Gv5turbo.pth"])

    def test_matching_finetuned_gpt_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, name in (("GPT_weights_v5dev", "speaker-e10.ckpt"),
                                 ("SoVITS_weights_v5dev", "speaker_e10_s20.pth")):
                path = root / folder / name
                path.parent.mkdir()
                path.write_bytes(b"weights")
            with patch.object(router, "_PROJECT_ROOT", root):
                models = router._scan_gsv_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["version"], "v5dev")
            self.assertEqual(models[0]["gpt"]["filename"], "speaker-e10.ckpt")
            self.assertEqual(models[0]["sovits"]["filename"], "speaker_e10_s20.pth")
