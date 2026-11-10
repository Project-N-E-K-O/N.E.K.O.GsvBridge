import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from api_neko import config
from api_neko.routers.config import VoiceConfigRequest, api_update_voice
from api_neko.routers.tts_v3 import TTSSubmitRequest, _build_tts_params


class VoiceParametersTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.voices = patch.object(config, "VOICES_DIR", self.root)
        self.voices.start()
        self.addCleanup(self.voices.stop)
        self.cache = patch.object(config, "_default_data_cache", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_auto_values_and_explicit_values_survive_toml(self):
        voice = config.VoiceConfig()
        voice.voice.id = "example"
        voice.ref_audio.path = "references/example.wav"
        voice.params.cfg_rate = 0.65
        voice.params.use_cuda_graph = False
        path = config.save_voice(voice)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("sample_steps =", text)
        self.assertNotIn('"None"', text)
        loaded = config.load_voice("example")
        self.assertIsNone(loaded.params.sample_steps)
        self.assertEqual(loaded.params.cfg_rate, 0.65)
        self.assertIs(loaded.params.use_cuda_graph, False)
        loaded.params.sample_steps = 32
        config.save_voice(loaded)
        self.assertEqual(config.load_voice("example").params.sample_steps, 32)

    def test_all_request_forms_can_override_v5_parameters(self):
        voice = config.VoiceConfig()
        voice.params.sample_steps = 32
        body = TTSSubmitRequest(text="hello", sample_steps=4, cfg_rate=0,
                                use_cuda_graph=False, use_flash_attention=False)
        values = _build_tts_params(voice, body.text, body.model_dump(exclude_unset=True))
        self.assertEqual(values["sample_steps"], 4)
        self.assertEqual(values["cfg_rate"], 0)
        self.assertIs(values["use_cuda_graph"], False)
        self.assertIs(values["use_flash_attention"], False)
        self.assertIsNone(_build_tts_params(voice, "hello", {"sample_steps": None})["sample_steps"])
        self.assertEqual(_build_tts_params(voice, "hello", {})["sample_steps"], 32)
        self.assertNotIn("voice_id", _build_tts_params(voice, "hello", {"voice_id": "other"}))

    def test_configured_default_is_separate_from_last_used_voice(self):
        for name in ("chosen", "last"):
            voice = config.VoiceConfig()
            voice.voice.id = name
            voice.ref_audio.path = "references/example.wav"
            config.save_voice(voice)
        settings = self.root / "settings.toml"
        with patch.object(config, "SETTINGS_PATH", settings):
            config.save_settings({"default_voice_id": "chosen", "last_voice_id": "last"})
            self.assertEqual(config.load_tts_voice("_default").voice.id, "chosen")
            self.assertEqual(config.load_tts_voice("last").voice.id, "last")


class LegacyVoiceUpdateTests(IsolatedAsyncioTestCase):
    async def test_old_editor_does_not_reset_unrendered_parameters(self):
        voice = config.VoiceConfig()
        voice.voice.id = "example"
        voice.ref_audio.path = "references/example.wav"
        voice.params.cfg_rate = 0.7
        voice.params.use_flash_attention = False
        data = asdict(voice)
        for name in ("sample_steps", "cfg_rate", "use_flash_attention", "use_cuda_graph"):
            data["params"].pop(name)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(config, "VOICES_DIR", Path(directory)), \
                patch.object(config, "_default_data_cache", {}):
            config.save_voice(voice)
            response = await api_update_voice("example", VoiceConfigRequest(**data))
            self.assertEqual(response["params"]["cfg_rate"], 0.7)
            self.assertIs(response["params"]["use_flash_attention"], False)
            data["params"]["cfg_rate"] = None
            await api_update_voice("example", VoiceConfigRequest(**data))
            self.assertIsNone(config.load_voice("example").params.cfg_rate)
