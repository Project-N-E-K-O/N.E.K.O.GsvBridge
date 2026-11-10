from io import BytesIO
from unittest import TestCase
import wave

import numpy as np

from api_neko.audio_utils import WavStreamEncoder, join_wav_chunks


def decode(chunks):
    frames = []
    for chunk in chunks:
        with wave.open(BytesIO(chunk), "rb") as reader:
            assert reader.getframerate() == 48000
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            frames.append(reader.readframes(reader.getnframes()))
    return np.frombuffer(b"".join(frames), dtype=np.int16)


class AudioOutputTests(TestCase):
    def test_streaming_resampler_flushes_all_tail_samples(self):
        for rate in (24000, 32000):
            with self.subTest(rate=rate):
                encoder = WavStreamEncoder()
                samples = np.full(rate // 10, 1000, dtype=np.int16)
                chunks = encoder.feed(samples[:37], rate) + encoder.feed(samples[37:], rate)
                before = len(decode(chunks))
                chunks += encoder.flush()
                self.assertLess(before, 4800)
                self.assertEqual(len(decode(chunks)), 4800)

    def test_native_pcm_is_preserved_and_each_frame_is_bounded(self):
        audio = np.arange(48000 * 3 + 19, dtype=np.int16)
        encoder = WavStreamEncoder()
        chunks = encoder.feed(audio, 48000) + encoder.flush()
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(chunk) <= 96044 for chunk in chunks))
        np.testing.assert_array_equal(decode(chunks), audio)
        combined = join_wav_chunks(chunks)
        np.testing.assert_array_equal(decode([combined]), audio)
