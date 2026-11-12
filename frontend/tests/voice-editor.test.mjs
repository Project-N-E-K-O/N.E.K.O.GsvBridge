import test from 'node:test';
import assert from 'node:assert/strict';
import { MODEL_VERSIONS, modelDefaults, newVoice, normalizeVoice, preferredWeight, setAutomaticParameter, editorText } from '../src/voice-editor.js';

test('V5 selection exposes model defaults without forcing 32 steps onto Turbo', () => {
  assert.ok(MODEL_VERSIONS.includes('v5dev'));
  assert.ok(MODEL_VERSIONS.includes('v5turbo'));
  assert.deepEqual(modelDefaults('v5turbo'), { sample_steps: 4, cfg_rate: 0 });
  assert.deepEqual(modelDefaults('v5dev'), { sample_steps: 32, cfg_rate: 1.3 });
  const voice = newVoice();
  assert.equal(voice.params.sample_steps, null);
  assert.equal(voice.params.cfg_rate, null);
});

test('saved explicit values and acceleration toggles survive normalization', () => {
  const voice = normalizeVoice({ model: { version: 'v5turbo' }, params: {
    sample_steps: 16, cfg_rate: 0.75, use_cuda_graph: false, use_flash_attention: false,
  } });
  assert.equal(voice.params.sample_steps, 16);
  assert.equal(voice.params.cfg_rate, 0.75);
  assert.equal(voice.params.use_cuda_graph, false);
  assert.equal(voice.params.use_flash_attention, false);
  setAutomaticParameter(voice.params, 'sample_steps', true, 'v5turbo');
  assert.equal(JSON.parse(JSON.stringify(voice)).params.sample_steps, null);
  setAutomaticParameter(voice.params, 'sample_steps', false, 'v5turbo');
  assert.equal(voice.params.sample_steps, 4);
});

test('V5 weight selection uses the shared GPT and the selected acoustic model', () => {
  assert.equal(preferredWeight(['other.ckpt', 'GPT_SoVITS\\pretrained_models\\s1v3.ckpt'], 'v5dev', 'gpt'),
    'GPT_SoVITS\\pretrained_models\\s1v3.ckpt');
  const target = 'GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth';
  assert.equal(preferredWeight(['old.pth', target], 'v5turbo', 'sovits'), target);
});

test('new UI controls have labels in every distributed language', () => {
  for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
    for (const key of ['auto', 'autoValue', 'cfg', 'acceleration', 'versionHelp', 'retainedHelp']) {
      assert.notEqual(editorText(locale, key), key);
    }
  }
});
