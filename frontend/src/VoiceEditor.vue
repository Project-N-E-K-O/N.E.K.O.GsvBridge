<script setup>
import { computed, onMounted, ref } from 'vue';
import { useRoute, useRouter, useI18n } from './nuxt-runtime.js';
import { u as useApi } from './C-oSG0qo.js';
import { _ as ParamSlider, a as ParamInput, T as languages, S as splitMethods, M as mediaTypes } from './CTDpj6JD.js';
import { MODEL_VERSIONS, modelDefaults, newVoice, normalizeVoice, preferredWeight, setAutomaticParameter, editorText } from './voice-editor.js';

const route = useRoute();
const router = useRouter();
const api = useApi();
const { t, locale } = useI18n();
const text = key => editorText(locale.value, key);
const isNew = computed(() => route.params.id === 'new');
const loading = ref(true);
const saving = ref(false);
const error = ref('');
const voice = ref(null);
const backend = computed(() => voice.value?.model.backend || (voice.value?.model.onnx_model_dir ? 'genie' : 'gsv'));
const defaults = computed(() => modelDefaults(voice.value?.model.version));
const modes = ref({ gpt: 'scan', sovits: 'scan', onnx: 'scan', audio: 'manual' });
const weights = ref({ gpt: [], sovits: [], onnx: [], audio: [] });
const scanning = ref({ gpt: false, sovits: false, onnx: false, audio: false });
const card = 'bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4';
const input = 'w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm';
const label = 'block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1';
const sectionTitle = 'text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3';
const sampling = [
  ['top_k', 'params.topK', 1, 100, 1], ['top_p', 'params.topP', 0, 1, 0.05],
  ['temperature', 'params.temperature', 0.01, 2, 0.05],
  ['repetition_penalty', 'params.repetitionPenalty', 1, 2, 0.05],
  ['speed_factor', 'params.speed', 0.25, 4, 0.05],
];
const flags = [
  ['split_bucket', 'config.splitBucket'], ['parallel_infer', 'config.parallelInfer'],
  ['super_sampling', 'config.superSampling'], ['streaming_mode', 'config.streaming'],
  ['return_fragment', 'config.returnFragment'], ['fixed_length_chunk', 'config.fixedChunk'],
];

function choices(kind, current) {
  const list = weights.value[kind];
  return current && !list.includes(current) ? [current, ...list] : list;
}

async function scan(kind, selectDefault = false) {
  if (!voice.value) return;
  const version = voice.value.model.version;
  scanning.value[kind] = true;
  try {
    let response;
    if (kind === 'gpt') response = await api.scanGptWeights(version);
    else if (kind === 'sovits') response = await api.scanSovitsWeights(version);
    else if (kind === 'onnx') response = await api.scanOnnxDirs();
    else response = await api.scanAudio();
    if (voice.value.model.version !== version && (kind === 'gpt' || kind === 'sovits')) return;
    weights.value[kind] = response.files || response.dirs || [];
    if (selectDefault && (kind === 'gpt' || kind === 'sovits')) {
      voice.value.model[`${kind}_weights`] = preferredWeight(weights.value[kind], version, kind);
    }
  } catch (failure) {
    error.value = failure?.data?.detail || failure.message;
  } finally {
    scanning.value[kind] = false;
  }
}

async function changeVersion(version) {
  voice.value.model.version = version;
  voice.value.model.gpt_weights = '';
  voice.value.model.sovits_weights = '';
  await Promise.all([scan('gpt', true), scan('sovits', true)]);
}

function setBackend(value) {
  voice.value.model.backend = value;
}

function automatic(name, checked) {
  setAutomaticParameter(voice.value.params, name, checked, voice.value.model.version);
}

function extractTranscript() {
  voice.value.ref_audio.prompt_text = voice.value.ref_audio.path.split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
}

async function save() {
  error.value = '';
  if (!voice.value.voice.id.trim()) {
    error.value = text('required');
    return;
  }
  saving.value = true;
  try {
    if (isNew.value) await api.createVoice(voice.value);
    else await api.updateVoice(route.params.id, voice.value);
    router.push('/config');
  } catch (failure) {
    error.value = failure?.data?.detail || failure.message;
  } finally {
    saving.value = false;
  }
}

onMounted(async () => {
  try {
    voice.value = isNew.value ? newVoice() : normalizeVoice(await api.getVoice(route.params.id));
    await Promise.all(['gpt', 'sovits', 'audio', 'onnx'].map(kind => scan(kind)));
  } catch (failure) {
    error.value = failure?.data?.detail || failure.message;
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <div class="space-y-6" data-testid="voice-editor">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <button type="button" class="text-sm text-gray-500 dark:text-gray-400" @click="router.push('/config')">{{ t('config.back') }}</button>
        <span class="text-gray-300 dark:text-gray-600">/</span>
        <h1 class="text-lg font-semibold text-gray-900 dark:text-gray-100">{{ isNew ? t('config.newProfile') : voice?.voice.name || route.params.id }}</h1>
      </div>
      <div class="flex items-center gap-2">
        <div v-if="voice" class="flex items-center gap-0.5 bg-gray-100 dark:bg-gray-700 rounded-lg p-0.5">
          <button v-for="value in ['gsv', 'genie']" :key="value" type="button"
            class="px-3 py-1.5 text-xs font-medium rounded-md" :class="backend === value ? 'bg-white dark:bg-gray-600 text-purple-700 dark:text-purple-300 shadow-sm' : 'text-gray-500 dark:text-gray-400'"
            @click="setBackend(value)">{{ value === 'gsv' ? '🔥 GSV' : '⚡ Genie' }}</button>
        </div>
        <button type="button" class="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg disabled:opacity-50" :disabled="saving || loading || !voice" data-testid="save-voice" @click="save">{{ t('common.save') }}</button>
      </div>
    </div>
    <p v-if="error" role="alert" class="text-sm text-red-500">{{ error }}</p>
    <div v-if="loading" class="text-center py-16 text-gray-400 text-sm">{{ text('loading') }}</div>
    <div v-else-if="voice" class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <div class="lg:col-span-1 space-y-4">
        <section :class="card">
          <h2 :class="sectionTitle">{{ t('config.voiceInfo') }}</h2>
          <div class="space-y-2">
            <div><label :class="label" for="voice-id">{{ t('config.voiceId') }}</label><input id="voice-id" v-model="voice.voice.id" :class="input" :disabled="!isNew" data-testid="voice-id"></div>
            <div><label :class="label" for="voice-name">{{ t('config.voiceName') }}</label><input id="voice-name" v-model="voice.voice.name" :class="input" data-testid="voice-name"></div>
            <div><label :class="label" for="voice-description">{{ t('config.voiceDesc') }}</label><textarea id="voice-description" v-model="voice.voice.description" :class="input" rows="2"></textarea></div>
          </div>
        </section>
        <section :class="card">
          <div class="flex items-center justify-between mb-3"><h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">{{ t('nav.models') }}</h2><span class="text-xs text-gray-500 dark:text-gray-400">{{ backend === 'gsv' ? '🔥 GSV' : '⚡ Genie' }}</span></div>
          <div class="space-y-2">
            <div>
              <label :class="label" for="model-version">{{ t('models.version') }}</label>
              <select id="model-version" :value="voice.model.version" :class="input" data-testid="model-version" @change="changeVersion($event.target.value)">
                <option v-for="version in MODEL_VERSIONS" :key="version" :value="version">{{ version === 'v5dev' ? 'V5 Dev' : version === 'v5turbo' ? 'V5 Turbo' : version }}</option>
              </select>
              <p class="text-xs text-gray-400 mt-2">{{ text('versionHelp') }}</p>
            </div>
            <template v-if="backend === 'gsv'">
              <div v-for="kind in ['gpt', 'sovits']" :key="kind">
                <div class="flex items-center justify-between mb-1">
                  <label :class="label" :for="`${kind}-weights`">{{ kind === 'gpt' ? 'GPT' : 'SoVITS' }}</label>
                  <div class="flex gap-1.5"><button v-for="mode in ['scan', 'manual']" :key="mode" type="button" class="text-xs px-2 py-0.5 rounded" :class="modes[kind] === mode ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100' : 'text-gray-400'" @click="modes[kind] = mode">{{ t(mode === 'scan' ? 'config.scanMode' : 'config.manualMode') }}</button></div>
                </div>
                <select v-if="modes[kind] === 'scan'" :id="`${kind}-weights`" v-model="voice.model[`${kind}_weights`]" :class="input" :disabled="scanning[kind]" :data-testid="`${kind}-weights`">
                  <option value="" disabled>{{ t(scanning[kind] ? 'common.loading' : 'config.selectFile') }}</option><option v-for="path in choices(kind, voice.model[`${kind}_weights`])" :key="path" :value="path">{{ path }}</option>
                </select>
                <input v-else :id="`${kind}-weights`" v-model="voice.model[`${kind}_weights`]" :class="input" :data-testid="`${kind}-weights`">
                <button type="button" class="text-xs text-indigo-500 mt-2" :disabled="scanning[kind]" @click="scan(kind)">↻ {{ t('config.scanMode') }}</button>
              </div>
            </template>
            <div v-else>
              <label :class="label" for="onnx-directory">ONNX</label>
              <select v-if="modes.onnx === 'scan'" id="onnx-directory" v-model="voice.model.onnx_model_dir" :class="input"><option value="" disabled>--</option><option v-for="path in choices('onnx', voice.model.onnx_model_dir)" :key="path" :value="path">{{ path }}</option></select>
              <input v-else id="onnx-directory" v-model="voice.model.onnx_model_dir" :class="input">
              <div class="flex gap-2 mt-2"><button type="button" class="text-xs text-indigo-500" @click="modes.onnx = modes.onnx === 'scan' ? 'manual' : 'scan'">{{ t(modes.onnx === 'scan' ? 'config.manualMode' : 'config.scanMode') }}</button><button type="button" class="text-xs text-indigo-500" @click="scan('onnx')">↻</button></div>
            </div>
          </div>
        </section>
        <section :class="card">
          <h2 :class="sectionTitle">{{ t('tts.refAudio') }}</h2>
          <div class="space-y-2">
            <div>
              <div class="flex items-center justify-between mb-1"><label :class="label" for="reference-path">{{ t('config.audioPath') }}</label><button type="button" class="text-xs text-indigo-500" @click="modes.audio = modes.audio === 'scan' ? 'manual' : 'scan'">{{ t(modes.audio === 'scan' ? 'config.manualMode' : 'config.scanMode') }}</button></div>
              <select v-if="modes.audio === 'scan'" id="reference-path" v-model="voice.ref_audio.path" :class="input" data-testid="reference-path"><option value="" disabled>--</option><option v-for="path in choices('audio', voice.ref_audio.path)" :key="path" :value="path">{{ path }}</option></select>
              <input v-else id="reference-path" v-model="voice.ref_audio.path" :class="input" data-testid="reference-path">
            </div>
            <div>
              <div class="flex items-center justify-between mb-1"><label :class="label" for="reference-transcript">{{ t('tts.promptText') }}</label><button type="button" class="text-xs text-indigo-500" @click="extractTranscript">{{ t('config.extractFilename') }}</button></div>
              <textarea id="reference-transcript" v-model="voice.ref_audio.prompt_text" :class="input" rows="2" data-testid="reference-transcript"></textarea>
            </div>
            <div><label :class="label" for="reference-language">{{ t('tts.promptLang') }}</label><select id="reference-language" v-model="voice.ref_audio.prompt_lang" :class="input"><option v-for="item in languages" :key="item.value" :value="item.value">{{ item.label }}</option></select></div>
          </div>
        </section>
      </div>
      <section class="lg:col-span-2 lg:self-start bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
        <h2 :class="sectionTitle">{{ t('params.title') }}</h2>
        <div class="space-y-3">
          <h3 class="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">{{ t('config.sampling') }}</h3>
          <div class="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
            <ParamSlider v-for="[name, key, min, max, step] in sampling" :key="name" v-model="voice.params[name]" :label="t(key)" :min="min" :max="max" :step="step" />
            <ParamInput v-model="voice.params.seed" :label="t('params.seed')" type="number" :min="-1" />
          </div>
          <div class="border-t border-gray-100 dark:border-gray-700/50"></div>
          <h3 class="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">{{ t('config.textProcessing') }}</h3>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div><label :class="label" for="text-language">{{ t('tts.textLang') }}</label><select id="text-language" v-model="voice.params.text_lang" :class="input"><option v-for="item in languages" :key="item.value" :value="item.value">{{ item.label }}</option></select></div>
            <div><label :class="label" for="text-split">{{ t('params.splitMethod') }}</label><select id="text-split" v-model="voice.params.text_split_method" :class="input"><option v-for="item in splitMethods" :key="item.value" :value="item.value">{{ item.label }}</option></select></div>
            <ParamSlider v-model="voice.params.batch_size" :label="t('params.batchSize')" :min="1" :max="32" :step="1" />
            <ParamSlider v-model="voice.params.fragment_interval" :label="t('params.fragmentInterval')" :min="0" :max="1" :step="0.05" />
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div v-for="name in ['sample_steps', 'cfg_rate']" :key="name" :data-testid="`${name}-control`">
              <label :class="label">{{ name === 'sample_steps' ? t('params.sampleSteps') : text('cfg') }}</label>
              <label class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 mb-2">
                <input type="checkbox" :checked="voice.params[name] == null" :data-testid="`${name}-auto`" @change="automatic(name, $event.target.checked)">
                <span>{{ text('auto') }}</span>
                <span v-if="voice.params[name] == null" :data-testid="`${name}-default`">{{ text('autoValue') }}: {{ defaults[name] }}</span>
              </label>
              <ParamSlider v-if="voice.params[name] != null" v-model="voice.params[name]" :label="text('custom')" :min="name === 'sample_steps' ? 1 : 0" :max="name === 'sample_steps' ? 100 : 2" :step="name === 'sample_steps' ? 1 : 0.05" />
            </div>
          </div>
          <p class="text-xs text-gray-400">{{ text('autoHelp') }}</p>
          <div v-if="backend === 'gsv'" class="space-y-2">
            <h3 class="text-xs font-medium text-gray-400 uppercase tracking-wide">{{ text('acceleration') }}</h3>
            <label v-for="[name, key] in [['use_cuda_graph', 'cuda'], ['use_flash_attention', 'flash']]" :key="name" class="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400"><input :checked="voice.params[name]" type="checkbox" :data-testid="name" @change="voice.params[name] = $event.target.checked"><span>{{ text(key) }}</span></label>
          </div>
          <div class="border-t border-gray-100 dark:border-gray-700/50"></div>
          <h3 class="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">{{ t('config.flags') }}</h3>
          <div class="grid grid-cols-2 sm:grid-cols-3 gap-x-3 gap-y-1.5">
            <label v-for="[name, key] in flags" :key="name" class="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400"><input :checked="voice.params[name]" type="checkbox" @change="voice.params[name] = $event.target.checked"><span>{{ t(key) }}</span></label>
          </div>
          <div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700/50"><label :class="label" for="media-type">{{ t('config.mediaType') }}</label><select id="media-type" v-model="voice.output.media_type" :class="input"><option v-for="item in mediaTypes" :key="item.value" :value="item.value">{{ item.label }}</option></select></div>
        </div>
      </section>
    </div>
    <p v-else role="alert" class="text-center py-16 text-gray-400 text-sm">{{ text('missing') }}</p>
  </div>
</template>
