export const MODEL_VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v2Pro', 'v2ProPlus', 'v5dev', 'v5turbo'];

export function modelDefaults(version) {
  return {
    sample_steps: version === 'v5turbo' ? 4 : 32,
    cfg_rate: version === 'v5dev' ? 1.3 : 0,
  };
}

export function newVoice() {
  return {
    voice: { id: '', name: '', description: '' },
    model: { gpt_weights: '', sovits_weights: '', version: 'v2', onnx_model_dir: '', backend: '' },
    ref_audio: { path: '', prompt_text: '', prompt_lang: 'all_zh', aux_ref_audio_paths: [] },
    params: {
      text_lang: 'all_zh', text_split_method: 'cut5', top_k: 15, top_p: 1, temperature: 1,
      repetition_penalty: 1.35, seed: -1, batch_size: 1, batch_threshold: 0.75,
      split_bucket: true, parallel_infer: true, speed_factor: 1, fragment_interval: 0.3,
      sample_steps: null, cfg_rate: null, use_cuda_graph: true, use_flash_attention: true,
      super_sampling: false, streaming_mode: false, return_fragment: false,
      overlap_length: 2, min_chunk_length: 16, fixed_length_chunk: false,
    },
    output: { media_type: 'wav' },
  };
}

export function normalizeVoice(saved) {
  const defaults = newVoice();
  return Object.fromEntries(Object.keys(defaults).map(section => [section, {
    ...defaults[section], ...saved[section],
  }]));
}

export function preferredWeight(files, version, type) {
  const suffix = type === 'gpt' ? 's1v3.ckpt' : `gsv-v5-pretrained/s2G${version}.pth`;
  if (version.startsWith('v5')) {
    const official = files.find(path => path.replaceAll('\\', '/').endsWith(suffix));
    if (official) return official;
  }
  return files[0] || '';
}

export function setAutomaticParameter(params, name, automatic, version) {
  params[name] = automatic ? null : modelDefaults(version)[name];
}

const labels = {
  'zh-CN': {
    auto: '跟随模型默认', autoValue: '当前默认', custom: '自定义', cfg: 'CFG 引导强度',
    acceleration: '推理加速', cuda: 'CUDA Graph', flash: 'FlashAttention',
    autoHelp: '自动值随模型版本变化；已有的自定义值会保留。',
    versionHelp: '切换版本会重新选择匹配的权重，参考音频和声音信息保持不变。',
    retainedHelp: '当前保留自定义参数；右侧显示新模型默认值。勾选“跟随模型默认”可切换，显式值不会自动重置。',
    loading: '正在加载配置…', missing: '配置不存在或加载失败', required: '请填写声音 ID',
    requiredWeights: '请等待扫描完成并选择 GPT 和 SoVITS 权重',
  },
  en: {
    auto: 'Use model default', autoValue: 'Current default', custom: 'Custom', cfg: 'CFG guidance',
    acceleration: 'Inference acceleration', cuda: 'CUDA Graph', flash: 'FlashAttention',
    autoHelp: 'Automatic values follow the model version. Existing custom values are preserved.',
    versionHelp: 'Changing the version selects matching weights and keeps the reference audio and voice information.',
    retainedHelp: 'Custom parameters are retained. The new model default is shown beside them; select Use model default to switch. Explicit values are not reset automatically.',
    loading: 'Loading configuration…', missing: 'Configuration not found or could not be loaded', required: 'Enter a voice ID',
    requiredWeights: 'Wait for scanning to finish and select both GPT and SoVITS weights',
  },
  ja: {
    auto: 'モデルの既定値', autoValue: '現在の既定値', custom: 'カスタム', cfg: 'CFG ガイダンス',
    acceleration: '推論高速化', cuda: 'CUDA Graph', flash: 'FlashAttention',
    autoHelp: '自動値はモデルに従います。既存のカスタム値は保持されます。',
    versionHelp: 'バージョン変更時に対応する重みを選択し、参照音声と音声情報を保持します。',
    retainedHelp: 'カスタム値を保持します。新モデルの既定値を併記しています。切り替えるには「モデルの既定値」を選択してください。明示した値は自動リセットしません。',
    loading: '設定を読み込み中…', missing: '設定が存在しないか読み込みに失敗しました', required: '音声 ID を入力してください',
    requiredWeights: 'スキャン完了後、GPT と SoVITS の重みを選択してください',
  },
  ko: {
    auto: '모델 기본값 사용', autoValue: '현재 기본값', custom: '사용자 지정', cfg: 'CFG 가이던스',
    acceleration: '추론 가속', cuda: 'CUDA Graph', flash: 'FlashAttention',
    autoHelp: '자동 값은 모델 버전을 따릅니다. 기존 사용자 지정 값은 유지됩니다.',
    versionHelp: '버전을 변경하면 일치하는 가중치를 선택하고 참조 오디오와 음성 정보를 유지합니다.',
    retainedHelp: '사용자 지정 값은 유지됩니다. 새 모델의 기본값이 함께 표시됩니다. 기본값 사용을 선택하면 전환되며 명시적인 값은 자동 초기화하지 않습니다.',
    loading: '설정 불러오는 중…', missing: '설정이 없거나 불러오지 못했습니다', required: '음성 ID를 입력하세요',
    requiredWeights: '스캔이 완료된 후 GPT와 SoVITS 가중치를 모두 선택하세요',
  },
};

export function editorText(locale, key) {
  return (labels[locale] || labels.en)[key] || key;
}
