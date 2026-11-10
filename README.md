# api_neko —— GPT-SoVITS TTS API v3

基于 FastAPI 的文本转语音（TTS）后端服务，封装 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 推理能力，提供声音配置管理、队列化推理、流式合成和内置 Web 前端。

支持两种推理后端，通过配置一键切换：

- **`gsv`**（默认）：GPT-SoVITS 原生 PyTorch 推理
- **`genie`**：[Genie-TTS](https://github.com/) 的 ONNX 轻量推理

## 项目定位

`api_neko` 是一个 **Python 包**，设计为放置在 GPT-SoVITS 项目根目录下作为子目录运行（与 `GPT_SoVITS/`、权重目录、`runtime/` 等同级）。它依赖宿主项目提供：

- `GPT_SoVITS/` 推理包及其 `configs/tts_infer.yaml`
- 预训练模型 `GPT_SoVITS/pretrained_models/`
- 各版本权重目录（`GPT_weights_v2/`、`SoVITS_weights_v2/` 等）
- `genie` 后端额外需要 `Genie-TTS/` 及其 ONNX 模型

典型的宿主目录结构：

```
<project_root>/
├── GPT_SoVITS/                # GPT-SoVITS 推理包
├── GPT_weights_v2/            # GPT 权重 (.ckpt)
├── SoVITS_weights_v2/         # SoVITS 权重 (.pth)
├── runtime/                   # 整合包自带 Python (Windows)
├── Genie-TTS/                 # 可选，genie 后端
└── api_neko/                  # ← 本项目
```

## 目录结构

```
api_neko/
├── entry.py              # 启动入口（注入 sys.path 后调用 server.main）
├── server.py             # FastAPI 应用：生命周期、路由挂载、前端托管
├── config.py             # 声音配置 / settings.toml 管理，VoiceConfig 数据模型
├── inference.py          # 推理引擎：单 Worker 双队列，模型热切换
├── audio_utils.py        # 音频打包（WAV/raw header）
├── _toml_compat.py       # TOML 读写兼容层
├── settings.toml         # 运行时设置（当前后端、上次使用的声音等）
├── backends/
│   ├── base.py           # BasePipeline 抽象基类
│   ├── gsv_backend.py    # GSV PyTorch 后端
│   └── genie_backend.py  # Genie ONNX 后端
├── routers/
│   ├── config.py         # 声音 CRUD、模型扫描、PyTorch→ONNX 转换
│   ├── health.py         # 健康检查
│   ├── tts.py            # v2 兼容推理接口 (/api/v2)
│   └── tts_v3.py         # v3 队列 / WebSocket 流式接口 (/api/v3)
├── voices/               # 声音配置（每个 .toml 一个角色）+ 参考音频
└── frontend/.output/     # 预构建的 Nuxt 静态前端（SPA）
```

## 快速开始

### 依赖

- Python 3.9+
- `fastapi`、`uvicorn`、`pydantic`、`numpy`，以及 GPT-SoVITS 自身的推理依赖
- （Windows 整合包用户：直接用自带的 `runtime/python.exe`，无需额外安装）

### 启动

从**项目根目录**运行（不是 `api_neko/` 内部），以保证 `api_neko` 包可被导入：

```bash
# 方式一：作为模块启动
python -m api_neko.server -p 9881

# 方式二：通过入口脚本（自动注入 sys.path）
python api_neko/entry.py
```

启动参数（`server.py`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-p`, `--port` | `9881` | 监听端口 |
| `-a`, `--host` | `0.0.0.0` | 监听地址 |
| `-c`, `--tts_config` | `GPT_SoVITS/configs/tts_infer.yaml` | TTS 配置文件路径 |

启动后访问 `http://localhost:9881/` 打开内置 Web 界面。

### Windows 一键启动

- `go-webui.bat`：用整合包 `runtime/` 启动 api_neko 服务（9881）并拉起原 WebUI
- `neko_go-webui.bat`：向上回溯查找项目根，用 `runtime\python.exe` 运行 `entry.py`

## 配置

### settings.toml

服务级运行时设置：

```toml
last_voice_id = "kelala"   # 启动时自动加载的声音
backend = "gsv"            # 推理后端: "gsv" | "genie"
# project_root = ""        # 留空则自动推断为 api_neko 的父目录
# gsv_package_dir = ""     # 留空则推断为 <project_root>/GPT_SoVITS
```

切换后端需修改 `backend` 字段并**重启服务**。

### 声音配置（voices/*.toml）

每个声音角色对应 `voices/` 下的一个 `.toml` 文件，包含模型权重、参考音频和推理参数。加载时会自动合并 `voices/default.toml` 作为默认值。文件名以 `_` 开头的为内部文件（如 `default.toml`），不出现在声音列表中。

声音配置结构（以仓库自带的 `voices/init.toml` 为例）：

```toml
[voice]
id = "init"
name = "Init"
description = ""

[model]
gpt_weights = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
sovits_weights = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
version = "v3"
onnx_model_dir = ""        # genie 后端使用
backend = ""               # 留空则跟随全局 settings.toml

[ref_audio]
path = "api_neko/voices/嗯，不知道为什么，我也有一种能做到的感觉.wav"
prompt_text = "嗯，不知道为什么，我也有一种能做到的感觉"
prompt_lang = "all_zh"
aux_ref_audio_paths = []

[params]
text_lang = "all_zh"
text_split_method = "cut5"
top_k = 15
top_p = 1.0
temperature = 1.0
repetition_penalty = 1.35
seed = -1
batch_size = 1
batch_threshold = 0.75
split_bucket = true
parallel_infer = true
speed_factor = 1.0
fragment_interval = 0.3
sample_steps = 32
super_sampling = false
streaming_mode = false
return_fragment = false
overlap_length = 2
min_chunk_length = 16
fixed_length_chunk = false

[output]
media_type = "wav"
```

相对路径均以 `project_root` 为基准解析为绝对路径。

## V5 兼容与配置边界

Bridge 继续使用原有 `/api/v3/voices` 和 `/api/v3/tts/stream-input`，NEKO 本体无需修改。
宿主需包含 V5 的 `TTS_infer_pack`、模型版本识别与配套声码器；Bridge 不安装或修改宿主。

- 支持 `v5dev`、`v5turbo`，模型扫描复用宿主的权重识别函数（未安装宿主的独立 checkout
  仍可按官方文件名发现模型）。两种官方底模配对共享 `s1v3.ckpt`，同名训练 GPT 优先。
- `sample_steps`、`cfg_rate` 未指定时传递宿主自动值：官方 V5 Turbo 默认 4 步 / CFG 0，
  V5 Dev 默认 32 步 / CFG 1.30。旧文件显式保存的数值继续生效。
- 支持 `use_cuda_graph`、`use_flash_attention`，默认启用，实际可用能力由宿主决定。
  上述 V5 参数适用于声音配置、v3 REST、两个 v3 WebSocket 入口和 v2 的 GET/POST。
- 参数优先级为请求显式覆盖 → 声音配置（含 `voices/default.toml` 公共默认值）→ 宿主默认。
  请求 JSON 可用 `null` 临时恢复 `sample_steps` / `cfg_rate` 的宿主默认；保存声音时省略自动值，
  因为 TOML 没有 null。重新读取时仍先合并公共默认配置。旧编辑器未提交的新字段保留原值。
- 通用示例位于 [`examples/voices/v5turbo.toml`](examples/voices/v5turbo.toml) 和
  [`examples/voices/v5dev.toml`](examples/voices/v5dev.toml)，不会自动注册为可用音色。
  需要用户在本机补充参考音频及准确文本，再通过配置 API 创建或复制到 `voices/`。

`settings.toml` 可选 `default_voice_id` 用于解析 `_default`。它与 `last_voice_id`（启动时预加载）
独立；未设置时保留原 `voices/default.toml` 行为。具体默认 ID、机器路径、用户音色 TOML、
权重和参考音频均属于本机数据，不应加入 Git。公共示例和通用代码、测试才属于仓库交付。

### 配置管理界面

声音编辑页提供 `V5 Dev`、`V5 Turbo` 选项。切换版本会扫描并选择对应 GPT/SoVITS 权重，
自动采样步数/CFG 显示当前模型的默认值；关闭自动后可以输入自定义值。
CUDA Graph 和 FlashAttention 可以独立选择。保存并重新打开后保留版本、参考音频和参数选择。
已有声音的显式参数继续保留，不会因升级或切换版本被默默重置。

可编辑源码、构建方法和分发集成说明见 [`frontend/README.md`](frontend/README.md)。
执行 `npm ci && npm run build` 重建编辑页，其余现有页面保持原有分发。
构建使用内容哈希和 import map 更新路由，静态托管对 HTML/JS 要求重新验证缓存。
更新正在运行的服务文件后需完整刷新浏览器页面；后端代码更新需重启 Bridge。

### NEKO 音频与任务生命周期

- `stream-input` 固定输出独立的 **48 kHz / 单声道 / PCM16 WAV** 帧，每帧最多一秒音频。
  旧采样率在 Bridge 内通过宿主已有的 `soxr` 重采样，正常结束前 flush 尾部；不新增运行时安装步骤。
  通用 REST 和 `/tts/stream` 保留源采样率与媒体类型选择。
- 保留 `init/append/text/flush/end`、`ready/sentence/sentence_done/flushed/done` 消息。
  `sentence` 在首块音频可用时才发送；合成前失败返回 `error` 并关闭，不再提前表示合成已接受。
  `end` 等待音频发完，再发送 `done` 并关闭。`flushed` 表示尾文本已提交，不表示音频全部发完。
- 断连只取消该连接提交的任务。空的初始化验证连接不会停止其他客户端；排队的取消任务不再执行。
  运行中的任务请求后端协作停止，丢弃迟到输出，并等待推理线程真正结束后才开始下一任务或换模型。
  GSV 的停止检查位于宿主推理边界，不能立即抢占正在执行的 GPU kernel。
- 流式任务完成/断连后释放任务和音频；REST 默认保留最近 100 个完成任务，较旧 ID 返回 404。
  模型切换使参考音频缓存失效，使用相同参考音频路径也会重新提取特征。
- v2 推理和手动模型切换也进入同一 Worker 队列；手动选择的权重与 v3 当前模型独立维护。
  v3 请求不会继承管理页面临时切换的模型，v2 也会恢复自己的选定权重。
  下载完整音频使用真实截止时间，连续音频更新不会被误计为等待超时。

本次兼容保持按文本段推理。`streaming_mode` / `return_fragment` 会传递给宿主，实际输出能力
取决于宿主；仅新增参数不能恢复宿主缺失的 `gpt_streaming` 语义 token 流式。
NEKO 的连接测试约 10 秒超时、收到部分音频后仍可能判成功，以及 `init` 与 `_default` 的
默认 ID 差异仍是客户端既有行为。Bridge 不通过改写 `init` 音色或伪造成功来绕过这些条件。

### 测试

在开发环境安装 `pytest`、`httpx` 和宿主提供的轻量运行依赖后，从仓库根目录运行：

```bash
python -m pytest tests -q
```

默认测试使用临时目录、模拟后端和进程内 WebSocket，不需要本机音色、模型、GPU 或监听端口。
编辑页还有 Node 状态测试和可选隔离浏览器验收，运行方法见 `frontend/README.md`。
可选真实宿主测试通过 `GSV_TEST_HOST_ROOT` 指定宿主位置，使用仓库已有公共参考音频和临时
推理配置，验证 V2ProPlus → Turbo → Dev → V2ProPlus 的协议与模型切换；不改写用户配置。

```powershell
$env:GSV_TEST_HOST_ROOT = 'C:\path\to\GPT-SoVITS'
# 在宿主下的 api_neko 仓库根目录，使用宿主解释器和标准库测试运行器。
# 测试程序通过变量定位宿主，不把该路径保存进仓库，也无需给宿主安装 pytest。
& "$env:GSV_TEST_HOST_ROOT\runtime\python.exe" -m unittest discover -s tests -p test_gsv_host_integration.py -q
```

## API 概览

### 健康检查

- `GET /api/v3/health` —— 状态、版本、当前后端、声音数量

### v3 推理（队列 + 流式）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v3/tts` | 提交推理任务，返回 `task_id` |
| `GET` | `/api/v3/tts/{id}` | 查询任务状态 |
| `GET` | `/api/v3/tts/{id}/audio` | 阻塞等待并返回完整音频 |
| `GET` | `/api/v3/tts/queue` | 查看队列状态 |
| `WS` | `/api/v3/tts/stream` | 提交并实时流式接收音频 |
| `WS` | `/api/v3/tts/stream-input` | 双向流式：文本碎片流入、自动切句、音频流出 |

`POST /api/v3/tts` 请求体示例：

```json
{
  "text": "你好，世界",
  "voice_id": "kelala",
  "speed_factor": 1.0,
  "media_type": "wav"
}
```

### v3 配置管理

- 声音 CRUD：`GET/POST/PUT/DELETE /api/v3/voices`、`GET /api/v3/voices/full`、`POST /api/v3/voices/batch-delete`
- 默认配置：`GET /api/v3/config/default`、`POST /api/v3/config/reload`
- 用户设置：`GET/PUT /api/v3/settings`
- 模型扫描：`GET /api/v3/models`、`GET /api/v3/scan/{gpt-weights,sovits-weights,audio}`、`GET /api/v3/models/scan/onnx`
- 模型转换：`POST /api/v3/models/convert`（PyTorch → Genie ONNX，后台异步）、`GET /api/v3/models/convert[/{task_id}]`

### v2 兼容接口

为兼容 GPT-SoVITS 原 v2 API 保留：

- `GET/POST /api/v2/tts` —— 直接传参推理（支持 `streaming_mode` 0/1/2/3）
- `GET /api/v2/set_gpt_weights`、`GET /api/v2/set_sovits_weights` —— 手动切换模型权重

## 架构说明

### 推理引擎（inference.py）

采用 **单 Worker + 双队列** 架构：

```
请求 → Input Queue → Worker（串行推理）→ 每任务独立 OutputBuffer → 流式/一次性获取
```

- 单 Worker 串行执行，避免 GPU 并发冲突
- 推理前自动检测并热切换 GPT / SoVITS 模型，命中当前已加载模型则跳过
- 每个任务有独立的 `OutputBuffer` 与 `asyncio.Event`，同时支持流式推送和等待完整结果

### 后端抽象（backends/base.py）

所有后端实现统一的 `BasePipeline` 接口（`run`、`init_t2s_weights`、`init_vits_weights`、`get_info`、`validate_params`、`get_model_paths`、`stop`）。`get_model_paths` 告知引擎启动时已加载的权重，`stop` 提供协作停止；`InferenceEngine` 统一调度模型切换和推理。

### 前端托管

`server.py` 将 `frontend/.output/public` 挂载为独立的前端应用。真实文件由 Starlette 的 `StaticFiles` 托管，只有构建产物未预渲染的 `/config/new` 和 `/config/{id}` 页面使用 Nuxt 的 `200.html`。缺失资源、未知页面和未知 `/api` 路径返回 404。资源 MIME 在 ASGI 响应头处固定，不依赖 Windows 注册表。声音编辑页可从仓库中的 Vue 源码重建，其他 Nuxt 页面继续使用仓库已有分发；不要删除分发目录后尝试仅凭编辑页源码重建整个应用。
