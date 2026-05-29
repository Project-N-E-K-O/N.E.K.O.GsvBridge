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

所有后端实现统一的 `BasePipeline` 接口（`run`、`init_t2s_weights`、`init_vits_weights`、`get_info`、`validate_params`），`InferenceEngine` 据此统一调度，因此新增后端无需改动引擎与路由。

### 前端托管

`server.py` 自动挂载 `frontend/.output/public` 下的 Nuxt 静态构建产物，并以 SPA fallback 方式处理客户端路由。若该目录不存在，启动时会打印警告，需先在 `frontend/` 下执行 `npx nuxt generate` 构建。
