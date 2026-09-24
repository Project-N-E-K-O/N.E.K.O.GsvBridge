"""
GPT-SoVITS API v3 - FastAPI 后端服务

提供声音配置管理和 TTS 推理接口。
路由拆分到 routers/ 目录下，通过 include_router 挂载。

支持多后端：通过 settings.toml 中的 backend 字段切换
- "gsv"（默认）: GPT-SoVITS 原项目 PyTorch 推理
- "genie": Genie-TTS ONNX 轻量推理（待实现）
"""

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 路径初始化：从 config.py 集中管理的路径配置中获取
from api_neko.config import PROJECT_ROOT, GSV_PACKAGE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GSV_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(GSV_PACKAGE_DIR))

from contextlib import asynccontextmanager

from api_neko.routers.config import router as config_router
from api_neko.routers.health import router as health_router
from api_neko.routers.tts import router as tts_router
from api_neko.routers.tts_v3 import router as tts_v3_router


# ─── TTS Pipeline 初始化 ───

_tts_config_path: str = str(PROJECT_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml")


def _detect_backend() -> str:
    """从 settings.toml 读取 backend 配置，默认 'gsv'"""
    from api_neko.config import load_settings
    try:
        settings = load_settings()
        return settings.get("backend", "gsv")
    except Exception:
        return "gsv"


def _create_pipeline(backend: str):
    """根据 backend 类型创建对应的 TTS pipeline

    Returns:
        (pipeline, cut_method_names) 元组
    """
    if backend == "genie":
        from api_neko.backends.genie_backend import GeniePipeline
        pipeline = GeniePipeline()
        return pipeline, pipeline.cut_method_names
    else:
        # 默认: GSV PyTorch 后端
        from api_neko.backends.gsv_backend import GSVPipeline
        pipeline = GSVPipeline(_tts_config_path)

        print("\n"*5)
        addr = "http://" +("localhost" if args.host == "0.0.0.0" else args.host)+f":{args.port}"
        print("地址 ( 按着'ctrl' 鼠标点击 可直接跳转 ) :",addr)
        print("\n"*5)
        return pipeline, pipeline.cut_method_names


def _auto_load_last_voice(pipeline, backend: str) -> tuple[str, str]:
    """根据 settings.toml 中的 last_voice_id 自动加载上次使用的模型。
    返回 (gpt_weights, sovits_weights) 路径，供 InferenceEngine 同步。"""
    from api_neko.config import load_settings, load_voice
    gpt_w, sovits_w = "", ""
    try:
        settings = load_settings()
        voice_id = settings.get("last_voice_id", "_default")
        if voice_id and voice_id != "_default":
            cfg = load_voice(voice_id)

            if backend == "genie" and cfg.model.onnx_model_dir:
                # Genie 模式：使用 onnx_model_dir 加载 ONNX 模型
                print(f"[startup] 自动加载 ONNX 模型目录: {cfg.model.onnx_model_dir}")
                pipeline.init_vits_weights(cfg.model.onnx_model_dir)
                sovits_w = cfg.model.onnx_model_dir
            else:
                # GSV 模式：分别加载 GPT 和 SoVITS 权重
                if cfg.model.gpt_weights:
                    print(f"[startup] 自动加载 GPT 权重: {cfg.model.gpt_weights}")
                    pipeline.init_t2s_weights(cfg.model.gpt_weights)
                    gpt_w = cfg.model.gpt_weights
                if cfg.model.sovits_weights:
                    print(f"[startup] 自动加载 SoVITS 权重: {cfg.model.sovits_weights}")
                    pipeline.init_vits_weights(cfg.model.sovits_weights)
                    sovits_w = cfg.model.sovits_weights

            print(f"[startup] 已加载声音配置: {voice_id}")
    except Exception as e:
        print(f"[startup] 自动加载声音配置失败: {e}")
    return gpt_w, sovits_w


@asynccontextmanager
async def lifespan(app_instance):
    """FastAPI lifespan: 启动时初始化 TTS pipeline、推理引擎，并自动加载上次使用的模型"""
    backend = _detect_backend()
    print(f"[startup] 使用后端: {backend}")

    pipeline, cut_method_names = _create_pipeline(backend)
    app_instance.state.tts_pipeline = pipeline
    app_instance.state.cut_method_names = cut_method_names
    app_instance.state.backend_name = backend

    loaded_gpt, loaded_sovits = _auto_load_last_voice(pipeline, backend)

    # 创建并启动推理引擎
    from api_neko.inference import InferenceEngine
    engine = InferenceEngine(
        pipeline,
        pipeline,  # pipeline 自身实现了 validate_params
        cut_method_names,
    )
    # 同步启动时已加载的模型路径，避免首次推理重复加载
    if loaded_gpt:
        engine._current_gpt = loaded_gpt
    if loaded_sovits:
        engine._current_sovits = loaded_sovits
    await engine.start()
    app_instance.state.inference_engine = engine

    yield

    await engine.stop()


app = FastAPI(
    title="GPT-SoVITS API v3",
    description="文本转语音合成 API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 挂载路由 ───
app.include_router(config_router)
app.include_router(health_router)
app.include_router(tts_router)
app.include_router(tts_v3_router)


# ─── 挂载前端静态文件 ───
_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / ".output" / "public"

if _FRONTEND_DIST.is_dir():
    # 挂载 _nuxt/ 等静态资源（带缓存）
    app.mount("/_nuxt", StaticFiles(directory=_FRONTEND_DIST / "_nuxt"), name="nuxt-assets")
    # 挂载 public 根目录下的静态文件（favicon.ico, robots.txt 等）
    app.mount("/static-root", StaticFiles(directory=_FRONTEND_DIST), name="static-root")

    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse(_FRONTEND_DIST / "favicon.ico")

    @app.get("/{full_path:path}")
    async def spa_fallback(request: Request, full_path: str):
        """SPA fallback: 非 API 路径都返回 index.html，由 Nuxt 客户端路由处理"""
        # 尝试精确匹配静态文件（如 /tts/index.html）
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # 尝试目录下的 index.html（如 /tts -> /tts/index.html）
        index_path = file_path / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        # 兜底返回根 index.html（SPA 客户端路由）
        return FileResponse(_FRONTEND_DIST / "index.html")
else:
    print(f"[warning] 前端静态文件目录不存在: {_FRONTEND_DIST}")
    print(f"[warning] 请先在 api_neko/frontend/ 下执行 npx nuxt generate 构建前端")


def main():
    import argparse
    import uvicorn
    global args
    global _tts_config_path
    parser = argparse.ArgumentParser(description="GPT-SoVITS API v3")
    parser.add_argument("-p", "--port", type=int, default=9881, help="监听端口 (默认: 9881)")
    parser.add_argument("-a", "--host", type=str, default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("-c", "--tts_config", type=str, default=_tts_config_path, help="TTS 配置文件路径")
    args = parser.parse_args()

    _tts_config_path = args.tts_config

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
