"""
GPT-SoVITS API v3 - 健康检查路由
"""

from fastapi import APIRouter, Request

from api_neko.config import alist_voices


router = APIRouter(prefix="/api/v3", tags=["health"])


@router.get("/health", summary="健康检查")
async def health_check(request: Request):
    voices = await alist_voices()
    backend = getattr(request.app.state, 'backend_name', 'gsv')
    return {
        "status": "ok",
        "version": "3.0.0",
        "backend": backend,
        "voices_count": len(voices),
    }
