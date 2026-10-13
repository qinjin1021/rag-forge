"""FastAPI应用入口。

职责: 组装所有组件(生命周期管理) + 注册路由/中间件/异常处理 + 托管前端页面。

启动: uvicorn app.main:app --host 0.0.0.0 --port 8000
接口文档: http://localhost:8000/docs (Swagger自动生成)
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_chat, routes_docs
from app.api.deps import get_retriever, get_vector_store
from app.config import get_settings
from app.core.logging import get_logger, new_trace_id, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时预热组件, 关闭时释放资源"""
    settings = get_settings()
    setup_logging(settings.debug)
    logger.info("%s v%s 启动中...", settings.app_name, settings.app_version)
    store = get_vector_store()
    if store.count() > 0:
        # 已有知识库 -> 预热BM25索引与检索链路, 首次请求不冷启动
        get_retriever().refresh()
        logger.info("知识库已加载: %d块", store.count())
    else:
        logger.warning("知识库为空, 请上传文档或执行: python scripts/ingest.py")
    yield
    logger.info("应用已停止")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="基于DeepSeek + 混合检索 + Agent编排的企业知识库问答系统",
        lifespan=lifespan,
    )

    # 跨域(前后端分离部署时需要; 演示阶段同源也可放宽)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        """链路追踪: 每个请求分配trace_id, 注入日志并返回响应头"""
        trace_id = new_trace_id()
        start = time.perf_counter()
        response = await call_next(request)
        cost = int((time.perf_counter() - start) * 1000)
        response.headers["X-Trace-Id"] = trace_id
        logger.info("%s %s -> %d (%dms)", request.method, request.url.path,
                    response.status_code, cost)
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常兜底: 统一错误结构并携带trace_id, 方便用户反馈问题时定位日志"""
        logger.exception("未处理异常: %s", exc)
        return JSONResponse(status_code=500, content={
            "detail": "服务内部错误, 请稍后重试",
            "trace_id": request.headers.get("X-Trace-Id", "-"),
        })

    @app.get("/healthz", tags=["运维"])
    async def healthz() -> dict:
        """健康检查(Docker/k8s探针与负载均衡使用)"""
        store = get_vector_store()
        return {"status": "ok", "chunks": store.count(),
                "docs": len(store.list_sources())}

    app.include_router(routes_chat.router)
    app.include_router(routes_docs.router)

    # 托管前端静态页面(聊天界面)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


app = create_app()
