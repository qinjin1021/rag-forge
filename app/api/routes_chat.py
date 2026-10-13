"""问答接口: 流式SSE + 同步JSON两种形态。"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.rag_agent import RAGAgent
from app.api.deps import get_rag_agent
from app.core.logging import get_logger
from app.core.schemas import ChatRequest, ChatResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["问答"])


@router.post("/stream")
async def chat_stream(
    request: ChatRequest, agent: RAGAgent = Depends(get_rag_agent)
) -> StreamingResponse:
    """流式问答(SSE)。事件类型:
    stage   - 流程阶段(analyze/retrieve/generate)
    rewrite - Agent改写后的检索问题
    sources - 引用来源列表
    delta   - 增量回答文本
    done    - 结束(含耗时)
    error   - 错误
    """

    async def event_generator():
        try:
            async for event in agent.chat_stream(request.session_id, request.query):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("流式问答异常")
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("", response_model=ChatResponse)
async def chat_sync(
    request: ChatRequest, agent: RAGAgent = Depends(get_rag_agent)
) -> ChatResponse:
    """同步问答(一次性返回完整结果, 供脚本调用/压测/评估)"""
    try:
        return await agent.chat(request.session_id, request.query)
    except Exception as exc:
        logger.exception("同步问答失败")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str, agent: RAGAgent = Depends(get_rag_agent)
) -> dict:
    """清空会话历史(开始新话题)"""
    agent.sessions.clear(session_id)
    return {"ok": True, "session_id": session_id}
