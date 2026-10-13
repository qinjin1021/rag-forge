"""RAG Agent核心编排: 意图识别 -> 查询改写 -> 混合检索 -> 重排 -> 带引用生成。

这是整个系统的"大脑", 把各组件按流程串起来, 并以事件流(stream events)驱动前端展示。

Agent能力体现:
1. 自主决策: 判断问题是否需要检索(闲聊直接回答, 省钱省时延)
2. 查询理解: 多轮对话中的指代消解与查询改写
3. 工具调用: 调用混合检索工具获取知识
4. 可信生成: 强制引用来源 + 资料不足时拒答(企业落地的合规底线)
5. 过程透明: 每个阶段以事件推送前端, 用户能看到Agent"怎么想的"
"""
import json
import re
import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.agent.prompts import CHITCHAT_PROMPT, GENERATION_PROMPT, QUERY_ANALYSIS_PROMPT, build_context
from app.agent.session import SessionStore
from app.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import ChatResponse, SourceRef
from app.retrieval.reranker import BaseReranker
from app.retrieval.retriever import HybridRetriever

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class RAGAgent:
    """RAG问答Agent"""

    def __init__(self, retriever: HybridRetriever, reranker: BaseReranker):
        self.settings = get_settings()
        self.retriever = retriever
        self.reranker = reranker
        self.sessions = SessionStore()
        self._llm: AsyncOpenAI | None = None

    # ---------------- LLM客户端(懒加载, 密钥缺失时给出明确提示) ----------------
    @property
    def llm(self) -> AsyncOpenAI:
        if self._llm is None:
            if not self.settings.llm_api_key:
                raise RuntimeError("LLM_API_KEY未配置, 请复制.env.example为.env并填入API Key")
            self._llm = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._llm

    # ==================== 流式问答主流程 ====================
    async def chat_stream(self, session_id: str, query: str) -> AsyncIterator[dict]:
        """流式问答: 依次产出事件(dict), 由API层转为SSE推给前端"""
        start = time.perf_counter()
        history = self.sessions.get_history(session_id)

        # ---- 阶段1: 意图识别 + 查询改写 ----
        yield {"type": "stage", "stage": "analyze", "message": "正在理解问题..."}
        analysis = await self._analyze_query(query, history)
        rewritten = analysis.get("query") or query
        need_retrieval = analysis.get("need_retrieval", True)
        yield {"type": "rewrite", "data": rewritten, "need_retrieval": need_retrieval}
        logger.info("查询分析: need_retrieval=%s rewritten=%r", need_retrieval, rewritten)

        # ---- 阶段2: 检索(仅知识类问题) ----
        chunks = []
        if need_retrieval:
            yield {"type": "stage", "stage": "retrieve", "message": "正在检索知识库..."}
            chunks = self.retriever.retrieve(rewritten)
            chunks = self.reranker.rerank(rewritten, chunks, self.settings.top_k_final)
            yield {
                "type": "stage", "stage": "retrieve_done",
                "message": f"检索到{len(chunks)}个相关片段",
                "data": {"hits": len(chunks), "top_score": round(chunks[0].score, 3) if chunks else 0},
            }
            # 先推送引用来源, 前端可与生成过程并行展示
            yield {"type": "sources", "data": [
                SourceRef(
                    index=i + 1, source=c.source,
                    snippet=c.text[:150] + ("..." if len(c.text) > 150 else ""),
                    score=round(c.score, 3),
                ).model_dump()
                for i, c in enumerate(chunks)
            ]}

        # ---- 阶段3: 流式生成 ----
        yield {"type": "stage", "stage": "generate", "message": "正在生成回答..."}
        answer_parts: list[str] = []
        prompt = (
            GENERATION_PROMPT.format(context=build_context(chunks), question=rewritten)
            if need_retrieval
            else CHITCHAT_PROMPT.format(query=query)
        )
        messages = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in history
        ]
        messages.append({"role": "user", "content": prompt})
        try:
            stream = await self.llm.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    answer_parts.append(delta)
                    yield {"type": "delta", "data": delta}
        except Exception as exc:
            logger.exception("LLM生成失败")
            yield {"type": "error", "data": f"模型调用失败: {exc}"}
            return

        answer = "".join(answer_parts)
        # ---- 写入会话记忆 ----
        self.sessions.append(session_id, "user", query)
        self.sessions.append(session_id, "assistant", answer)
        latency = int((time.perf_counter() - start) * 1000)
        logger.info("问答完成: %d字, %dms", len(answer), latency)
        yield {"type": "done", "data": {"latency_ms": latency, "used_retrieval": need_retrieval}}

    async def chat(self, session_id: str, query: str) -> ChatResponse:
        """非流式问答: 消费事件流聚合成完整响应(供同步接口/评估脚本使用)"""
        answer, sources, rewritten, used_retrieval, latency, error = [], [], "", False, 0, ""
        async for event in self.chat_stream(session_id, query):
            etype = event["type"]
            if etype == "delta":
                answer.append(event["data"])
            elif etype == "sources":
                sources = [SourceRef(**s) for s in event["data"]]
            elif etype == "rewrite":
                rewritten, used_retrieval = event["data"], event["need_retrieval"]
            elif etype == "done":
                latency = event["data"]["latency_ms"]
            elif etype == "error":
                error = event["data"]
        if error:
            raise RuntimeError(error)
        return ChatResponse(
            answer="".join(answer), sources=sources, rewritten_query=rewritten,
            used_retrieval=used_retrieval, latency_ms=latency,
        )

    # ==================== 查询分析 ====================
    async def _analyze_query(self, query: str, history) -> dict:
        """一次LLM调用完成意图识别+查询改写, 返回{need_retrieval, query}"""
        history_text = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in history[-6:]
        ) or "(无)"
        try:
            resp = await self.llm.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": QUERY_ANALYSIS_PROMPT.format(
                    history=history_text, query=query)}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception:
            # json模式不支持/网络异常时降级: 直接用原问题检索, 保证服务可用
            logger.warning("查询分析失败, 降级为直接检索原问题", exc_info=True)
            return {"need_retrieval": True, "query": query}
