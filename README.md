# 企业智能RAG问答系统

基于 **DeepSeek + 混合检索 + Agent编排** 的企业知识库问答系统。
不是玩具Demo：分层解耦、配置集中、可评估、可测试、可Docker部署，按企业落地标准构建。

---

## 一、系统架构(数据流)

```
                        ┌─────────────────── 离线: 知识摄取管道 ───────────────────┐
 企业文档                │                                                          │
 (md/pdf/docx/xlsx) ──► │  加载loaders ─► 清洗cleaners ─► 分块splitters            │
                        │       (多格式解析)   (去噪声)     (递归语义分块)            │
                        └──────────────┬───────────────────────────────┬───────────┘
                                       ▼                               ▼
                              Embedding向量化                  Chroma向量库(持久化)
                              (本地/API可切换)                        │
------------------------------------------------------------------+------------------
                        ┌─────────────── 在线: Agent问答编排 ──────────┴───────────┐
 用户提问 ─────────────► │                                                          │
                        │  ①意图识别+查询改写(是否检索? 指代消解)                      │
                        │  ②混合检索 = 向量语义检索 + BM25关键词检索 ──► 加权融合      │
                        │  ③重排序(可选CrossEncoder精排)                              │
                        │  ④LLM生成: 强制引用来源[1][2], 资料不足拒答                  │
                        └──────────────┬───────────────────────────────────────────┘
                                       ▼
                        SSE流式返回 ──► Web聊天界面(过程可视化/来源引用)
```

**一次提问的完整旅程**: 你问"普通员工住宿标准多少" → Agent判断这是知识类问题 → 改写检索query →
向量检索找语义相近片段 + BM25精确命中"住宿""400元"等词 → 融合排序取Top4 →
DeepSeek基于片段生成答案并标注引用 → 前端打字机输出 + 显示引用来源卡片。

---

## 二、目录结构与模块职责

```
rag问答项目/
├── app/
│   ├── config.py              # 配置中心: 所有参数集中, .env覆盖, 启动即校验
│   ├── main.py                # FastAPI入口: 组件组装/路由/中间件/异常兜底/健康检查
│   ├── core/
│   │   ├── schemas.py         # Pydantic数据模型: 各层流转的数据结构统一定义
│   │   └── logging.py         # 统一日志 + trace_id链路追踪(一次问答全链路可追踪)
│   ├── ingestion/             # 【摄取管道】决定"知识进库的质量"
│   │   ├── loaders.py         #   多格式加载器: txt/md/pdf/docx/xlsx, 工厂模式分发
│   │   ├── cleaners.py        #   清洗: 全角半角/控制符/空行/页眉残留去噪
│   │   ├── splitters.py       #   递归分块: 段落→句子→字符逐级切, 带overlap防边界丢语义
│   │   └── pipeline.py        #   流水线编排: 失败隔离 + 幂等(重复入库=覆盖更新)
│   ├── embeddings/
│   │   └── provider.py        # 【向量化】抽象接口 + 本地/API双实现, 配置一键切换
│   ├── vectordb/
│   │   └── store.py           # 【向量库】Chroma仓储封装, 上层不碰底层API(可换Milvus)
│   ├── retrieval/
│   │   ├── retriever.py       # 【检索】向量+BM25混合检索, 加权融合(核心效果保障)
│   │   └── reranker.py        #   重排序: CrossEncoder精排(可选), Noop空实现兜底
│   ├── agent/
│   │   ├── rag_agent.py       # 【Agent大脑】意图→改写→检索→生成 事件流编排
│   │   ├── prompts.py         #   提示词集中管理(提示词=系统的"代码")
│   │   └── session.py         #   多轮会话记忆(演示内存版, 生产换Redis)
│   └── api/
│       ├── routes_chat.py     # 问答接口: SSE流式 + 同步JSON + 会话清理
│       ├── routes_docs.py     # 知识库管理: 上传入库/列表/删除/全量重建
│       └── deps.py            # 依赖注入: 组件单例入口
├── static/index.html          # 聊天前端: 流式渲染/Agent过程指示/引用来源/文档管理
├── scripts/
│   ├── ingest.py              # CLI入库工具: python scripts/ingest.py --rebuild
│   └── evaluate.py            # 评估脚本: 检索命中率 + 回答覆盖率 + 延迟
├── tests/                     # 单元测试: 清洗/分块/融合算法 + API冒烟
├── data/
│   ├── raw_docs/              # 知识库文档(含3篇示例: 员工手册/产品FAQ/报销制度)
│   ├── eval.jsonl             # 评测集(9条标注样本)
│   └── chroma/                # 向量库持久化(自动生成)
├── Dockerfile                 # 分层构建+健康检查+国内镜像源
├── docker-compose.yml         # 一键部署, env_file注入密钥, 卷持久化数据
├── start.bat                  # Windows一键启动(自动建配置/装依赖/起服务)
└── requirements.txt
```

---

## 三、快速开始

### 1. 安装与配置

```bash
cd rag问答项目
pip install -r requirements.txt        # 首次安装约2GB(含torch), 请耐心
copy .env.example .env                 # Windows
# 编辑 .env 填入 LLM_API_KEY (https://platform.deepseek.com 获取)
```

### 2. 知识入库

```bash
python scripts/ingest.py               # 入库 data/raw_docs 全部文档
# 首次运行会自动下载中文向量模型(约100MB, 之后离线可用)
```

### 3. 启动服务

```bash
start.bat                              # Windows一键启动
# 或: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 聊天界面: http://localhost:8000
- 接口文档: http://localhost:8000/docs

### 4. 验证效果(可选)

```bash
python scripts/evaluate.py --retrieval   # 只评检索命中率(不耗LLM额度)
python scripts/evaluate.py               # 全量评估(含生成质量)
pytest                                   # 跑单元测试
```

### 5. Docker部署

```bash
docker compose up -d --build
```

---

## 四、核心API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | 流式问答(SSE), 事件: stage/rewrite/sources/delta/done |
| POST | `/api/chat` | 同步问答, 一次性返回完整结果 |
| DELETE | `/api/chat/session/{id}` | 清空会话(开启新话题) |
| POST | `/api/documents/upload` | 上传文档并入库(多文件, 同名覆盖更新) |
| GET | `/api/documents` | 知识库文档列表及分块数 |
| DELETE | `/api/documents/{source}` | 删除指定文档(向量+文件) |
| POST | `/api/documents/ingest` | 全量重建知识库 |
| GET | `/healthz` | 健康检查(k8s/负载均衡探针) |

curl示例:

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
     -d '{"session_id":"s1","query":"年假有几天？"}'
```

---

## 五、工程化设计详解(面试讲解要点)

### 1. 为什么分层解耦?
`ingestion / embeddings / vectordb / retrieval / agent / api` 各层只依赖抽象:
换向量库只改 `vectordb/store.py`; 换Embedding只改配置; 换LLM只改 `.env`。
**接口稳定, 实现可替换**, 这是企业系统应对供应商变化的标准做法。

### 2. 为什么混合检索?
- 纯向量: 语义泛化好, 但"400元""P0级"这类精确token容易被淹没
- 纯BM25: 精确命中, 但不理解"住宿标准≈住宿限额"
- 融合策略: 两路各自归一化到[0,1], 按 0.7/0.3 加权求和排序
- 漏斗架构: 粗召回(top 8~16) → 精排(CrossEncoder取top 4) → 生成, 兼顾效果与成本

### 3. Agent体现在哪?
不是简单"检索+拼接", 而是**自主决策的过程编排**:
- **意图识别**: 闲聊不检索, 直接回答(省token省时延); 知识类问题才走RAG
- **查询改写**: "那报销呢?" → "差旅报销的流程是什么?"(指代消解, 检索质量大幅提升)
- **过程透明**: 每阶段推送事件, 前端实时显示"理解问题→检索→生成"
- **拒答机制**: 资料不足时明确说不知道 —— **宁可拒答不可编造**, 企业合规底线

### 4. 质量如何保障(评估体系)?
任何调参(分块大小/权重/模型)前后跑 `scripts/evaluate.py`:
- `hit@k` 检索命中率 → 定位问题在检索层还是生成层
- 关键词覆盖率 → 端到端答案质量
- 延迟分布 → 体验
**没有评估的RAG优化都是盲调**。

### 5. 可观测性?
- 每个请求自动分配 `trace_id`, 全链路日志串联, 响应头 `X-Trace-Id` 返回
- 排查用户反馈: 拿trace_id一查即定位到当次检索命中与LLM输入输出
- 全局异常兜底, 错误响应携带trace_id

### 6. 数据安全(企业关心)?
- Embedding默认本地模型: **文档内容不出内网**, 只有提问与片段送LLM
- 可进一步: DeepSeek不记录数据(以官方协议为准), 或替换为内网私有化LLM(改.env即可)

---

## 六、从演示到生产的演进路线

| 阶段 | 演示版(本项目) | 生产版改造点 |
|---|---|---|
| 会话记忆 | 进程内存 | Redis(多实例共享) |
| BM25索引 | 进程内存重建 | Elasticsearch集群 |
| 向量库 | Chroma嵌入式 | Milvus/Qdrant集群 |
| 权限 | 无 | 文档级/部门级ACL, 检索时过滤 |
| 模型 | API调用 | 本地vLLM部署, 数据完全不出域 |
| 评测 | 手动脚本 | CI自动回归 + badcase回流标注 |
| 监控 | 日志 | Prometheus指标 + Grafana面板 |

---

## 七、常见问题

- **首次启动慢/卡住**: 在下载Embedding模型(约100MB), 已默认走国内镜像, 之后离线可用
- **回答说"无法回答"**: 知识库为空或未命中, 先 `python scripts/ingest.py`, 或用 `evaluate.py --retrieval` 检查检索层
- **换了Embedding配置后报维度错误**: 向量维度变了, 执行 `python scripts/ingest.py --rebuild` 重建
- **Windows终端中文乱码**: 执行 `chcp 65001` 后再运行
