# ===== 构建阶段 =====
FROM python:3.11-slim AS base

# 时区与编码
ENV PYTHONUNBUFFERED=1 TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖(利用Docker层缓存: 代码变动不重装依赖)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再拷贝代码
COPY app/ ./app/
COPY static/ ./static/
COPY scripts/ ./scripts/

# 数据目录(向量库/模型缓存/知识库文档, 运行时挂载卷持久化)
RUN mkdir -p data/raw_docs data/chroma data/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
