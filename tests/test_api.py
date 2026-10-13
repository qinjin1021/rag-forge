"""API冒烟测试: 不依赖LLM Key与模型下载, 验证服务能正常组装启动"""
from fastapi.testclient import TestClient

from app.main import create_app


def test_healthz():
    """健康检查接口可用"""
    with TestClient(create_app()) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "chunks" in body and "docs" in body


def test_documents_list():
    """文档列表接口返回结构正确"""
    with TestClient(create_app()) as client:
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_chat_validation():
    """空问题被参数校验拦截(422), 不会打到LLM"""
    with TestClient(create_app()) as client:
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422
