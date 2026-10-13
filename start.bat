@echo off
chcp 65001 >nul
REM ====== 企业智能RAG问答系统 一键启动脚本 ======
cd /d "%~dp0"

REM 1. 首次运行自动创建配置文件
if not exist .env (
    copy .env.example .env >nul
    echo [提示] 已生成 .env, 请编辑填入 LLM_API_KEY 后重新运行本脚本
    notepad .env
)

REM 2. 虚拟环境: 不存在则创建并安装依赖(首次约10分钟)
if not exist .venv\Scripts\python.exe (
    echo [安装] 创建虚拟环境并安装依赖, 请耐心等待...
    python -m venv .venv
    .venv\Scripts\pip.exe install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

REM 3. 启动服务(使用项目内虚拟环境, 不污染系统Python)
echo [启动] 服务地址: http://localhost:8000  (接口文档: http://localhost:8000/docs)
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
