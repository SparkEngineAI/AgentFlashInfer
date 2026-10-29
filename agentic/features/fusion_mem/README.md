# Fusion Mem

参考项目：[mem0](https://github.com/mem0ai/mem0)。

Fusion Mem 是一个基于 FastAPI 和 LangChain 构建的智能记忆服务系统。它提供了一个高性能的服务端用于处理数据存储与检索，并配备了易用的命令行工具（CLI）和 Python 客户端 SDK，支持与 OpenAI、Neo4j、ChromaDB 等多种后端集成。

## 📋 功能特性
服务端 (Server): 基于 FastAPI，支持高并发处理。
客户端 (Client): 封装好的 Python 模块，方便在其他代码中集成。
命令行工具 (CLI): 统一的 fusion_mem 命令，支持启动服务、快速存取数据。
AI 集成: 内置 OpenAI、LangChain 支持，适配向量数据库与图数据库。
## 🛠 环境依赖
Python: >= 3.8
基础依赖: FastAPI, Uvicorn, Pydantic
AI & Data: OpenAI, LangChain, ChromaDB, Psycopg (PostgreSQL), Neo4j
## 🚀 安装部署
1. 克隆/下载代码
将代码下载到本地目录 fusion_mem_project。

2. 创建虚拟环境（推荐）
为了避免依赖冲突，建议创建一个 Python 虚拟环境：


3. 安装项目
使用 pip 以编辑模式安装，这样修改代码后无需重新安装即可生效：

pip install -e .

此命令会自动安装 setup.py 中定义的所有依赖（包括 FastAPI, OpenAI, LangChain 等）。

4. 配置文件 (.env)

cp .env.example .env

配置必要的环境变量（根据你的实际依赖情况填写）：


### .env 示例

FUSION_MEM_HOST=127.0.0.1
FUSION_MEM_PORT=15002

OPENAI_API_KEY="none"
OPENAI_API_URL="http://127.0.0.1:8080/v1"
OPENAI_API_MODEL="/path/to/Qwen2.5-14B-Instruct/"

EMBEDDING_API_KEY="none"
EMBEDDING_API_URL="http://127.0.0.1:8001/v1"
EMBEDDING_API_MODEL="qwen3-embedding-0.6b"
EMBEDDING_MODEL_DIM=1024
DATA_PATH="/path/to/agentic/workspace"

### 💻 使用指南
安装完成后，系统会向终端注册一个全局命令 fusion_mem。

1. 启动服务端

在终端执行以下命令启动 API Server：

fusion_mem server

默认监听地址: http://127.0.0.1:15002


2. 命令行 (CLI) 操作

在 Server 启动的状态下，你可以打开一个新的终端窗口进行数据操作。

保存数据 (Save):

fusion_mem save "这里是需要记忆的一段文本内容"

搜索数据 (Search):

fusion_mem search "查询关键词"

3. Python SDK 调用

如果你需要在其他 Python 项目中使用 Fusion Mem 的功能，可以直接导入 client 模块：

from fusion_mem.client import MemoryClient
