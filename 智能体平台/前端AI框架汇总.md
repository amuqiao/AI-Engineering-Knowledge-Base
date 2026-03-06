# GitHub 开源前端 AI 对话框架汇总

以下是从两个文档中整理的前端AI对话框架，按 GitHub 星数降序排列，包含框架名称、GitHub 地址、星数、核心功能和适配 FastAPI 情况。

## 📊 前端 AI 框架汇总表

| 框架名称 | GitHub 地址 | 星数 | 核心功能 | 适配 FastAPI |
|---------|------------|------|----------|-------------|
| **Open WebUI** | https://github.com/Open-Webui/Open-Webui | 90,900+ | 零代码搭建AI界面，自动识别本地Ollama模型，支持OpenAI兼容API，企业级安全，内置RAG | 支持，实现`/v1/chat/completions`和`/v1/models`端点 |
| **NextChat (原ChatGPT Next Web)** | https://github.com/Yidadaa/ChatGPT-Next-Web | 86,000+ | 一键部署，支持多模型，完整Markdown渲染，上下文记忆，暗色/亮色主题 | 支持，设置`OPENAI_API_BASE_URL`环境变量 |
| **LobeChat** | https://github.com/lobehub/lobe-chat | 67,000+ | 现代化设计，支持语音合成/识别，多模态交互，可扩展插件系统，多用户管理 | 支持，修改API端点为FastAPI地址 |
| **LibreChat** | https://github.com/LibreChat-AI/LibreChat | 33,900+ | 支持15+ AI提供商，兼容ChatGPT插件规范，多用户管理，企业级认证 | 支持，添加自定义API端点 |
| **Onyx (前Danswer)** | https://github.com/onyx-dot-app/onyx | 17,000+ | 企业级AI搜索与聊天平台，支持40+知识源连接，内置RAG功能 | 原生使用FastAPI作为后端 |
| **ChatGPT Web** | https://github.com/chatgpt-web-dev/chatgpt-web | 15,000+ | 基于Express和Vue3的ChatGPT页面复刻，双模型支持，多会话管理 | 支持，实现OpenAI兼容接口 |
| **Chatbot UI (Mckay Wrigley)** | https://github.com/mckaywrigley/chatbot-ui | 20,000+ | 1:1复刻ChatGPT原生界面，支持流式响应，自定义API端点，本地存储对话记录 | 支持，修改API调用地址指向FastAPI |
| **Chainlit** | https://github.com/Chainlit/chainlit | 11,400+ | Python框架，无需前端知识，支持多模态交互，与LangChain等无缝集成 | 与FastAPI天然契合，可直接集成 |
| **Assistant UI** | https://github.com/assistant-ui/assistant-ui | 7,500+ | TypeScript/React组件库，自动处理流式响应，完全可组合的原语 | 支持，实现`/api/chat`接口 |
| **Open Chat Ai** | https://github.com/suneelkumarr/Open-Chat-Ai | 5,000+ | 下一代React聊天机器人，支持任意LLM提供商，流式响应，多种人格风格 | 支持，实现`/v1/chat/completions`接口 |
| **AI Chat Assistant** | https://github.com/flyryan/ai-chat | 3,000+ | 完整前后端分离架构，支持WebSocket和HTTP fallback，实时流式响应 | 与FastAPI天然契合，示例完善 |
| **CopilotKit** | https://github.com/copilotkit/copilotkit | - | 提供完整的聊天窗口组件，支持流式响应和上下文管理，内置提示组件 | 支持，通过`runtimeUrl`配置FastAPI地址 |
| **Ant Design X** | https://github.com/ant-design/ant-design | - | ChatGPT风格界面，多轮对话支持，消息编辑功能，暗黑模式 | 支持，通过`onRequest`回调调用FastAPI |
| **ChatUI (字节跳动)** | https://github.com/chatui/chatui | - | 高度可定制的聊天窗口组件，支持流式消息，Vue3/React双框架兼容 | 支持，需自行实现API对接逻辑 |
| **OpenChat UI** | https://github.com/openchat-ui/openchat-ui | - | 多租户/多用户权限管理，支持主流LLM，内置RAG能力 | 支持，通过配置文件设置API地址 |
| **NLUX** | https://github.com/nluxai/nlux | - | 支持React和Vanilla JS，适配多种LLM，流式响应支持，无外部依赖 | 支持，通过适配器配置FastAPI |
| **Flowise UI** | https://github.com/FlowiseAI/Flowise | - | 拖拽式可视化编辑器，内置数百个LLM/向量库/工具组件，一键导出API | 支持，添加「Custom API」组件配置FastAPI |
| **ChatUI (阿里开源)** | https://github.com/alibaba/chatui | - | 高度可定制的聊天窗口组件，支持流式消息，与阿里生态系统无缝集成 | 支持，需自行实现API对接逻辑 |

## 🌟 框架分类与推荐

### 按功能分类

| 分类 | 推荐框架 | 特点 |
|------|----------|------|
| **快速部署** | NextChat, Chatbot UI | 一键部署，轻量高效，界面美观 |
| **企业级应用** | LibreChat, OpenChat UI | 多模型支持，权限控制，RAG能力 |
| **本地模型** | Open WebUI, LobeChat | 对本地模型（Ollama）支持最佳 |
| **Python开发者** | Chainlit | 无需前端知识，与FastAPI天然契合 |
| **轻量级集成** | NLUX | 无依赖，快速集成，适合小型项目 |
| **可视化搭建** | Flowise UI | 拖拽式搭建，无需代码即可对接FastAPI |
| **大厂组件库** | Ant Design X, ChatUI (字节跳动), ChatUI (阿里开源) | 组件质量高，维护有保障 |

### 按技术栈分类

| 技术栈 | 推荐框架 |
|--------|----------|
| React | CopilotKit, Chatbot UI, Ant Design X, ChatUI (字节跳动), ChatUI (阿里开源) |
| Vue | ChatUI (字节跳动) |
| Next.js | NextChat, LobeChat |
| Python | Chainlit |
| 原生JS | NLUX |

## 🔧 FastAPI 集成要点

无论选择哪个前端框架，FastAPI 后端需要实现以下核心功能：

1. **流式响应**：使用 `StreamingResponse` 支持 SSE（Server-Sent Events）
2. **OpenAI 兼容接口**：实现 `/v1/chat/completions` 端点
3. **CORS 配置**：允许前端跨域请求
4. **请求验证**：使用 Pydantic 模型验证请求数据

### 基础集成代码示例

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "gpt-3.5-turbo"
    stream: bool = True

# 流式响应生成器
async def ai_stream_generator(user_content: str):
    response_chunks = ["你好！", "这是FastAPI后端的响应"]
    for chunk in response_chunks:
        await asyncio.sleep(0.5)
        yield f"data: {json.dumps({'content': chunk, 'role': 'assistant'})}\n\n"
    yield "data: [DONE]\n\n"

# 聊天接口
@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="最后一条消息必须是用户消息")
    
    user_content = request.messages[-1].content
    
    if request.stream:
        return StreamingResponse(
            ai_stream_generator(user_content),
            media_type="text/event-stream"
        )
    else:
        return {
            "messages": [*request.messages, {"role": "assistant", "content": f"你好！这是对「{user_content}」的回复。"}]
        }

# OpenAI 兼容接口
@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    # 处理 OpenAI 格式的请求
    messages = request.get("messages", [])
    user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    async def stream_generator():
        chunks = ["你好！", "这是OpenAI兼容接口的响应"]
        for chunk in chunks:
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'choices': [{"delta": {"content": chunk}}]})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
```

## 📝 总结

本汇总表涵盖了当前主流的前端AI对话框架，从功能丰富的企业级解决方案到轻量级的快速集成选项，满足不同场景的需求。所有框架都支持与FastAPI后端集成，可根据具体项目需求选择最合适的框架。

选择框架时，建议考虑以下因素：
1. 项目规模和复杂度
2. 技术栈偏好
3. 功能需求（如RAG、多模态、插件支持等）
4. 部署环境和资源限制
5. 开发团队的技术背景

通过合理选择前端框架并正确集成FastAPI后端，可以快速构建高质量的AI对话应用。
## 最佳适配框架
1. NextChat (原ChatGPT Next Web)
   
   - 支持通过环境变量 OPENAI_API_BASE_URL 直接指向FastAPI
   - 零配置对接，修改简单
   - 界面体验接近ChatGPT官方
2. Chatbot UI (Mckay Wrigley)
   
   - 支持自定义API端点，无需修改核心代码
   - 直接修改API调用地址指向FastAPI即可
   - 代码结构清晰，易二次开发
3. NLUX
   
   - 通过适配器配置FastAPI地址
   - 轻量级，集成简单
   - 支持原生JavaScript，学习成本低
4. Assistant UI
   
   - 实现 /api/chat 接口即可对接
   - 高度可定制，适合嵌入现有应用
   - 自动处理流式响应
5. Chainlit
   
   - 与FastAPI天然契合，可直接集成
   - 对Python开发者友好，无需前端知识
   - 快速原型设计能力强 

### 不同场景的考虑：
1. 企业级应用 ：
   
   - 建议后端实现完整的RAG和记忆功能
   - 前端主要负责展示和交互
   - 推荐框架：LibreChat、OpenChat UI
2. 个人或小型项目 ：
   
   - 可利用前端的本地存储功能
   - 减少后端复杂度
   - 推荐框架：NextChat、Chatbot UI
3. 嵌入式应用 ：
   
   - 前端轻量化，依赖后端处理核心逻辑
   - 推荐框架：NLUX、Assistant UI