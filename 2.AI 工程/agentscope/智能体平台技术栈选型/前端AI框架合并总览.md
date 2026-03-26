# 前端 AI 对话框架合并总览（整合版）

本文将以下 3 份文档合并为一份统一清单，并保证项目不遗漏：

- `docs/智能体平台技术栈选型/开源 ai 框架.md`
- `docs/智能体平台技术栈选型/前端AI框架汇总.md`
- `docs/智能体平台技术栈选型/知乎推荐.md`

## 合并原则

- 完整保留三份文档中出现过的所有项目
- 同一项目使用统一名称，并在“别名”列保留原文称呼
- 不同来源信息并存，便于后续二次筛选

## 全量项目清单（不遗漏）

| 统一名称 | 星数 | 别名/原文写法 | GitHub 地址 | 技术栈 | FastAPI 对接方式（摘要） | 来源 |
|---|---:|---|---|---|---|---|
| Open WebUI | 90,900+ | Open WebUI | https://github.com/Open-Webui/Open-Webui | Web（多技术栈） | 对接 OpenAI 兼容接口（如 `/v1/chat/completions`） | 前端AI框架汇总 |
| NextChat | 86,000+ | ChatGPT-Next-Web / ChatGPT Next Web | https://github.com/Yidadaa/ChatGPT-Next-Web | Next.js + React + TypeScript | 配置 API Base URL 指向 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| LobeChat | 67,000+ | Lobe Chat | https://github.com/lobehub/lobe-chat | Next.js | 修改 API 调用层或端点地址对接 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| LibreChat | 33,900+ | LibreChat | https://github.com/LibreChat-AI/LibreChat | React + Node.js | 配置自定义模型/API 端点对接 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| Chatbot UI | 20,000+ | Chatbot UI (Mckay Wrigley) | https://github.com/mckaywrigley/chatbot-ui | React + TypeScript | 修改 API 地址指向 FastAPI，支持流式响应 | 前端AI框架汇总、开源 ai 框架 |
| Onyx | 17,000+ | Danswer（旧名） | https://github.com/onyx-dot-app/onyx | 全栈（企业检索+聊天） | 文档中标注为原生 FastAPI 后端体系 | 前端AI框架汇总 |
| ChatGPT Web | 15,000+ | chatgpt-web | https://github.com/chatgpt-web-dev/chatgpt-web | Vue3 + Express | 实现/对接 OpenAI 兼容接口 | 前端AI框架汇总 |
| Chainlit | 11,400+ | Chainlit | https://github.com/Chainlit/chainlit | Python | 可与 FastAPI 协作，适合 Python 场景 | 前端AI框架汇总 |
| Assistant UI | 7,500+ | Assistant UI | https://github.com/assistant-ui/assistant-ui | React + TypeScript | 实现 `/api/chat` 或自定义后端适配 | 前端AI框架汇总 |
| Open Chat Ai | 5,000+ | Open-Chat-Ai | https://github.com/suneelkumarr/Open-Chat-Ai | React | 对接 OpenAI 兼容接口 | 前端AI框架汇总 |
| AI Chat Assistant | 3,000+ | ai-chat | https://github.com/flyryan/ai-chat | 前后端分离（WebSocket/HTTP） | 天然适配 FastAPI（HTTP + WebSocket） | 前端AI框架汇总 |
| CopilotKit | 未标注 | CopilotKit | https://github.com/copilotkit/copilotkit | React + TypeScript | 通过 `runtimeUrl` 或自定义运行时接入 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| Ant Design X | 未标注 | Ant Design X | https://github.com/ant-design/ant-design | React（Ant Design 生态） | 通过回调函数调用 FastAPI 接口 | 前端AI框架汇总、开源 ai 框架 |
| ChatUI（字节） | 未标注 | ChatUI（字节跳动） | https://github.com/chatui/chatui | React / Vue | 组件层对接，需自行实现 API 逻辑 | 前端AI框架汇总、开源 ai 框架 |
| OpenChat UI | 未标注 | OpenChat UI | https://github.com/openchat-ui/openchat-ui | React + Node.js（常见） | 配置文件设置 API 地址对接 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| NLUX | 未标注 | NLUX | https://github.com/nluxai/nlux | React / Vanilla JS | 通过适配器配置 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| Flowise UI | 未标注 | Flowise / Flowise UI | https://github.com/FlowiseAI/Flowise | Node.js + 可视化 UI | 使用 Custom API 组件接入 FastAPI | 前端AI框架汇总、开源 ai 框架 |
| ChatUI（阿里） | 未标注 | ChatUI（阿里开源） | https://github.com/alibaba/chatui | React | 组件层对接，需自行实现 API 逻辑 | 前端AI框架汇总、开源 ai 框架 |
| LLMChat | 未标注 | LLMChat | （原文未给出） | Flutter (Dart) | 通过 REST/WebSocket 对接 FastAPI | 知乎推荐 |
| ai-chatkit | 未标注 | ai-chatkit（全栈项目前端部分） | （原文未给出） | Next.js + React + TypeScript + Ant Design | 通过 SSE/WebSocket 调用 FastAPI | 知乎推荐 |
| Streamlit UI | 未标注 | Streamlit（agent-service-toolkit UI层） | （项目级别，原文未单列仓库） | Python + Streamlit | 在 Streamlit 中通过 `requests/httpx` 调用 FastAPI | 知乎推荐 |
| FastChat Web UI | 未标注 | FastChat 提供的 Web UI | （随 FastChat 项目） | Gradio (Python) | FastAPI 侧提供 OpenAI 兼容接口后可直接接入 | 知乎推荐 |

## 去重与别名说明

- `NextChat` 与 `ChatGPT-Next-Web` 视为同一项目（同仓库）。
- `LobeChat` 与 `Lobe Chat` 视为同一项目（同仓库）。
- `Onyx` 与 `Danswer` 为同一项目的新旧名称。
- `ChatUI（字节）` 与 `ChatUI（阿里）`为两个不同仓库，分别保留。

## 快速选型建议（按场景）

- 快速上线：`NextChat`、`Chatbot UI`
- 企业级与权限体系：`LibreChat`、`OpenChat UI`、`Onyx`
- Python/低前端门槛：`Chainlit`、`Streamlit UI`
- 轻量嵌入：`NLUX`、`Assistant UI`
- 可视化流程编排：`Flowise UI`
- 跨平台端（Web/移动/桌面）：`LLMChat`

## FastAPI 最小兼容要求（建议统一）

- 提供流式响应（SSE 或 WebSocket）
- 提供 OpenAI 兼容接口（建议 `/v1/chat/completions`）
- 配置 CORS
- 统一鉴权与模型路由策略（多前端共用时尤其重要）
