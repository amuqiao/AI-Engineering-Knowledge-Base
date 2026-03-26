# AgentScope Agent 类型与编排方式深度解析

> 基于 `examples/` 目录下的完整示例，系统梳理 AgentScope 支持的 Agent 类型、配置方法、编排模式，以及在复杂任务场景中的实战用法。

---

## 一、全局架构概览

AgentScope 的 Agent 体系由**核心类层次**、**能力模块**和**编排管道**三部分共同构成。

```mermaid
flowchart TB
    %% ── 配色定义 ─────────────────────────────────────────────────────
    classDef clientStyle  fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef gatewayStyle fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef svcStyle     fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef mqStyle      fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef dbStyle      fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef cacheStyle   fill:#ea580c,stroke:#7c2d12,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px
    classDef authStyle    fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff

    %% ── 用户 / 外部系统 ───────────────────────────────────────────────
    subgraph ENTRY["输入层"]
        direction LR
        USER["UserAgent<br>用户交互"]:::clientStyle
        EXT["外部系统<br>A2A / A2UI"]:::clientStyle
        AUDIO["实时音频<br>WebSocket"]:::clientStyle
    end
    class ENTRY layerStyle

    %% ── Agent 层 ────────────────────────────────────────────────────
    subgraph AGENTS["Agent 层"]
        direction LR
        REACT["ReActAgent<br>推理+行动"]:::gatewayStyle
        REALTIME["RealtimeAgent<br>实时语音"]:::authStyle
        A2A["A2AAgent<br>跨 Agent 通信"]:::svcStyle
        CUSTOM["自定义 Agent<br>AgentBase 扩展"]:::svcStyle
    end
    class AGENTS layerStyle

    %% ── 能力模块层 ──────────────────────────────────────────────────
    subgraph CAPABILITY["能力模块层"]
        direction LR
        TOOLKIT["Toolkit<br>工具调用"]:::mqStyle
        MEMORY["Memory<br>短期/长期记忆"]:::dbStyle
        RAG["RAG<br>知识检索增强"]:::cacheStyle
        PLAN["PlanNotebook<br>任务规划"]:::mqStyle
    end
    class CAPABILITY layerStyle

    %% ── 编排管道层 ──────────────────────────────────────────────────
    subgraph PIPELINE["编排管道层"]
        direction LR
        SEQ["sequential_pipeline<br>顺序管道"]:::svcStyle
        FAN["fanout_pipeline<br>扇出并发管道"]:::svcStyle
        HUB["MsgHub<br>广播消息中枢"]:::gatewayStyle
        CHAT["ChatRoom<br>实时聊天室"]:::authStyle
    end
    class PIPELINE layerStyle

    %% ── 模型 / 工具底座 ──────────────────────────────────────────────
    subgraph INFRA["基础设施层"]
        direction LR
        LLM[("LLM 模型<br>DashScope / OpenAI / Gemini")]:::dbStyle
        MCP[("MCP 服务<br>外部工具协议")]:::cacheStyle
        VDB[("向量数据库<br>Qdrant / Milvus / MongoDB")]:::dbStyle
    end
    class INFRA layerStyle

    %% ── 数据流 ──────────────────────────────────────────────────────
    USER     -->|"Msg 消息"| AGENTS
    EXT      -->|"A2A 协议"| AGENTS
    AUDIO    -->|"音频流"| REALTIME

    REACT    -->|"调用工具"| TOOLKIT
    REACT    -->|"读写记忆"| MEMORY
    REACT    -->|"检索知识"| RAG
    REACT    -->|"执行计划"| PLAN
    REALTIME -->|"实时推理"| LLM
    A2A      -->|"转发请求"| REACT

    TOOLKIT  -->|"MCP 协议"| MCP
    RAG      -->|"向量检索"| VDB
    REACT    -->|"LLM 推理"| LLM
    CUSTOM   -->|"LLM 推理"| LLM

    AGENTS   -->|"编排协调"| PIPELINE

    %% ── 注记 ────────────────────────────────────────────────────────
    NOTE["架构要点<br>① ReActAgent 是核心推理单元，支持工具/记忆/RAG/计划四大扩展<br>② Pipeline 层将多个 Agent 组合成工作流<br>③ MCP 协议统一接入外部工具（浏览器/搜索/地图等）"]:::noteStyle
    NOTE -.- CAPABILITY

    %% 边索引：0-15，共 16 条
    linkStyle 0,1,2 stroke:#374151,stroke-width:2px
    linkStyle 3,4,5,6 stroke:#d97706,stroke-width:1.5px
    linkStyle 7 stroke:#dc2626,stroke-width:2px
    linkStyle 8 stroke:#0891b2,stroke-width:1.5px
    linkStyle 9 stroke:#d97706,stroke-width:1.5px
    linkStyle 10 stroke:#059669,stroke-width:1.5px
    linkStyle 11,12 stroke:#059669,stroke-width:2px
    linkStyle 13 stroke:#1d4ed8,stroke-width:2px
    linkStyle 14 stroke:#1d4ed8,stroke-width:2px
```

---

## 二、Agent 类型详解

### 2.1 类层次结构

```mermaid
flowchart TB
    classDef baseStyle   fill:#1f2937,stroke:#111827,stroke-width:2.5px,color:#f9fafb
    classDef coreStyle   fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef extStyle    fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef customStyle fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    BASE["AgentBase<br>基础抽象类<br>reply() / observe() / print()"]:::baseStyle

    subgraph BUILTIN["内置 Agent 类型"]
        direction LR
        REACT["ReActAgent<br>推理-行动循环"]:::coreStyle
        USER["UserAgent<br>人机交互代理"]:::coreStyle
        A2A["A2AAgent<br>跨 Agent 通信"]:::coreStyle
        RT["RealtimeAgent<br>实时语音"]:::coreStyle
    end
    class BUILTIN layerStyle

    subgraph CUSTOM["自定义 Agent（examples 中的扩展）"]
        direction LR
        DR["DeepResearchAgent<br>深度研究"]:::customStyle
        BA["BrowserAgent<br>浏览器操控"]:::customStyle
        EX["ExampleAgent<br>用户自定义"]:::customStyle
    end
    class CUSTOM layerStyle

    BASE --> REACT
    BASE --> USER
    BASE --> A2A
    BASE --> RT
    BASE --> DR
    BASE --> BA
    BASE --> EX

    NOTE["自定义扩展只需继承 AgentBase<br>并实现 reply() 和 handle_interrupt() 方法"]:::noteStyle
    NOTE -.- CUSTOM

    %% 边索引：0-6，共 7 条
    linkStyle 0,1,2,3,4,5,6 stroke:#374151,stroke-width:2px
```

---

### 2.2 ReActAgent — 核心推理 Agent

`ReActAgent` 是 AgentScope 中最核心的 Agent，实现了 **ReAct（Reasoning + Acting）** 循环——在每个迭代步骤中，Agent 先通过 LLM 进行推理，再决定是否调用工具，直到任务完成。

#### 配置参数一览

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | Agent 的显示名称 |
| `sys_prompt` | `str` | 系统提示词，定义 Agent 的角色和行为约束 |
| `model` | `ChatModelBase` | 绑定的 LLM 模型实例 |
| `formatter` | `FormatterBase` | 消息格式化器，适配不同 LLM API |
| `toolkit` | `Toolkit` | 工具集，包含可调用的函数和 MCP 工具 |
| `memory` | `MemoryBase` | 短期记忆（默认 `InMemoryMemory`） |
| `plan_notebook` | `PlanNotebook` | 任务规划本，支持多步骤分解 |
| `max_iters` | `int` | ReAct 循环最大迭代次数，防止无限循环 |
| `enable_meta_tool` | `bool` | 是否启用元工具（`finish` 等内置工具） |

#### 最小配置示例

```python
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_shell_command, execute_python_code

toolkit = Toolkit()
toolkit.register_tool_function(execute_shell_command)
toolkit.register_tool_function(execute_python_code)

agent = ReActAgent(
    name="Friday",
    sys_prompt="You are a helpful assistant named Friday.",
    model=DashScopeChatModel(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-max",
        stream=True,
    ),
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
)
```

> 来源：`examples/agent/react_agent/main.py`

---

### 2.3 UserAgent — 人机交互代理

`UserAgent` 封装了人类用户的输入逻辑，通常作为对话循环的另一端存在，使 Agent 与用户交互保持统一的消息格式。

```python
from agentscope.agent import UserAgent

user = UserAgent(name="User")

# 对话循环
msg = None
while True:
    msg = await user(msg)           # 等待用户在终端输入
    if msg.get_text_content() == "exit":
        break
    msg = await agent(msg)
```

> 来源：`examples/agent/react_agent/main.py`

---

### 2.4 A2AAgent — Agent-to-Agent 通信代理

`A2AAgent` 基于 **Google A2A 协议**，允许 AgentScope Agent 与其他框架（LangChain、CrewAI 等）或远程 Agent 进行标准化通信。其核心是 `AgentCard`，描述 Agent 的能力和接入方式。

#### AgentCard 配置

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

agent_card = AgentCard(
    name="Friday",
    description="A simple ReAct agent that handles input queries",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=AgentCapabilities(
        push_notifications=False,
        state_transition_history=True,
        streaming=True,
    ),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            name="execute_python_code",
            id="execute_python_code",
            description="Execute Python code snippets.",
            tags=["code_execution"],
        ),
    ],
)
```

#### A2A 调用方式

```python
from agentscope.agent import A2AAgent

agent = A2AAgent(agent_card=agent_card)
msg = await agent(user_msg)
```

> 来源：`examples/agent/a2a_agent/`

---

### 2.5 RealtimeAgent — 实时语音 Agent

`RealtimeAgent` 专为**实时音频流**设计，配合 `ChatRoom` 管道实现多 Agent 实时语音对话场景。支持 DashScope、Gemini、OpenAI 三家实时语音模型。

```python
from agentscope.agent import RealtimeAgent
from agentscope.realtime import DashScopeRealtimeModel

agent1 = RealtimeAgent(
    name="Alice",
    sys_prompt="You are a helpful assistant.",
    model=DashScopeRealtimeModel(
        model_name="qwen3-omni-flash-realtime",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        voice="Dylan",
        enable_input_audio_transcription=False,
    ),
)
```

> 来源：`examples/workflows/multiagent_realtime/run_server.py`

---

### 2.6 自定义 Agent — AgentBase 扩展

通过继承 `AgentBase` 并实现 `reply()` 方法，可创建完全自定义的 Agent 行为。

```python
from agentscope.agent import AgentBase
from agentscope.message import Msg

class MyCustomAgent(AgentBase):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def reply(self, *args, **kwargs) -> Msg:
        # 自定义逻辑：处理消息、调用外部服务等
        return Msg(self.name, "custom response", "assistant")

    async def handle_interrupt(self, *args, **kwargs) -> Msg:
        pass  # 处理中断信号

    async def observe(self, *args, **kwargs) -> None:
        pass  # 观察外部消息（不回复）
```

> 来源：`examples/workflows/multiagent_concurrent/main.py`

**examples 中的扩展示例：**

| 自定义 Agent | 所在路径 | 扩展内容 |
|---|---|---|
| `DeepResearchAgent` | `examples/agent/deep_research_agent/` | 集成 Tavily 搜索 MCP，支持深度研究流程 |
| `BrowserAgent` | `examples/agent/browser_agent/` | 集成 Playwright MCP，驱动浏览器操作 |
| `ExampleAgent` | `examples/workflows/multiagent_concurrent/` | 模拟并发任务，记录执行时间 |

---

## 三、能力模块配置详解

### 3.1 模型（Model）配置

AgentScope 通过统一的 `ChatModelBase` 接口适配多家 LLM。

| 模型类 | 适用平台 | 典型 model_name |
|--------|----------|-----------------|
| `DashScopeChatModel` | 阿里云百炼 | `qwen-max`, `qwen3-max` |
| `OpenAIChatModel` | OpenAI / 兼容接口 | `gpt-4o`, `qwen3-omni-flash` |
| `DashScopeRealtimeModel` | 阿里云实时语音 | `qwen3-omni-flash-realtime` |
| `GeminiRealtimeModel` | Google Gemini | `gemini-2.5-flash-native-audio-preview` |
| `OpenAIRealtimeModel` | OpenAI 实时 | `gpt-4o-realtime-preview` |

**高级模型参数示例：**

```python
DashScopeChatModel(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    model_name="qwen3-max",
    enable_thinking=False,   # 关闭思维链（加速响应）
    stream=True,             # 流式输出
)
```

---

### 3.2 消息格式化器（Formatter）配置

Formatter 负责将 AgentScope 内部的 `Msg` 对象转换为各 LLM API 所需的请求格式。

| 格式化器 | 适用场景 |
|---------|----------|
| `DashScopeChatFormatter` | 单 Agent 对话（DashScope API） |
| `DashScopeMultiAgentFormatter` | 多 Agent 对话，提示词中出现多个实体时使用 |
| `OpenAIChatFormatter` | OpenAI 兼容 API |

---

### 3.3 工具包（Toolkit）配置

`Toolkit` 是 AgentScope 的工具管理中心，支持三种工具接入方式：

#### 方式一：注册普通 Python 函数

```python
from agentscope.tool import Toolkit, execute_shell_command, execute_python_code, view_text_file

toolkit = Toolkit()
toolkit.register_tool_function(execute_shell_command)
toolkit.register_tool_function(execute_python_code)
toolkit.register_tool_function(view_text_file)
```

#### 方式二：注册 MCP 客户端（外部工具服务）

```python
from agentscope.mcp import HttpStatefulClient, HttpStatelessClient, StdIOStatefulClient

# HTTP SSE 模式（有状态）
add_client = HttpStatefulClient(name="add_mcp", transport="sse", url="http://127.0.0.1:8001/sse")
await add_client.connect()
await toolkit.register_mcp_client(add_client)

# HTTP Streamable 模式（无状态）
multiply_client = HttpStatelessClient(
    name="multiply_mcp", transport="streamable_http", url="http://127.0.0.1:8002/mcp"
)
await toolkit.register_mcp_client(multiply_client)

# stdio 模式（本地进程）
browser_client = StdIOStatefulClient(
    name="playwright-mcp", command="npx", args=["@playwright/mcp@latest"]
)
await browser_client.connect()
await toolkit.register_mcp_client(browser_client)
```

#### 方式三：注册 Agent Skill（知识型技能文件）

```python
toolkit.register_agent_skill("./skill/analyzing-agentscope-library")
```

#### 工具分组管理

```python
toolkit.create_tool_group(group_name="browser_tools", description="Web browsing related tools.")
await toolkit.register_mcp_client(browser_client, group_name="browser_tools")
```

---

### 3.4 记忆（Memory）配置

AgentScope 支持短期记忆和长期记忆两种模式。

| 记忆类型 | 类名 | 存储位置 | 适用场景 |
|---------|------|---------|---------|
| 短期记忆 | `InMemoryMemory` | 进程内存 | 单次会话，轻量对话 |
| SQLite 持久化 | `session_with_sqlite` | 本地文件 | 多次会话，本地存储 |
| 长期记忆（Reme） | `RemeMemory` | 外部存储 | 个人画像、任务历史 |
| 长期记忆（Mem0） | `Mem0Memory` | Mem0 服务 | 云端持久化记忆 |

```python
from agentscope.memory import InMemoryMemory

agent = ReActAgent(
    ...,
    memory=InMemoryMemory(),
)
```

---

### 3.5 知识增强（RAG）配置

通过 `SimpleKnowledge` + 向量数据库实现检索增强生成：

```python
from agentscope.rag import SimpleKnowledge, QdrantStore, TextReader
from agentscope.embedding import DashScopeTextEmbedding

knowledge = SimpleKnowledge(
    embedding_store=QdrantStore(
        location=":memory:",
        collection_name="my_collection",
        dimensions=1024,
    ),
    embedding_model=DashScopeTextEmbedding(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="text-embedding-v4",
    ),
)

# 将 RAG 工具注册到 Toolkit 供 Agent 调用
toolkit.register_tool_function(
    knowledge.retrieve_knowledge,
    func_description="Retrieve relevant documents from the knowledge base.",
)
```

**支持的向量数据库：**

| 向量库 | 类名 | 部署方式 |
|--------|------|---------|
| Qdrant | `QdrantStore` | 内存 / 本地 / 云端 |
| Milvus Lite | `MilvusLiteStore` | 本地文件 |
| MongoDB Atlas | `MongoDBVectorStore` | 云端托管 |
| 阿里云 MySQL 向量 | `AlibabaCloudMySQLVectorStore` | 云端 |
| OceanBase | `OceanBaseVectorStore` | 云端 |

---

### 3.6 任务规划（PlanNotebook）配置

`PlanNotebook` 赋予 Agent 任务分解和进度追踪能力，支持手动指定和 Agent 自动生成两种模式。

#### 模式一：手动创建计划

```python
from agentscope.plan import PlanNotebook, SubTask

plan_notebook = PlanNotebook()
await plan_notebook.create_plan(
    name="Comprehensive Report on AgentScope",
    description="Study the code and write a report.",
    expected_outcome="A markdown report.",
    subtasks=[
        SubTask(name="Clone the repository", description="...", expected_outcome="..."),
        SubTask(name="View the documentation", description="...", expected_outcome="..."),
        SubTask(name="Summarize the findings", description="...", expected_outcome="..."),
    ],
)

agent = ReActAgent(..., plan_notebook=plan_notebook)
```

#### 模式二：Agent 自主规划（Meta Planner）

```python
agent = ReActAgent(
    ...,
    plan_notebook=PlanNotebook(),
    enable_meta_tool=True,   # 启用元工具，Agent 可自主创建和更新计划
)
```

> 来源：`examples/functionality/plan/`

---

## 四、编排管道详解

### 4.1 四种编排模式对比

| 编排模式 | API | 消息流向 | 适用场景 |
|---------|-----|---------|---------|
| 顺序管道 | `sequential_pipeline` | A→B→C 链式传递 | 流水线处理、多步骤串行任务 |
| 扇出并发管道 | `fanout_pipeline` | A → [B, C, D] 并行 | 多视角分析、并行投票、批量处理 |
| 消息广播中枢 | `MsgHub` | 任意成员 → 全体广播 | 多 Agent 辩论、圆桌讨论 |
| 实时聊天室 | `ChatRoom` | 音频双向流 | 实时语音多 Agent 对话 |

---

### 4.2 顺序管道（Sequential Pipeline）

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    INPUT["用户输入<br>Msg"]:::userStyle
    A1["Agent A<br>处理步骤 1"]:::agentStyle
    A2["Agent B<br>处理步骤 2"]:::agentStyle
    A3["Agent C<br>处理步骤 3"]:::agentStyle
    OUTPUT["最终输出<br>Msg"]:::userStyle

    INPUT -->|"msg_0"| A1
    A1    -->|"msg_1"| A2
    A2    -->|"msg_2"| A3
    A3    -->|"msg_3"| OUTPUT

    NOTE["上一个 Agent 的输出作为下一个 Agent 的输入<br>sequential_pipeline([A, B, C], msg_input)"]:::noteStyle
    NOTE -.- A2

    %% 边索引：0-3，共 4 条
    linkStyle 0,1,2,3 stroke:#1e40af,stroke-width:2px
```

**使用方式：**

```python
from agentscope.pipeline import sequential_pipeline

result = await sequential_pipeline([alice, bob, charlie], initial_msg)
```

---

### 4.3 扇出并发管道（Fanout Pipeline）

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef aggStyle     fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    INPUT["相同输入<br>Msg（广播）"]:::userStyle

    subgraph PARALLEL["并行执行层（asyncio.gather）"]
        direction TB
        A1["Agent Alice<br>视角 1"]:::agentStyle
        A2["Agent Bob<br>视角 2"]:::agentStyle
        A3["Agent Chalice<br>视角 3"]:::agentStyle
    end
    class PARALLEL layerStyle

    COLLECT["结果收集<br>list[Msg]"]:::aggStyle

    INPUT -->|"deep copy"| A1
    INPUT -->|"deep copy"| A2
    INPUT -->|"deep copy"| A3
    A1 --> COLLECT
    A2 --> COLLECT
    A3 --> COLLECT

    NOTE["fanout_pipeline(agents, enable_gather=True)<br>所有 Agent 同时启动，等待全部完成后汇总"]:::noteStyle
    NOTE -.- COLLECT

    %% 边索引：0-5，共 6 条
    linkStyle 0,1,2 stroke:#1e40af,stroke-width:2px
    linkStyle 3,4,5 stroke:#0891b2,stroke-width:2px
```

**使用方式：**

```python
from agentscope.pipeline import fanout_pipeline

results = await fanout_pipeline(
    agents=[alice, bob, chalice],
    enable_gather=True,   # True=并发，False=顺序
)
```

> 来源：`examples/workflows/multiagent_concurrent/main.py`

---

### 4.4 消息广播中枢（MsgHub）

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef hubStyle     fill:#d97706,stroke:#92400e,stroke-width:2.5px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    ANNOUNCE["系统公告<br>announcement Msg"]:::userStyle

    subgraph HUB_GROUP["MsgHub 广播域"]
        direction TB
        HUB["MsgHub<br>消息路由中枢"]:::hubStyle
        ALICE["Alice<br>参与者 A"]:::agentStyle
        BOB["Bob<br>参与者 B"]:::agentStyle
        CHARLIE["Charlie<br>参与者 C"]:::agentStyle
    end
    class HUB_GROUP layerStyle

    ANNOUNCE -->|"进入时广播"| HUB
    HUB      -->|"公告分发"| ALICE
    HUB      -->|"公告分发"| BOB
    HUB      -->|"公告分发"| CHARLIE

    ALICE  -->|"回复自动广播"| HUB
    HUB    -->|"转发给 Bob"| BOB
    HUB    -->|"转发给 Charlie"| CHARLIE

    NOTE["任何参与者的回复消息<br>自动广播给其他所有参与者<br>hub.delete(bob) 可动态移除"]:::noteStyle
    NOTE -.- HUB

    %% 边索引：0-7，共 8 条
    linkStyle 0,1,2,3 stroke:#d97706,stroke-width:2px
    linkStyle 4 stroke:#0891b2,stroke-width:2px
    linkStyle 5,6 stroke:#d97706,stroke-width:1.5px,stroke-dasharray:4 3
```

**使用方式：**

```python
from agentscope.pipeline import MsgHub, sequential_pipeline

async with MsgHub(
    participants=[alice, bob, charlie],
    announcement=Msg("system", "Please introduce yourself.", "system"),
) as hub:
    await sequential_pipeline([alice, bob, charlie])

    # 动态移除参与者
    hub.delete(bob)
    await hub.broadcast(Msg("bob", "See you later!", "assistant"))
```

> 来源：`examples/workflows/multiagent_conversation/main.py`

---

### 4.5 实时聊天室（ChatRoom）

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef roomStyle    fill:#059669,stroke:#064e3b,stroke-width:2.5px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    FRONTEND["浏览器前端<br>WebSocket"]:::userStyle
    WS["FastAPI WebSocket<br>音频流接入"]:::agentStyle

    subgraph ROOM["ChatRoom 实时多 Agent"]
        direction TB
        AGENT1["RealtimeAgent<br>Dylan 音色"]:::agentStyle
        AGENT2["RealtimeAgent<br>Peter 音色"]:::agentStyle
    end
    class ROOM layerStyle

    subgraph MODELS["实时语音模型"]
        direction LR
        DS["DashScope<br>qwen3-omni-flash-realtime"]:::roomStyle
        GM["Gemini<br>gemini-2.5-flash-native-audio"]:::roomStyle
        OA["OpenAI<br>gpt-4o-realtime-preview"]:::roomStyle
    end
    class MODELS layerStyle

    FRONTEND -->|"音频帧 / 文本"| WS
    WS        -->|"ClientEvent"| ROOM
    AGENT1    <-->|"双向音频流"| DS
    AGENT2    <-->|"双向音频流"| DS
    ROOM      -->|"ServerEvent"| WS
    WS        -->|"音频回放"| FRONTEND

    NOTE["ChatRoom 管理多个 RealtimeAgent 的生命周期<br>chat_room.start(queue) / chat_room.stop()"]:::noteStyle
    NOTE -.- ROOM

    %% 边索引：0-5，共 6 条
    linkStyle 0,1 stroke:#1e40af,stroke-width:2px
    linkStyle 2,3 stroke:#dc2626,stroke-width:2px
    linkStyle 4,5 stroke:#059669,stroke-width:2px
```

**使用方式：**

```python
from agentscope.pipeline import ChatRoom

chat_room = ChatRoom(agents=[agent1, agent2])
await chat_room.start(frontend_queue)
# ... 处理客户端事件 ...
await chat_room.stop()
```

> 来源：`examples/workflows/multiagent_realtime/run_server.py`

---

## 五、复杂任务场景实战

### 场景一：单 Agent 工具调用（ReAct 基础用法）

**适用任务：** 代码执行、文件操作、Shell 命令等需要工具辅助的单轮任务。

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef llmStyle     fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef toolStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    USER["用户输入<br>User Input"]:::userStyle

    subgraph REACT_LOOP["ReAct 推理-行动循环"]
        direction LR
        THINK["LLM 推理<br>Thought"]:::llmStyle
        ACT["工具调用决策<br>Action"]:::agentStyle
        OBS["工具结果观察<br>Observation"]:::toolStyle
    end
    class REACT_LOOP layerStyle

    subgraph TOOLS["工具层"]
        direction LR
        SHELL["execute_shell_command"]:::toolStyle
        PYTHON["execute_python_code"]:::toolStyle
        FILE["view_text_file"]:::toolStyle
    end
    class TOOLS layerStyle

    FINISH["最终回答<br>Final Answer"]:::userStyle

    USER   --> THINK
    THINK  -->|"需要工具"| ACT
    ACT    -->|"调用"| TOOLS
    TOOLS  -->|"结果"| OBS
    OBS    -->|"继续推理"| THINK
    THINK  -->|"完成"| FINISH

    NOTE["max_iters 控制最大循环次数<br>ReActAgent 内置 finish 工具终止循环"]:::noteStyle
    NOTE -.- REACT_LOOP

    %% 边索引：0-5，共 6 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1,2 stroke:#d97706,stroke-width:2px
    linkStyle 3,4 stroke:#0891b2,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 5 stroke:#dc2626,stroke-width:2.5px
```

> **示例代码：** `examples/agent/react_agent/main.py`

---

### 场景二：RAG 知识增强问答

**适用任务：** 基于私有知识库（文档、个人信息、企业数据）的智能问答。

```mermaid
flowchart LR
    classDef userStyle     fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle    fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef llmStyle      fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef retrieveStyle fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef dbStyle       fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle     fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle    fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    USER["用户提问<br>User Query"]:::userStyle

    subgraph RAG_FLOW["RAG 增强检索层"]
        direction LR
        EMBED["文本向量化<br>DashScope Embedding"]:::retrieveStyle
        SEARCH["向量相似度检索<br>Qdrant Search"]:::retrieveStyle
        RERANK["结果筛选<br>score_threshold 过滤"]:::retrieveStyle
    end
    class RAG_FLOW layerStyle

    VDB[("向量数据库<br>Qdrant In-Memory")]:::dbStyle

    LLM["LLM 推理<br>上下文 + 知识片段"]:::llmStyle

    ANSWER["增强回答<br>Answer"]:::userStyle

    USER    --> EMBED
    EMBED   -->|"查询向量"| SEARCH
    SEARCH  -->|"相似度查询"| VDB
    VDB     -->|"相关文档"| RERANK
    RERANK  -->|"注入上下文"| LLM
    USER    -->|"原始问题"| LLM
    LLM     --> ANSWER

    NOTE["Agent 通过 retrieve_knowledge 工具主动调整<br>查询参数（query / limit / score_threshold）<br>实现自适应检索"]:::noteStyle
    NOTE -.- RAG_FLOW

    %% 边索引：0-6，共 7 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1,2 stroke:#d97706,stroke-width:2px
    linkStyle 3,4 stroke:#059669,stroke-width:2px
    linkStyle 5 stroke:#1e40af,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 6 stroke:#dc2626,stroke-width:2.5px
```

> **示例代码：** `examples/functionality/rag/agentic_usage.py`

---

### 场景三：多 Agent 辩论（MsgHub + 结构化输出）

**适用任务：** 集体决策、答案验证、多视角论证（如数学推理、事实核查）。

```mermaid
flowchart LR
    classDef userStyle   fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle  fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef judgeStyle  fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef hubStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    TOPIC["辩题<br>Topic"]:::userStyle

    subgraph ROUND["单轮辩论（MsgHub 广播域）"]
        direction LR
        ALICE["Alice<br>正方辩手"]:::agentStyle
        BOB["Bob<br>反方辩手"]:::agentStyle
        HUB["MsgHub<br>广播中枢"]:::hubStyle
    end
    class ROUND layerStyle

    JUDGE["Aggregator<br>裁判 Agent"]:::judgeStyle

    subgraph VERDICT["结构化裁决"]
        direction LR
        FINISHED["finished: bool"]:::judgeStyle
        ANSWER["correct_answer: str | None"]:::judgeStyle
    end
    class VERDICT layerStyle

    TOPIC  -->|"正方立场"| ALICE
    TOPIC  -->|"反方立场"| BOB
    ALICE  -->|"回复广播"| HUB
    HUB    -->|"转发给 Bob"| BOB
    BOB    -->|"回复广播"| HUB
    HUB    -->|"转发给 Alice"| ALICE
    ALICE  -->|"辩论内容"| JUDGE
    BOB    -->|"辩论内容"| JUDGE
    JUDGE  -->|"JudgeModel"| VERDICT

    NOTE["裁判 Agent 在 MsgHub 外部调用<br>使用 structured_model=JudgeModel 获取结构化输出<br>finished=True 时终止循环"]:::noteStyle
    NOTE -.- JUDGE

    %% 边索引：0-8，共 9 条
    linkStyle 0,1 stroke:#1e40af,stroke-width:2px
    linkStyle 2,3 stroke:#d97706,stroke-width:2px
    linkStyle 4,5 stroke:#d97706,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 6,7 stroke:#0891b2,stroke-width:1.5px
    linkStyle 8 stroke:#dc2626,stroke-width:2.5px
```

> **示例代码：** `examples/workflows/multiagent_debate/main.py`

---

### 场景四：Meta Planner — 元规划 Agent 编排子 Agent

**适用任务：** 复杂长任务自动分解 → 创建专属子 Worker → 协调执行（旅行规划、深度调研等）。

```mermaid
flowchart LR
    classDef userStyle   fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef planStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef workerStyle fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef toolStyle   fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef dbStyle     fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    USER["用户复杂任务<br>Complex Task"]:::userStyle

    PLANNER["Friday（Meta Planner）<br>ReActAgent + PlanNotebook"]:::planStyle

    subgraph DECOMPOSE["任务分解层"]
        direction LR
        ST1["子任务 1"]:::planStyle
        ST2["子任务 2"]:::planStyle
        ST3["子任务 N"]:::planStyle
    end
    class DECOMPOSE layerStyle

    subgraph WORKERS["动态创建 Worker 层"]
        direction TB
        W1["Worker 1<br>ReActAgent"]:::workerStyle
        W2["Worker 2<br>ReActAgent"]:::workerStyle
        W3["Worker N<br>ReActAgent"]:::workerStyle
    end
    class WORKERS layerStyle

    subgraph MCP_TOOLS["MCP 工具层"]
        direction LR
        BROWSER["Playwright<br>浏览器"]:::toolStyle
        AMAP["高德地图<br>API"]:::toolStyle
        GITHUB["GitHub<br>API"]:::toolStyle
    end
    class MCP_TOOLS layerStyle

    RESULT["综合结果<br>Final Result"]:::userStyle

    USER       --> PLANNER
    PLANNER    -->|"PlanNotebook 分解"| DECOMPOSE
    ST1        -->|"create_worker()"| W1
    ST2        -->|"create_worker()"| W2
    ST3        -->|"create_worker()"| W3
    W1         -->|"MCP 调用"| MCP_TOOLS
    W2         -->|"MCP 调用"| MCP_TOOLS
    W3         -->|"MCP 调用"| MCP_TOOLS
    W1         -->|"ResultModel"| PLANNER
    W2         -->|"ResultModel"| PLANNER
    W3         -->|"ResultModel"| PLANNER
    PLANNER    --> RESULT

    NOTE["create_worker 是注册给 Planner 的工具函数<br>每次调用动态创建一个 ReActAgent 实例<br>使用 stream_printing_messages 实时流式展示进度"]:::noteStyle
    NOTE -.- WORKERS

    %% 边索引：0-11，共 12 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1 stroke:#dc2626,stroke-width:2px
    linkStyle 2,3,4 stroke:#dc2626,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 5,6,7 stroke:#d97706,stroke-width:2px
    linkStyle 8,9,10 stroke:#0891b2,stroke-width:2px
    linkStyle 11 stroke:#1e40af,stroke-width:2.5px
```

> **示例代码：** `examples/agent/meta_planner_agent/`

---

### 场景五：Deep Research Agent — 深度研究

**适用任务：** 需要网络搜索 + 文件读写 + 多步推理的研究型任务（如文献调研、数据收集分析）。

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef searchStyle  fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef fileStyle    fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    QUERY["研究问题<br>Research Query"]:::userStyle

    AGENT["DeepResearchAgent<br>ReActAgent 扩展"]:::agentStyle

    subgraph SEARCH_LAYER["搜索层"]
        direction LR
        TAVILY["Tavily MCP 搜索<br>StdIOStatefulClient"]:::searchStyle
        WEB["网页内容抓取<br>Web Scraping"]:::searchStyle
    end
    class SEARCH_LAYER layerStyle

    subgraph STORAGE["本地文件存储"]
        direction LR
        NOTES[("中间笔记<br>Tmp Files")]:::fileStyle
        REPORT[("最终报告<br>Final Report")]:::fileStyle
    end
    class STORAGE layerStyle

    RESULT["研究结论<br>Research Result"]:::userStyle

    QUERY  --> AGENT
    AGENT  -->|"搜索工具调用"| SEARCH_LAYER
    SEARCH_LAYER -->|"搜索结果（≤10k words）"| AGENT
    AGENT  -->|"写入中间结果"| NOTES
    NOTES  -->|"读取上下文"| AGENT
    AGENT  -->|"生成报告"| REPORT
    REPORT --> RESULT

    NOTE["max_tool_results_words=10000 防止上下文溢出<br>使用工作目录隔离临时文件<br>支持链式搜索：每次搜索结果驱动下一次查询"]:::noteStyle
    NOTE -.- AGENT

    %% 边索引：0-6，共 7 条
    linkStyle 0,1 stroke:#1e40af,stroke-width:2px
    linkStyle 2 stroke:#d97706,stroke-width:2px
    linkStyle 3 stroke:#059669,stroke-width:2px
    linkStyle 4 stroke:#059669,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 5,6 stroke:#059669,stroke-width:2px
```

> **示例代码：** `examples/agent/deep_research_agent/`

---

### 场景六：Browser Agent — 浏览器自动化

**适用任务：** 网页操作、表单填写、数据采集、UI 自动化测试。

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef mcpStyle     fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef browserStyle fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    USER["用户指令<br>Browser Task"]:::userStyle

    AGENT["BrowserAgent<br>ReActAgent 扩展"]:::agentStyle

    subgraph MCP_LAYER["MCP 浏览器控制层"]
        direction LR
        PLAYWRIGHT["Playwright MCP<br>StdIOStatefulClient"]:::mcpStyle
        TOOLS["browser_snapshot<br>browser_click<br>browser_type<br>browser_navigate"]:::mcpStyle
    end
    class MCP_LAYER layerStyle

    subgraph BROWSER["真实浏览器（Chromium）"]
        direction LR
        PAGE["网页渲染<br>Page"]:::browserStyle
        DOM["DOM 操作<br>Element Interaction"]:::browserStyle
    end
    class BROWSER layerStyle

    RESULT["任务结果<br>FinalResult (Structured)"]:::userStyle

    USER      --> AGENT
    AGENT     -->|"工具调用"| MCP_LAYER
    MCP_LAYER -->|"CDP 协议"| BROWSER
    BROWSER   -->|"截图/DOM 快照"| MCP_LAYER
    MCP_LAYER -->|"工具结果"| AGENT
    AGENT     -->|"structured_model=FinalResult"| RESULT

    NOTE["start_url 指定初始页面<br>max_iters=50 控制操作步数上限<br>支持多模态输入（截图理解）"]:::noteStyle
    NOTE -.- AGENT

    %% 边索引：0-5，共 6 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1 stroke:#d97706,stroke-width:2px
    linkStyle 2 stroke:#059669,stroke-width:2px
    linkStyle 3 stroke:#059669,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 4 stroke:#d97706,stroke-width:2px
    linkStyle 5 stroke:#1e40af,stroke-width:2.5px
```

> **示例代码：** `examples/agent/browser_agent/`

---

## 六、结构化输出（Structured Output）

任何 `ReActAgent` 调用时均可传入 `structured_model` 参数，强制 Agent 输出符合 Pydantic 模型定义的 JSON 结构，结果存入 `msg.metadata`。

```python
from pydantic import BaseModel, Field

class JudgeModel(BaseModel):
    finished: bool = Field(description="Whether the debate is finished.")
    correct_answer: str | None = Field(default=None)

# 调用时传入结构化模型
msg = await moderator(
    Msg("user", "Is the debate finished?", "user"),
    structured_model=JudgeModel,
)

# 结果在 metadata 中
if msg.metadata.get("finished"):
    print(msg.metadata.get("correct_answer"))
```

---

## 七、MCP 工具接入模式对比

```mermaid
flowchart TB
    classDef clientStyle  fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef svcStyle     fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef statefulStyle fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef statelessStyle fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef stdioStyle   fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    TOOLKIT["Toolkit<br>工具注册中心"]:::clientStyle

    subgraph HTTP_STATEFUL["HTTP 有状态客户端<br>HttpStatefulClient"]
        direction LR
        SSE["SSE Transport<br>持久连接 / 会话保持"]:::statefulStyle
    end
    class HTTP_STATEFUL layerStyle

    subgraph HTTP_STATELESS["HTTP 无状态客户端<br>HttpStatelessClient"]
        direction LR
        STREAM_HTTP["Streamable HTTP<br>每次请求独立 / 无需 connect()"]:::statelessStyle
    end
    class HTTP_STATELESS layerStyle

    subgraph STDIO["本地进程客户端<br>StdIOStatefulClient"]
        direction LR
        NPX["npx / 本地命令<br>Playwright / Tavily 等"]:::stdioStyle
    end
    class STDIO layerStyle

    TOOLKIT --> HTTP_STATEFUL
    TOOLKIT --> HTTP_STATELESS
    TOOLKIT --> STDIO

    NOTE["有状态客户端必须手动 connect() 和 close()<br>无状态客户端可直接 register，无需连接管理<br>StdIO 适合驱动本地 npx 工具进程"]:::noteStyle
    NOTE -.- TOOLKIT

    %% 边索引：0-2，共 3 条
    linkStyle 0 stroke:#dc2626,stroke-width:2px
    linkStyle 1 stroke:#d97706,stroke-width:2px
    linkStyle 2 stroke:#059669,stroke-width:2px
```

| 客户端类型 | 连接方式 | 状态管理 | 典型使用场景 |
|---|---|---|---|
| `HttpStatefulClient` | HTTP SSE / 持久连接 | 需手动 `connect()` / `close()` | 需要会话状态的服务（如计算上下文） |
| `HttpStatelessClient` | HTTP Streamable / 无连接 | 无需管理 | 无状态 REST-like 工具（地图 API、搜索 API） |
| `StdIOStatefulClient` | 本地进程 stdio | 需手动 `connect()` / `close()` | 本地 npx 工具（Playwright、Tavily MCP 等） |

---

## 八、最佳实践速查

| 场景 | 推荐方案 |
|------|---------|
| 单 Agent + 工具调用 | `ReActAgent` + `Toolkit`（注册内置工具函数） |
| 需要联网搜索 | `ReActAgent` + `StdIOStatefulClient`（Tavily MCP） |
| 需要操作浏览器 | `BrowserAgent` + `StdIOStatefulClient`（Playwright MCP） |
| 多 Agent 轮流对话 | `MsgHub` + `sequential_pipeline` |
| 多 Agent 同时并行 | `fanout_pipeline(enable_gather=True)` |
| 多 Agent 辩论/投票 | `MsgHub` + 结构化输出（`JudgeModel`）控制终止条件 |
| 复杂任务自动分解 | `ReActAgent` + `PlanNotebook` + `create_worker` 动态创建子 Agent |
| 实时语音多 Agent | `RealtimeAgent` + `ChatRoom` + WebSocket 服务 |
| 跨框架 Agent 调用 | `A2AAgent` + `AgentCard`（Google A2A 协议） |
| 私有知识库问答 | `ReActAgent` + `SimpleKnowledge`（Qdrant / Milvus）+ RAG 工具 |
| 需要持久化记忆 | `InMemoryMemory`（短期）或 `RemeMemory` / `Mem0Memory`（长期） |
| 需要强制结构化输出 | 调用时传入 `structured_model=YourPydanticModel` |

---

> **文档版本：** 基于 AgentScope v1.0.16 `examples/` 目录分析生成
> **参考示例目录：**
> - `examples/agent/` — 各类专用 Agent 实现
> - `examples/workflows/` — 多 Agent 编排工作流
> - `examples/functionality/` — 工具/记忆/RAG/规划等功能模块
