# Meta Planner Agent vs Planning Agent 对比分析

## 一、核心功能对比

| 维度 | Meta Planner Agent | Planning Agent |
|------|-------------------|----------------|
| **主要用途** | 本地演示规划代理的任务分解与子代理协调能力 | 部署为HTTP服务的路由代理系统 |
| **运行环境** | 本地控制台交互式运行 | Quart服务器部署运行 |
| **会话管理** | 无持久化会话管理 | 使用JSONSession保存和加载会话状态 |
| **输出方式** | 控制台直接输出 | SSE流式响应 |
| **核心特性** | 任务分解、子代理协调、中断事件传播 | 服务器部署、会话持久化、API接口 |
| **技术栈** | AgentScope核心库 | AgentScope + Quart |

## 二、系统架构对比

### 2.1 Meta Planner Agent 架构

```mermaid
flowchart TB
    %% 配色定义
    classDef userStyle fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle fill:#4f46e5,stroke:#3730a3,stroke-width:2px,color:#fff
    classDef toolStyle fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef modelStyle fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef noteStyle fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px
    
    %% 核心组件
    USER["用户输入"]:::userStyle
    PLANNER["规划代理<br>ReActAgent"]:::agentStyle
    PLAN["计划管理<br>PlanNotebook"]:::toolStyle
    CREATE_WORKER["创建子代理<br>create_worker"]:::toolStyle
    SUB_AGENTS["子代理<br>Worker Agents"]:::agentStyle
    LLM["大语言模型<br>DashScopeChatModel"]:::modelStyle
    MCP["MCP服务<br>GitHub/AMap/Playwright"]:::toolStyle
    
    %% 数据流
    USER --> PLANNER
    PLANNER --> PLAN
    PLANNER --> CREATE_WORKER
    CREATE_WORKER --> SUB_AGENTS
    PLANNER --> LLM
    SUB_AGENTS --> LLM
    SUB_AGENTS --> MCP
    SUB_AGENTS --> PLANNER
    PLANNER --> USER
    
    %% 设计注记
    NOTE["核心特性<br>① 任务分解与计划管理<br>② 动态创建子代理<br>③ 子代理输出处理<br>④ 中断事件传播"]:::noteStyle
    NOTE -.- PLANNER
    
    %% 边样式
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1,2 stroke:#4f46e5,stroke-width:2px
    linkStyle 3 stroke:#0891b2,stroke-width:2px
    linkStyle 4,5 stroke:#dc2626,stroke-width:2px
    linkStyle 6 stroke:#0891b2,stroke-width:2px
    linkStyle 7 stroke:#4f46e5,stroke-width:2px
    linkStyle 8 stroke:#1e40af,stroke-width:2px
```

### 2.2 Planning Agent 架构

```mermaid
flowchart TB
    %% 配色定义
    classDef userStyle fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef serverStyle fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle fill:#4f46e5,stroke:#3730a3,stroke-width:2px,color:#fff
    classDef toolStyle fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef modelStyle fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef storageStyle fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px
    
    %% 核心组件
    CLIENT["客户端<br>HTTP Request"]:::userStyle
    QUART["Quart服务器<br>/chat_endpoint"]:::serverStyle
    SESSION["会话管理<br>JSONSession"]:::storageStyle
    ROUTER["路由代理<br>ReActAgent"]:::agentStyle
    CREATE_WORKER["创建子代理<br>create_worker"]:::toolStyle
    SUB_AGENTS["子代理<br>Worker Agents"]:::agentStyle
    LLM["大语言模型<br>DashScopeChatModel"]:::modelStyle
    MCP["MCP服务<br>GitHub/AMap/Playwright"]:::toolStyle
    
    %% 数据流
    CLIENT --> QUART
    QUART --> SESSION
    QUART --> ROUTER
    ROUTER --> CREATE_WORKER
    CREATE_WORKER --> SUB_AGENTS
    ROUTER --> LLM
    SUB_AGENTS --> LLM
    SUB_AGENTS --> MCP
    SUB_AGENTS --> ROUTER
    ROUTER --> SESSION
    ROUTER --> QUART
    QUART --> CLIENT
    
    %% 设计注记
    NOTE["核心特性<br>① HTTP服务器部署<br>② 会话状态持久化<br>③ SSE流式响应<br>④ 多用户支持"]:::noteStyle
    NOTE -.- QUART
    
    %% 边样式
    linkStyle 0,9 stroke:#1e40af,stroke-width:2px
    linkStyle 1,7 stroke:#059669,stroke-width:2px
    linkStyle 2,8 stroke:#1d4ed8,stroke-width:2px
    linkStyle 3,6 stroke:#4f46e5,stroke-width:2px
    linkStyle 4 stroke:#0891b2,stroke-width:2px
    linkStyle 5,10 stroke:#dc2626,stroke-width:2px
    linkStyle 11 stroke:#0891b2,stroke-width:2px
```

## 三、实现细节对比

### 3.1 启动方式

**Meta Planner Agent**:
- 直接在控制台运行 `python main.py`
- 交互式命令行界面
- 支持通过 AgentScope-Studio 可视化代理交互

**Planning Agent**:
- 启动 Quart 服务器 `python main.py`
- 通过 HTTP POST 请求访问 `/chat_endpoint` 端点
- 提供 `test_post.py` 脚本用于测试

### 3.2 核心实现差异

| 特性 | Meta Planner Agent | Planning Agent |
|------|-------------------|----------------|
| **会话管理** | 无会话管理，每次运行都是新会话 | 使用 JSONSession 持久化会话状态 |
| **计划管理** | 使用 PlanNotebook 管理任务计划 | 无计划管理功能 |
| **输出处理** | 子代理输出作为工具流式响应返回 | 子代理输出转换为 SSE 流式响应 |
| **部署方式** | 本地运行，无网络服务 | 部署为 HTTP 服务，支持远程访问 |
| **并发处理** | 单用户交互式 | 支持多用户并发请求 |

### 3.3 代码结构对比

**Meta Planner Agent**:
```
meta_planner_agent/
    ├── assets/           # 截图资源
    ├── README.md         # 说明文档
    ├── main.py           # 主入口，创建规划代理
    └── tool.py           # 工具函数，创建子代理
```

**Planning Agent**:
```
planning_agent/
    ├── README.md         # 说明文档
    ├── main.py           # 主入口，启动Quart服务器
    ├── tool.py           # 工具函数，创建子代理
    └── test_post.py      # 测试脚本，发送HTTP请求
```

## 四、适用场景

**Meta Planner Agent** 适用于：
- 本地开发和测试规划代理功能
- 学习任务分解和子代理协调的实现
- 演示 AgentScope 的核心能力
- 不需要持久化会话的场景

**Planning Agent** 适用于：
- 生产环境部署多代理系统
- 需要通过 API 接口访问代理功能
- 要求会话状态持久化的场景
- 多用户并发访问的场景

## 五、技术要点

### 5.1 子代理创建与管理

两个示例都使用 `create_worker` 工具函数动态创建子代理，但实现细节有所不同：

- **Meta Planner Agent**：通过 PlanNotebook 管理任务计划，按计划顺序执行子任务
- **Planning Agent**：作为路由代理，根据用户请求动态创建子代理处理特定任务

### 5.2 输出处理

- **Meta Planner Agent**：子代理输出通过工具函数的流式响应返回给规划代理，再由规划代理展示给用户
- **Planning Agent**：子代理输出转换为 SSE 流式响应，通过 HTTP 连接实时返回给客户端

### 5.3 会话管理

- **Meta Planner Agent**：无会话管理，每次运行都是独立的会话
- **Planning Agent**：使用 JSONSession 保存和加载会话状态，支持会话持久化

## 六、总结

**Meta Planner Agent** 是一个本地演示示例，重点展示了 AgentScope 中规划代理的核心能力，包括任务分解、子代理协调和中断事件传播。它适合用于学习和测试 AgentScope 的基本功能。

**Planning Agent** 是一个部署示例，展示了如何将多代理系统部署为 HTTP 服务，支持会话持久化和 SSE 流式响应。它适合用于生产环境部署和通过 API 接口访问代理功能。

两个示例互补，分别展示了 AgentScope 在不同场景下的应用方式，为开发者提供了从本地开发到生产部署的完整参考。