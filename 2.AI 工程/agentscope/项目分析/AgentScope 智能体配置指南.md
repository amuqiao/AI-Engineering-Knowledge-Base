# AgentScope 智能体配置指南

## 智能体配置方式

**不是必须**按照示例中的文件夹结构来配置你的智能体。AgentScope 设计为一个灵活的库，你可以根据自己的项目结构来使用它。

### 两种主要使用方式：

1. **作为库集成到现有项目**：
   - 将 AgentScope 安装为依赖
   - 在你的项目代码中直接创建和配置智能体
   - 适用于大多数场景，特别是将智能体功能集成到现有应用中

2. **参考示例结构**：
   - 当你需要构建独立的智能体应用时
   - 示例中的文件夹结构提供了一种组织方式，但不是强制要求
   - 适合构建完整的智能体服务

## 前端页面配置

**前端页面不是必须的**，这取决于你的具体需求：

- **无前端场景**：如果你的智能体在后端运行，通过 API 或其他方式交互，不需要前端页面
- **简单交互**：可以使用命令行或简单的文本界面
- **复杂交互**：当需要更丰富的用户界面时，可以参考以下方式：
  - **内置示例**：如 `realtime_voice_agent` 提供了 Web 界面
  - **自定义前端**：根据你的需求构建前端页面，通过 API 与智能体通信
  - **AgentScope Studio**：使用官方提供的本地部署 Web 应用，提供项目管理和可视化追踪功能

## 与 AgentScope Studio 搭配使用

AgentScope Studio 是一个本地部署的 Web 应用程序，为智能体应用提供项目管理和可视化追踪功能。

### 1. 安装 AgentScope Studio

使用 npm 安装：

```bash
npm install -g @agentscope/studio
```

### 2. 启动 AgentScope Studio

```bash
as_studio
```

默认情况下，Studio 会在 `http://localhost:3000` 启动。

### 3. 配置智能体连接到 Studio

在初始化 AgentScope 时，添加 `studio_url` 参数：

```python
import agentscope

# 初始化并连接到 Studio
agentscope.init(
    project="MyProject",
    name="TestRun",
    studio_url="http://localhost:3000"
)

# 后续的智能体创建和运行代码...
```

### 4. 完整示例

```python
import asyncio
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
from agentscope.message import Msg
import agentscope

async def main():
    # 初始化并连接到 AgentScope Studio
    agentscope.init(
        project="DemoProject",
        name="ChatAssistant",
        studio_url="http://localhost:3000"
    )

    # 创建智能体
    agent = ReActAgent(
        name="assistant",
        sys_prompt="You're a helpful assistant.",
        model=OpenAIChatModel(
            model_name="gpt-4",
            api_key="YOUR_API_KEY",
        ),
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )

    # 运行智能体
    msg = None
    while True:
        msg = await agent(msg)
        user_input = input("User: ")
        if user_input == "exit":
            break
        msg = Msg("user", user_input, "user")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. 在 Studio 中查看和管理

启动应用后，你可以在 AgentScope Studio 中：

- **项目管理**：查看所有连接的智能体应用
- **运行状态**：实时查看应用的运行状态和日志
- **可视化追踪**：查看 token 使用情况、模型调用和详细追踪信息
- **Friday 智能体**：使用内置的 Friday 智能体获取开发帮助

### 6. 高级配置

你可以同时配置追踪功能，将追踪数据发送到 Studio：

```python
agentscope.init(
    project="MyProject",
    name="TestRun",
    studio_url="http://localhost:3000",
    # 自动使用 Studio 的追踪端点
    # tracing_url 会自动设置为 studio_url + "/v1/traces"
)
```

## 智能体配置步骤

1. **安装 AgentScope**：
   ```bash
   pip install agentscope
   # 或安装特定功能
   pip install "agentscope[full]"
   ```

2. **创建智能体**：
   ```python
   from agentscope.agent import ReActAgent
   from agentscope.model import OpenAIChatModel
   from agentscope.memory import InMemoryMemory
   from agentscope.tool import Toolkit

   agent = ReActAgent(
       name="assistant",
       sys_prompt="You're a helpful assistant.",
       model=OpenAIChatModel(
           model_name="gpt-4",
           api_key="YOUR_API_KEY",
       ),
       memory=InMemoryMemory(),
       toolkit=Toolkit(),
   )
   ```

3. **配置存储**：
   - 使用默认内存存储（开发测试）
   - 配置 SQL 数据库或 Redis（生产环境）

4. **添加工具**：
   ```python
   from agentscope.tool import execute_python_code, execute_shell_command

   toolkit = Toolkit()
   toolkit.register_tool_function(execute_python_code)
   toolkit.register_tool_function(execute_shell_command)
   ```

5. **运行智能体**：
   ```python
   async def main():
       msg = None
       while True:
           msg = await agent(msg)
           user_input = input("User: ")
           if user_input == "exit":
               break
           msg = Msg("user", user_input, "user")

   asyncio.run(main())
   ```

## 项目结构建议

### 小型项目
```
your_project/
├── main.py          # 主入口
└── requirements.txt # 依赖
```

### 中型项目
```
your_project/
├── app/
│   ├── agents/      # 智能体定义
│   ├── models/      # 模型配置
│   ├── tools/       # 自定义工具
│   └── memory/      # 存储配置
├── main.py          # 主入口
└── requirements.txt # 依赖
```

### 大型项目
```
your_project/
├── backend/
│   ├── agents/      # 智能体定义
│   ├── api/         # API 接口
│   ├── services/    # 业务逻辑
│   └── config/      # 配置
├── frontend/        # 前端页面（可选）
├── main.py          # 主入口
└── requirements.txt # 依赖
```

## 示例的作用

示例文件夹中的代码主要是：
1. **功能展示**：展示 AgentScope 的各种功能和用法
2. **参考实现**：提供特定场景的实现示例
3. **学习资源**：帮助你理解如何使用 AgentScope

你可以根据自己的需求参考示例，但不需要完全按照示例的结构来组织你的项目。

## 总结

- **灵活性**：AgentScope 可以集成到各种项目结构中
- **可扩展性**：支持从简单脚本到复杂应用的各种场景
- **前端可选**：根据需求决定是否需要前端页面
- **参考示例**：示例是学习资源，不是强制结构

通过合理配置，你可以构建从简单到复杂的各种智能体应用，而不需要严格遵循示例中的文件夹结构。