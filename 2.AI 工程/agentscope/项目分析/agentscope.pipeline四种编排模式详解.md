## agentscope.pipeline 四种编排模式详解

### 1. SequentialPipeline — 顺序链式管道

**源码机制**：将一组 Agent 排成一条流水线，**前一个 Agent 的输出自动作为下一个 Agent 的输入**，最终返回最后一个 Agent 的输出。

```1:44:src/agentscope/pipeline/_functional.py
# -*- coding: utf-8 -*-
"""Functional counterpart for Pipeline"""
import asyncio
from copy import deepcopy
from typing import Any, AsyncGenerator, Tuple, Coroutine
from ..agent import AgentBase
from ..message import Msg, AudioBlock


async def sequential_pipeline(
    agents: list[AgentBase],
    msg: Msg | list[Msg] | None = None,
) -> Msg | list[Msg] | None:
    """An async syntactic sugar pipeline that executes a sequence of agents
    sequentially. The output of the previous agent will be passed as the
    input to the next agent. The final output will be the output of the
    last agent.
    ...
    """
    for agent in agents:
        msg = await agent(msg)
    return msg
```

**执行流**：
```
输入消息 → Agent A → 输出A → Agent B → 输出B → Agent C → 最终输出
```

**适用场景**：
- **多步骤加工流水线**：文本生成 → 质量检测 → 格式化润色
- **链式推理**：问题分解 Agent → 子问题求解 Agent → 答案汇总 Agent
- **翻译-审校流**：翻译 Agent → 校对 Agent → 排版 Agent
- **代码生成**：需求理解 Agent → 代码编写 Agent → Code Review Agent

---

### 2. FanoutPipeline — 扇出并行管道

**源码机制**：将**同一条消息的深拷贝**分发给多个 Agent，所有 Agent **并发执行**（`asyncio.gather`），最后收集所有响应放入列表返回。

```47:104:src/agentscope/pipeline/_functional.py
async def fanout_pipeline(
    agents: list[AgentBase],
    msg: Msg | list[Msg] | None = None,
    enable_gather: bool = True,
    **kwargs: Any,
) -> list[Msg]:
    """...distributes the same input to multiple agents..."""
    if enable_gather:
        tasks = [
            asyncio.create_task(agent(deepcopy(msg), **kwargs))
            for agent in agents
        ]
        return await asyncio.gather(*tasks)
    else:
        return [await agent(deepcopy(msg), **kwargs) for agent in agents]
```

**执行流**：
```
                 ┌→ Agent A（并发）→ 响应A ┐
输入消息（deepcopy）├→ Agent B（并发）→ 响应B ├→ [响应A, 响应B, 响应C]
                 └→ Agent C（并发）→ 响应C ┘
```

**适用场景**：
- **多模型/多视角投票**：同一问题发给 3 个不同模型的 Agent，结果投票取最优
- **专家组并行评估**：法律专家 Agent + 财务专家 Agent + 技术专家 Agent 同时分析同一份合同
- **多角色分析**：正方辩手 Agent + 反方辩手 Agent 分别生成观点（注意：各自独立，不互知，适合第一轮）
- **并行子任务拆分**：将一个大任务的不同维度同时分发给不同 Agent 处理（比 Sequential 快得多）

---

### 3. MsgHub — 消息广播总线

**源码机制**：异步上下文管理器，将一组 Agent 接入同一个"消息广播室"。**任意 Agent 发出的回复消息，自动广播给所有其他参与者**（通过调用其他 Agent 的 `observe` 方法），无需手动传递消息。支持动态加减参与者。

```14:156:src/agentscope/pipeline/_msghub.py
class MsgHub:
    """MsgHub class that controls the subscription of the participated agents.

    Example:
        In the following example, the reply message from `agent1`, `agent2`,
        and `agent3` will be broadcast to all the other agents in the MsgHub.

        .. code-block:: python

            with MsgHub(participant=[agent1, agent2, agent3]):
                agent1()
                agent2()
        ...
    """
```

真实示例（来自 `multiagent_debate/main.py`）：

```85:127:examples/workflows/multiagent_debate/main.py
async def run_multiagent_debate() -> None:
    """Run the multi-agent debate workflow."""
    while True:
        # The reply messages in MsgHub from the participants will be
        # broadcasted to all participants.
        async with MsgHub(participants=[alice, bob, moderator]):
            await alice(
                Msg("user", "You are affirmative side...", "user"),
            )
            await bob(
                Msg("user", "You are negative side...", "user"),
            )
        # Alice and Bob doesn't need to know the moderator's message,
        # so moderator is called outside the MsgHub.
        msg_judge = await moderator(...)
```

**执行流**：
```
Alice 发言 → 自动广播给 Bob、Moderator（observe）
Bob 发言 → 自动广播给 Alice、Moderator（observe）
Moderator 被移出 Hub 后单独裁决（不广播回辩手）
```

**适用场景**：
- **多 Agent 辩论/协商**：多个 Agent 轮流发言，每人发言所有人都知道
- **头脑风暴会议**：创意 Agent A + 批评 Agent B + 综合 Agent C 互相讨论
- **角色扮演游戏**（如狼人杀）：所有玩家 Agent 处于同一"公开频道"，谁发言谁广播
- **多智能体圆桌**：任意 Agent 随时切入对话，其他人自动感知上下文

**与 SequentialPipeline 组合使用**（最常见模式）：
```python
async with MsgHub(participants=[alice, bob, charlie]) as hub:
    # 顺序轮流发言，且每人发言自动广播给其他人
    await sequential_pipeline([alice, bob, charlie])
    # 动态移除中途退出的成员
    hub.delete(bob)
```

---

### 4. ChatRoom — 实时语音/多模态对话房间

**源码机制**：专为 **`RealtimeAgent`**（实时语音 Agent）设计的房间级广播器，基于 WebSocket 事件流。区分 `ClientEvents`（来自前端用户）和 `ServerEvents`（来自 Agent）两类事件，分别做路由：前端事件分发给所有 Agent；Agent 事件转发给前端，同时广播给其他 Agent（排除发送方自身）。

```10:97:src/agentscope/pipeline/_chat_room.py
class ChatRoom:
    """The chat room abstraction to broadcast messages among multiple realtime
    agents, and handle the messages from the frontend.
    """
    # ClientEvents → 分发给所有 Agent
    # ServerEvents → 转发到前端队列 + 广播给其他 Agent（排除发送方）
```

**与 MsgHub 的本质区别**：

| 维度 | MsgHub | ChatRoom |
|------|--------|---------|
| 通信协议 | Python 对象传递（`observe`） | WebSocket 事件流（`ClientEvents`/`ServerEvents`） |
| Agent 类型 | 通用 `AgentBase` | 专用 `RealtimeAgent` |
| 实时性 | 异步任务级 | 实时音频流级（低延迟） |
| 使用场景 | 文本多智能体协作 | 实时语音对话、多人语音会议 |

**适用场景**：
- **实时语音助手**：多个语音 Agent 接入同一房间，用户说话由房间广播给所有 Agent
- **AI 语音会议**：多个专业领域的语音 Agent 同时参与，支持实时打断和轮转
- **实时多模态协作**：用户上传图片/说话，多个专业 Agent 同步响应

---

## 四种模式对比总览

| 模式 | 消息流向 | 并发 | Agent 类型 | 核心场景 |
|------|---------|------|------------|---------|
| **SequentialPipeline** | A→B→C（链式，输出转输入） | 串行 | 任意 AgentBase | 流水线加工、链式推理 |
| **FanoutPipeline** | 1→[A,B,C]（广播，独立响应） | 并发（可选串行） | 任意 AgentBase | 多角度评估、并行专家组 |
| **MsgHub** | 任意方向自动广播（全连接） | 串行轮转 | 任意 AgentBase | 多 Agent 对话、辩论、圆桌协作 |
| **ChatRoom** | 前端↔Agent 事件双向路由 | 实时流 | RealtimeAgent | 实时语音对话房间 |

> **平台落地建议**：在 Pipeline BC 中，将这四种类型作为 `PipelineType` 枚举，让用户在控制台通过配置而非编码来选择编排模式，底层直接调用对应的 `agentscope.pipeline` 原语即可，无需重建任何调度引擎。