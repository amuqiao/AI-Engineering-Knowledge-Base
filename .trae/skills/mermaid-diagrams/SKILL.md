---
name: mermaid-diagrams
description: Generate Mermaid diagrams with a consistent, professional visual style. Use when the user asks to draw, create, or generate any diagram. Trigger words include: 架构图, 流程图, 时序图, 状态机, 甘特图, 时间线图, 思维导图, 数据模型图, 类图, 关系图, mermaid, 画图, 可视化, architecture diagram, flow chart, sequence diagram, state machine, Gantt chart, mind map, class diagram, ER diagram.
---

# Mermaid 作图规范

## 第一步：识别图表类型并读取对应参考文件

生成任何 Mermaid 图前，**先用 Read 工具读取对应参考文件**，严格遵循其配色、结构和语法规范。

| 意图 / 触发词 | 图表类型 | Mermaid 语法 | 读取文件 |
|-------------|---------|------------|---------|
| 系统架构、组件构成、部署结构 | 系统架构图 | `flowchart TB` | [layer-A.md](layer-A.md) |
| 业务流程、请求链路、数据流转 | 端到端流程图 | `flowchart LR` | [layer-A.md](layer-A.md) |
| 数据库表结构、表关联关系 | 数据模型关系图 | `flowchart TB` | [layer-B.md](layer-B.md) |
| 类继承、接口实现、代码组织 | 类层级关系图 | `flowchart TB` | [layer-B.md](layer-B.md) |
| 状态转换、实体生命周期、异步任务状态 | 状态机图 | `stateDiagram-v2` | [layer-C.md](layer-C.md) |
| 异步链路、跨进程消息、SSE / Celery | 时序图 | `sequenceDiagram` | [layer-C.md](layer-C.md) |
| 项目排期、任务依赖 | 甘特图 | `gantt` | [layer-D.md](layer-D.md) |
| 技术演进、历史里程碑 | 时间线图 | `timeline` | [layer-D.md](layer-D.md) |
| 技术栈总览、模块分类、概念归类 | 思维导图 | `mindmap` | [layer-D.md](layer-D.md) |

## 第二步：生成图表

参照读取的参考文件中的完整原图，按相同配色、结构和规范生成目标图。

## 核心约束（A/B 层 flowchart 图通用）

- **`linkStyle` 计数**：边按声明顺序从 0 编号；凡使用 `linkStyle` 一律将 `A & B --> C` 拆为独立行；在 `linkStyle` 前插入 `%% 边索引：0-N，共 X 条` 注释强制核对，索引越界会导致渲染崩溃
- **节点换行**：flowchart 用 `<br>`；mindmap 专用 `<br/>`
- **连接线语义**：`-->` 同步强关联；`-.->` 异步或弱关联；`==>` 关键强制路径
- **NOTE 注记**：用 `NOTE -.- 核心节点` 悬浮挂载，避免与主流程线混淆
- **语法隔离**：C/D 层图（`stateDiagram-v2`、`sequenceDiagram`、`gantt`、`timeline`、`mindmap`）**不支持** `classDef`、`linkStyle`、`subgraph` 等 flowchart 专属语法，不得混用
