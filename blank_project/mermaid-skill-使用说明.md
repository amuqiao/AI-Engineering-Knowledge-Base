# Mermaid Skill 使用说明

本文档说明如何使用已经整理好的 `mermaid-diagram-guides` skill。

先说结论：**你的提示词完全可以用中文。**  
不需要因为这份 skill 就改成英文。真正需要保持原样的，通常只有 skill 名称 `$mermaid-diagram-guides`，因为它是调用标识，不是提示词内容本身。

## 1. 当前状态

这份 skill 现在有两份：

- 项目内副本：`E:\github_project\blank_project\skills\mermaid-diagram-guides`
- 全局安装目录：`C:\Users\97821\.codex\skills\mermaid-diagram-guides`

因为已经安装到全局目录，后续通常可以直接写：

```text
使用 $mermaid-diagram-guides ...
```

一般不需要再写完整路径。

## 2. 这份 skill 是做什么的

这份 skill 的作用不是固定生成某一种 Mermaid 图，而是：

1. 先判断你的需求属于哪一类图。
2. 再加载对应的参考规范。
3. 按参考规范生成 Mermaid 代码。
4. 避免不同图类的 Mermaid 语法混用。

它适合这些场景：

- 画系统架构图、业务流程图
- 梳理表关系、类继承关系、模块结构
- 分析状态机、时序图、异步链路
- 生成甘特图、时间线、思维导图

## 3. 最推荐的调用方式

### 方式一：直接使用全局 skill

现在最推荐这样写：

```text
使用 $mermaid-diagram-guides，根据下面的说明生成 Mermaid 图。
```

或者更完整一点：

```text
使用 $mermaid-diagram-guides，根据下面的系统说明生成 Mermaid 架构图，并遵循内置参考规范。
```

### 方式二：显式指定 skill 路径

如果你希望强制使用项目里的那一份 skill，也可以这样写：

```text
使用 $mermaid-diagram-guides，技能路径为 E:\github_project\blank_project\skills\mermaid-diagram-guides，请根据下面的说明生成 Mermaid 图。
```

这种写法适合：

- 你正在修改项目内 skill，还没同步到全局
- 你希望固定使用某个版本
- 你想测试某个 skill 副本

## 4. 提示词到底能不能用中文

可以，而且**应该优先用你最自然的中文表达**。

更准确地说：

- skill 名：建议保持 `$mermaid-diagram-guides`
- 任务描述：完全可以写中文
- 图的要求：完全可以写中文
- 补充约束：完全可以写中文

也就是说，你可以这样写：

```text
使用 $mermaid-diagram-guides，把下面的系统说明整理成 Mermaid 架构图，突出前端、网关、服务层、缓存、消息队列和数据库。
```

这就是标准、自然、可用的中文提示词。

## 5. 如何选择 A / B / C / D 四类参考

这份 skill 内部已经整理了 4 组参考规范，你只需要按任务类型表达需求即可。

### A 类：系统认知层

适合：

- 系统架构图
- 端到端业务流程图
- 服务分层关系图

你可以这样提：

- “帮我画这个系统的 Mermaid 架构图”
- “把这个业务流程整理成 Mermaid 流程图”
- “梳理这个项目的整体结构”

### B 类：代码深潜层

适合：

- 数据模型关系图
- 表结构关系图
- 类图、继承图、Mixin 关系图
- 模块内部设计分析

你可以这样提：

- “帮我画这个模块的类层级图”
- “把这些表关系整理成 Mermaid 图”
- “分析这个模块的对象关系”

### C 类：运行时行为层

适合：

- 状态机图
- 时序图
- Celery / SSE / WebHook / 异步任务链路

你可以这样提：

- “画出这个任务的状态机图”
- “把 SSE 推送过程画成 Mermaid 时序图”
- “分析 Celery 异步链路”

### D 类：规划可视化层

适合：

- 甘特图
- 时间线
- 思维导图
- 路线图、演进图、概念分解图

你可以这样提：

- “生成项目路线图甘特图”
- “整理一条技术演进时间线”
- “把这个系统画成 Mermaid 思维导图”

## 6. 可直接复制的中文使用示例

下面这些例子都可以直接拿去用。

### 示例 1：系统架构图

```text
使用 $mermaid-diagram-guides，根据下面的系统说明生成 Mermaid 架构图，并遵循内置参考规范。系统包含 Web 前端、API Gateway、认证服务、订单服务、用户服务、通知服务、Redis、MySQL 和 Kafka。请用分层结构展示，并在图后补充 3 条关键设计说明。
```

### 示例 2：业务流程图

```text
使用 $mermaid-diagram-guides，把下面的业务链路整理成 Mermaid 流程图：用户提交订单 -> 风控校验 -> 库存锁定 -> 支付 -> 通知发货。要求突出主流程和关键节点。
```

### 示例 3：表关系图

```text
使用 $mermaid-diagram-guides，根据下面的表结构生成 Mermaid 数据模型关系图：users、orders、order_items、payments。请突出主键、外键和核心关联关系。
```

### 示例 4：类层级图

```text
使用 $mermaid-diagram-guides，把下面这个模块整理成 Mermaid 类层级图，要求包含抽象基类、具体实现类、Mixin 以及它们之间的关系。
```

### 示例 5：状态机图

```text
使用 $mermaid-diagram-guides，把下面的生命周期整理成 Mermaid 状态机图：pending -> running -> completed，失败时进入 failed，支持从 failed 重试回 running。
```

### 示例 6：时序图

```text
使用 $mermaid-diagram-guides，根据下面的说明生成 Mermaid 时序图，并突出同步调用、异步消息和 SSE 回传。流程为：前端发起聊天请求，API 创建任务并入队，Celery Worker 调用 LLM 流式生成 token，Redis PubSub 转发消息，API 通过 SSE 返回前端。
```

### 示例 7：甘特图

```text
使用 $mermaid-diagram-guides，生成一个 3 阶段 MVP 路线图甘特图，覆盖基础设施、认证、运行时、聊天能力和测试收尾。
```

### 示例 8：时间线

```text
使用 $mermaid-diagram-guides，整理一条从原型阶段到生产阶段的 Agent 平台演进时间线，每个阶段列出关键里程碑。
```

### 示例 9：思维导图

```text
使用 $mermaid-diagram-guides，把这个 AI Agent 平台的技术栈整理成 Mermaid 思维导图，分成接入层、核心运行时、数据层、前端层和扩展能力。
```

## 7. 中文提示词怎么写更好

推荐你按这个结构组织提示词：

```text
使用 $mermaid-diagram-guides，
根据下面的内容生成 Mermaid【图类型】，
重点展示【你最关心的部分】，
并遵循内置参考规范。
如果信息不完整，可以做合理假设。
```

例如：

```text
使用 $mermaid-diagram-guides，根据下面的说明生成 Mermaid 架构图，重点展示系统分层、核心服务关系和数据流向，并遵循内置参考规范。如果信息不完整，可以做合理假设。
```

## 8. 使用时的注意事项

### 不要混用图类语法

虽然这份 skill 已经处理了图类边界，但你的需求里最好也尽量说清楚你要的是什么图：

- 架构图 / 流程图：偏 `flowchart`
- 状态图：偏 `stateDiagram-v2`
- 时序图：偏 `sequenceDiagram`
- 甘特图：偏 `gantt`
- 时间线：偏 `timeline`
- 思维导图：偏 `mindmap`

### 尽量说明“你想看什么”

比起只说“画个 Mermaid 图”，更好的说法是：

- “我想看系统分层”
- “我想看请求怎么流转”
- “我想看类之间怎么继承”
- “我想看异步链路谁调谁”
- “我想看项目排期”

这样 skill 更容易自动选对参考规范。

### 信息不完整时，可以明确允许合理假设

你可以直接这样写：

```text
如果部分信息不完整，请在保持结构合理的前提下做必要假设。
```

## 9. 一条完整中文示例

这是一个更完整的中文提示词示例：

```text
使用 $mermaid-diagram-guides，根据下面的系统说明生成 Mermaid 架构图，并遵循内置参考规范。系统包含 Web 前端、API Gateway、认证服务、订单服务、用户服务、通知服务、Redis、MySQL 和 Kafka。请用分层结构展示这些组件之间的关系，突出请求入口、核心服务协作、缓存和消息队列的作用，并在图后补充 3 条关键设计说明。如果部分信息不完整，请做合理假设。
```

## 10. 后续维护方式

如果你后面继续补充 Mermaid 规范，建议这样维护：

1. 原始规范文档继续放在项目里。
2. skill 的 `references/` 里同步复制一份。
3. 如果新增了新图类或新规则，再更新 `SKILL.md` 的路由说明。
4. 修改后运行校验。

Windows 下建议这样校验：

```powershell
$env:PYTHONUTF8='1'
python C:\Users\97821\.codex\skills\.system\skill-creator\scripts\quick_validate.py E:\github_project\blank_project\skills\mermaid-diagram-guides
```
