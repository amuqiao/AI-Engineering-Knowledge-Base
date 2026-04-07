# Mermaid 作图风格指南 · B 代码深潜层

> 适用场景：深入分析模块实现，理解数据持久化结构与代码对象设计。
> 包含图表：③ 数据模型关系图　④ 类层级关系图

---

## 一、两种图的区别与选用

| 维度 | 数据模型关系图 | 类层级关系图 |
|------|-------------|------------|
| **Mermaid 语法** | `flowchart TB` | `flowchart TB` |
| **回答的问题** | 数据如何持久化？表与表如何关联？ | 代码如何组织？抽象层次和设计模式是什么？ |
| **视角** | 静态数据视图 | 静态代码结构视图 |
| **节点代表** | 数据库表（含核心字段） | 类 / 接口 / 抽象基类 |
| **箭头代表** | 外键引用（实线）/ 逻辑关联（虚线） | 继承（实线）/ 实现（虚线）/ 组合 |
| **类比** | 户型图（房间内部陈设与连通） | 建筑构件施工图（材料规格与受力结构） |

```
需要理解"数据怎么存"           → 数据模型关系图
需要理解"代码怎么组织"         → 类层级关系图
```

**与 ER 图的区别**：ER 图强调字段类型和规范化；数据模型关系图强调**业务语义和跨表关系的设计意图**，适合团队沟通而非数据库文档。

**与流程图的区别**：流程图的节点是处理步骤；类层级图的节点是**类/接口**，关注设计意图和扩展机制，而非处理顺序。

---

## 二、数据模型关系图

### 2.1 适用场景

用于回答：这个业务域有哪些数据库表？表与表之间是外键强约束还是逻辑关联？哪些字段是跨表的关键指针？

理解核心业务的数据持久化结构、分析版本快照设计、梳理多租户/多模式数据隔离边界时使用。

### 2.2 完整参考原图

> 展示 Dify 应用管理域：核心应用层 → 传统模式 / 画布模式 → 执行层 → 对话层的完整表关系

```mermaid
flowchart TB
    classDef appStyle    fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef configStyle fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef wfStyle     fill:#7c3aed,stroke:#5b21b6,stroke-width:2px,color:#fff
    classDef runStyle    fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef chatStyle   fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef siteStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef agentStyle  fill:#ea580c,stroke:#7c2d12,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    subgraph CORE["核心应用层"]
        APP["apps 表<br>app_model_config_id ─→ 传统配置<br>workflow_id ─→ 画布版本"]:::appStyle
        SITE["sites 表<br>WebApp 站点配置<br>code / domain / token_strategy"]:::siteStyle
    end
    class CORE layerStyle

    subgraph TRAD["传统模式（Chat / Completion / Agent-Chat）"]
        AMC["app_model_configs 表<br>model / pre_prompt / agent_mode<br>file_upload / dataset_configs<br>发布即覆盖，无历史版本"]:::configStyle
    end
    class TRAD layerStyle

    subgraph CANVAS["画布模式（Workflow / Advanced-Chat / RAG-Pipeline）"]
        direction LR
        DRAFT["workflows 表<br>version = 'draft'<br>草稿行（唯一，持续更新）"]:::wfStyle
        PUB["workflows 表<br>version = str(datetime)<br>已发布快照（每次新建）"]:::wfStyle
    end
    class CANVAS layerStyle

    subgraph EXEC["执行层"]
        direction LR
        WFR["workflow_runs 表<br>graph 快照<br>inputs / outputs / status"]:::runStyle
        WNE["workflow_node_executions 表<br>每节点输入输出<br>node_type / status / metadata"]:::runStyle
    end
    class EXEC layerStyle

    subgraph DIALOG["对话层（聊天类型应用）"]
        direction LR
        CONV["conversations 表<br>app_model_config_id 快照<br>dialogue_count / status"]:::chatStyle
        MSG["messages 表<br>query / answer<br>workflow_run_id（Advanced Chat）"]:::chatStyle
        MAT["message_agent_thoughts 表<br>thought / tool / observation<br>Agent 推理步骤（仅 agent-chat）"]:::agentStyle
    end
    class DIALOG layerStyle

    APP --> SITE
    APP -->|"app_model_config_id"| AMC
    APP -->|"workflow_id（生产指针）"| PUB
    APP -.->|"draft 查询"| DRAFT
    DRAFT -.->|"发布时克隆"| PUB
    PUB -->|"执行时关联"| WFR
    WFR -->|"包含节点明细"| WNE
    CONV -->|"包含多条"| MSG
    MSG -->|"agent-chat 模式"| MAT
    MSG -.->|"advanced-chat 模式"| WFR
    APP -.->|"对话创建"| CONV

    NOTE["表关系要点<br>① App.workflow_id 是线上版本切换的唯一操作点<br>② WorkflowRun 内嵌 graph 快照，历史执行不受版本更新影响<br>③ Conversation.app_model_config_id 记录对话创建时的配置快照<br>④ Message.workflow_run_id 在 Advanced Chat 模式下关联执行记录<br>⑤ MessageAgentThought 专属于 agent-chat，记录每步推理"]:::noteStyle
    NOTE -.- APP

    %% 边索引：0-11，共 12 条
    linkStyle 0 stroke:#dc2626,stroke-width:2px
    linkStyle 1 stroke:#d97706,stroke-width:2.5px
    linkStyle 2 stroke:#7c3aed,stroke-width:2.5px
    linkStyle 3 stroke:#7c3aed,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 4 stroke:#7c3aed,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 5 stroke:#0891b2,stroke-width:2px
    linkStyle 6 stroke:#0891b2,stroke-width:2px
    linkStyle 7 stroke:#059669,stroke-width:2px
    linkStyle 8 stroke:#ea580c,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 9 stroke:#0891b2,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 10 stroke:#059669,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 11 stroke:#f59e0b,stroke-width:1px,stroke-dasharray:2 2
```

---

## 三、类层级关系图

### 3.1 适用场景

用于回答：这个模块 / 子域的代码是如何组织的？哪些是抽象接口，哪些是具体实现？继承和组合关系如何体现设计意图？

**三个视角及其适用目标**：

| 视角 | 核心轴 | 典型适用目标 |
|------|--------|------------|
| **A. 抽象层级视角** | 从 ABC/接口 → 能力类型分支 → 具体实现 | `core/model_runtime/`、`core/workflow/nodes/`、`core/tools/` |
| **B. 聚合建模视角** | 以聚合根为中心 → 实体 → 值对象 | 分析某个子域的领域对象职责（补充数据模型图的业务语义层） |
| **C. 分层依赖视角** | 按 DDD 四层分组 → 类跨层依赖关系 | 验证某个功能模块是否符合 Clean Architecture 约束 |

### 3.2 完整参考原图

> 展示 `core/workflow/nodes/` 工作流节点类型体系（视角 A：抽象层级视角）：Generic ABC 抽象基类层 → 按职责分组的具体节点层，同时展示 Mixin 组合模式。实际共 30+ 个节点类型，图中仅选取 7 个代表性节点，通过分组标注传达完整规模。

```mermaid
flowchart TB
    %% ── 配色主题：按层次职责区分 ────────────────────────────────────
    classDef abcStyle    fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef mixinStyle  fill:#7c3aed,stroke:#5b21b6,stroke-width:2px,color:#fff
    classDef simpleStyle fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef llmStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef ctStyle     fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    %% ── L1：抽象基类层 ───────────────────────────────────────────
    subgraph L1["L1 抽象基类（base/node.py）"]
        BASE["Node[NodeDataT]<br>«Generic ABC»<br>@abstractmethod _run() → NodeRunResult<br>run() 【模板方法：统一处理执行生命周期】"]:::abcStyle
    end
    class L1 layerStyle

    %% ── Mixin：能力扩展（独立于继承链，按需混入） ──────────────────
    MIXIN["LLMUsageTrackingMixin<br>«mixin»<br>_accumulate_usage(usage)<br>_merge_usage(current, new)"]:::mixinStyle

    %% ── L2：具体节点层（按职责分组，仅展示代表性节点） ──────────────
    subgraph L2["L2 具体节点层（实际 30+ 个节点类型，__init_subclass__ 自动注册到全局 registry）"]
        direction LR

        subgraph G1["流程控制节点（单继承 Node）"]
            direction LR
            START["StartNode<br>«concrete»<br>node_type = START<br>_run() → 初始化变量"]:::simpleStyle
            IFELSE["IfElseNode<br>«concrete»<br>node_type = IF_ELSE<br>_run() → 条件路由"]:::simpleStyle
            END_N["EndNode<br>«concrete»<br>node_type = END<br>_run() → 输出结果"]:::simpleStyle
        end
        class G1 layerStyle

        subgraph G2["LLM 能力节点（多继承：Node + Mixin）"]
            direction LR
            LLM["LLMNode<br>«concrete»<br>node_type = LLM<br>_run() → 调用 LLM · 追踪用量"]:::llmStyle
            KR["KnowledgeRetrievalNode<br>«concrete»<br>node_type = KNOWLEDGE_RETRIEVAL<br>_run() → 向量检索 · 追踪用量"]:::llmStyle
        end
        class G2 layerStyle

        subgraph G3["容器节点（多继承：Node + Mixin）"]
            direction LR
            ITER["IterationNode<br>«concrete»<br>node_type = ITERATION<br>_run() → 遍历子图执行"]:::ctStyle
            LOOP["LoopNode<br>«concrete»<br>node_type = LOOP<br>_run() → 循环子图执行"]:::ctStyle
        end
        class G3 layerStyle
    end
    class L2 layerStyle

    %% ── L1 → L2：继承关系（所有节点均继承 Node） ────────────────────
    BASE -->|"继承"| START
    BASE -->|"继承"| IFELSE
    BASE -->|"继承"| END_N
    BASE -->|"继承"| LLM
    BASE -->|"继承"| KR
    BASE -->|"继承"| ITER
    BASE -->|"继承"| LOOP

    %% ── Mixin → L2：混入关系（仅 G2/G3 节点按需使用） ────────────
    MIXIN -.->|"混入"| LLM
    MIXIN -.->|"混入"| KR
    MIXIN -.->|"混入"| ITER
    MIXIN -.->|"混入"| LOOP

    %% ── 设计注记 ─────────────────────────────────────────────────
    NOTE["类设计要点<br>① Node[NodeDataT] 是 Generic ABC，子类必须实现 @abstractmethod _run()（模板方法契约）<br>② run() 是模板方法，统一处理 started/succeeded/failed 事件，子类只需关注业务逻辑<br>③ LLMUsageTrackingMixin 按需混入，只在需要追踪 LLM 用量的节点使用（多继承组合能力）<br>④ node_type ClassVar 声明后，__init_subclass__ 自动注册到全局 NODE_TYPE_CLASSES_MAPPING<br>⑤ 新增节点类型：继承 Node[T] 并声明 node_type，无需手动维护映射表，自动注册"]:::noteStyle
    NOTE -.- BASE

    %% 边索引：0-11，共 12 条
    linkStyle 0,1,2,3,4,5,6 stroke:#dc2626,stroke-width:2px
    linkStyle 7,8,9,10      stroke:#7c3aed,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 11            stroke:#f59e0b,stroke-width:1px,stroke-dasharray:2 2
```

---

## 四、最佳实践速查

### 通用规则（两种图共用）

| 设计原则 | 说明 |
|----------|------|
| **`linkStyle` 索引精准计数** | `linkStyle N` 按边的**声明顺序**从 0 开始编号，索引越界会触发渲染崩溃。两条规避守则：① **展开 `&`**：`A & B --> C` 会展开为多条独立边，凡使用 `linkStyle` 的图一律拆成独立行；② **注释标注边总数**：在连接线声明结束后、`linkStyle` 之前插入 `%% 边索引：0-N，共 X 条` 注释强制核对 |
| **连接线语义** | `-->` 实线表示强关联（外键约束引用 / 继承关系）；`-.->` 虚线表示弱关联（逻辑关联无 FK / 接口实现 / Mixin 混入）；连接线标签简明描述关系语义 |
| **subgraph 分组** | 按**业务领域或抽象层次**分组，用 `class SubgraphName layerStyle` 统一背景色；同一 subgraph 内关系密切的节点用 `direction LR` 横排，跨 subgraph 用纵向主流 `TB` |
| **节点换行** | 换行用 `<br>`；首行写主标识（表名或类名），后续行补充核心字段或关键方法签名 |
| **NOTE 注记** | 用 `NOTE -.- 核心节点` 悬浮挂载说明；用 `①②③④⑤` 序号逐条说明设计意图 |

### 数据模型关系图专用

| 设计原则 | 说明 |
|----------|------|
| **节点内容** | 节点首行写**表名**，`<br>` 后列出 2-4 个**核心字段或外键字段**，末行可附加一句关键业务约束说明（如 `"发布即覆盖，无历史版本"`） |
| **subgraph 分组依据** | 按**业务领域**（核心层、执行层、对话层等）分组，而非处理阶段 |
| **箭头颜色含义** | 用 `linkStyle` 为不同业务线的关联边着色：同一业务主线用同一颜色（如工作流相关边统一用紫色 `#7c3aed`，对话相关边用绿色 `#059669`），帮助读者快速跟踪某条业务线的完整关系链 |

### 类层级关系图专用

| 设计原则 | 说明 |
|----------|------|
| **节点内容** | 节点首行写**类名**，第二行写 `«stereotype»`（`«abstract»` / `«concrete»` / `«interface»` / `«mixin»`），第三行列出 1-2 个最能体现设计意图的关键方法签名；不罗列全部字段，只展示设计骨架 |
| **分层分组** | 按**抽象层次**分组：`subgraph L1` 放抽象基类/接口，`subgraph L2` 放能力分支层，`subgraph L3` 放具体实现层；整体呈现"从顶层契约到底层实现"的层次结构 |
| **配色语义** | 抽象基类/ABC 用红色（`#dc2626`）表示"必须被实现的契约"；中间抽象/能力分支用琥珀（`#d97706`）；具体实现类用青蓝（`#0891b2`）；接口/Protocol/Mixin 用紫色（`#7c3aed`） |
| **关系箭头** | 继承用实线 `-->` 加标签 `"继承"`；接口实现用虚线 `-.->` 加标签 `"实现"`；Mixin 混入用虚线加标签 `"混入"`；方向统一从父类指向子类（抽象在上，具体在下） |
