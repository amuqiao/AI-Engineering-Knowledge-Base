# Dify 后端架构实战指南：`core/` 目录深度解析与新功能开发指导

> 本文档解决三个核心问题：
> 1. Mermaid 架构图中的四层 DDD 如何映射到真实目录结构？
> 2. `core/` 下这么多目录，哪些是"领域"、哪些不是，区别是什么？
> 3. 新增功能、新增领域时，代码应该放在哪里、怎么写？

---

## 一、Mermaid 图与真实目录的完整映射

架构图描述了四层，每一层对应真实的文件夹：

```
架构图层级                  真实目录
─────────────────────────────────────────────────────────────
① 接口层 (Interface)   →   api/controllers/
② 应用层 (Application) →   api/services/  +  api/tasks/
③ 领域层 (Domain)      →   api/core/         ← 重点讨论
④ 基础设施层 (Infra)   →   api/models/  +  api/extensions/  +  api/repositories/
```

### 每层的职责一句话总结

| 层 | 目录 | 职责 | 举例 |
|---|---|---|---|
| ① 接口层 | `controllers/` | 解析 HTTP 请求、鉴权、序列化响应，**不含业务逻辑** | `controllers/console/app/workflow.py` → 解析 POST body，调用 service，返回 JSON |
| ② 应用层 | `services/` `tasks/` | **编排**多个领域服务完成一个业务用例，协调事务、触发异步任务 | `services/workflow_service.py` → 调用 `WorkflowEntry`、触发事件、写数据库 |
| ③ 领域层 | `core/` | **业务规则的实现**：图遍历、向量检索、模型调用、Prompt 构建等 | `core/workflow/graph_engine/` → 图遍历执行逻辑 |
| ④ 基础设施层 | `models/` `extensions/` `repositories/` | 技术实现：ORM 映射、Redis、向量库、对象存储 | `models/workflow.py` → SQLAlchemy ORM 模型 |

### 一次"发布工作流"请求的完整穿越路径

```
HTTP POST /apps/{id}/workflows/publish
        ↓
① controllers/console/app/workflow.py      # 解析请求、权限校验
        ↓
② services/workflow_service.py             # 读草稿、图结构校验、创建快照、切换指针、触发事件
        ↓
③ core/workflow/workflow_entry.py          # 图初始化领域服务
   core/workflow/graph_engine/             # 图遍历领域服务
   core/workflow/nodes/                    # 各节点业务规则
        ↓
④ models/workflow.py                       # ORM：Workflow 表读写
   extensions/ext_database.py             # SQLAlchemy session
```

**关键洞察**：`services/workflow_service.py` 大量 `from core.workflow.xxx import xxx`，说明应用层依赖领域层；而 `core/` 里的代码**从不** `import services`，这就是架构图里"禁止反向依赖"的含义。

---

## 二、`core/` 目录内部三类角色

`core/` 不是纯粹按 DDD 子域划分的——它实际上混合了三种性质的代码：

### 类型 A：真正的领域子域（Bounded Context）

**判断标准**：
- 有**业务不变式**（Invariant）：某些状态不允许出现，由代码强制保证
- 有**生命周期**：对象要被创建、修改、持久化、删除
- 有**自己的实体/值对象**（`entities/` 子目录）
- 可以**独立演化**：换掉这个子域，其他子域不需要大改

| 目录 | 对应子域 | 核心业务规则举例 |
|---|---|---|
| `core/app/` | App 子域 | 不同模式（Chat/Agent/Workflow）的推理编排逻辑 |
| `core/workflow/` | Workflow 子域 | 图遍历规则、节点依赖关系、变量池一致性 |
| `core/rag/` | RAG 子域 | 索引管道流程、分块策略、检索后处理规则 |
| `core/model_runtime/` | Model 子域 | 统一模型接口抽象、各供应商协议适配 |
| `core/agent/` | Agent 子域 | ReAct/Function Call 策略、思考链输出解析 |
| `core/tools/` | Tools 子域 | 工具抽象基类、工具调用协议、参数校验规则 |

**代码证据**（`core/workflow/runtime/graph_runtime_state.py`）：

```python
# 这是业务不变式：token 数不能为负
def add_tokens(self, tokens: int) -> None:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")   # 主动守护规则
    self._total_tokens += tokens

# 这是生命周期管理：状态可以序列化/反序列化（跨请求持久化）
def dumps(self) -> str: ...
@classmethod
def from_snapshot(cls, data: str) -> GraphRuntimeState: ...
```

---

### 类型 B：领域支撑服务（Domain Support Service）

**判断标准**：
- 有一定业务语义（不是纯工具函数），但**不守护业务规则**
- **无独立生命周期**：执行完就丢弃，不需要持久化自身状态
- 被多个子域复用，是"共享能力"而非"独立边界"
- 你很难为它识别出一个"聚合根"

| 目录 | 性质 | 说明 |
|---|---|---|
| `core/prompt/` | 支撑服务 | Prompt 模板组装/转换，被 app、agent 子域共用 |
| `core/memory/` | 支撑服务 | 对话历史 token 截断，支撑 Chat/Agent 子域 |
| `core/variables/` | 支撑服务 | 变量类型系统，workflow 子域内部使用 |
| `core/file/` | 共享值对象 | `File` 是跨子域传递的值对象，无独立生命周期 |
| `core/moderation/` | 支撑服务 | 内容审核策略（关键词/API/OpenAI），被 app 子域调用 |
| `core/plugin/` | 支撑服务 | 插件调用协议和反向调用通道 |
| `core/mcp/` | 支撑服务 | MCP 协议客户端/服务端实现 |
| `core/datasource/` | 支撑服务 | 外部数据源抓取（网页/文件/云盘） |

**代码证据**（`core/prompt/prompt_transform.py`）：

```python
class PromptTransform:
    # 注意：没有任何成员变量 ← 无状态
    def _append_chat_histories(
        self,
        memory: TokenBufferMemory,      # 上下文从外部传入
        memory_config: MemoryConfig,    # 不守护任何规则
        prompt_messages: list[PromptMessage],
        model_config: ModelConfigWithCredentialsEntity,
    ) -> list[PromptMessage]:           # 进来数据，出去数据
        ...
```

---

### 类型 C：共享内核与基础设施混入（本不应在此）

**判断标准**：
- 纯技术工具、无业务语义
- 在严格 DDD 中应归属基础设施层或 `shared_kernel/`

| 目录 | 实际性质 | 理想归属 |
|---|---|---|
| `core/entities/` | 跨子域共享枚举/实体 | `shared_kernel/` |
| `core/errors/` | 通用错误类型定义 | `shared_kernel/` |
| `core/schemas/` | JSON Schema 定义 | `shared_kernel/` |
| `core/helper/` | 工具函数（加密/HTTP/SSRF） | `libs/` 或基础设施层 |
| `core/base/` | 抽象基类（TTS 等） | 基础设施层 |
| `core/logging/` | 日志配置 | 基础设施层 |
| `core/db/` | 数据库工具函数 | 基础设施层 |
| `core/repositories/` | 仓储实现 | **基础设施层**（已经有 `repositories/` 顶级目录） |
| `core/ops/` | 可观测性追踪（Langfuse 等） | 横切关注点 |

> **为什么这些出现在 `core/` 里？**
> Dify 是实用主义项目，`core/` 在历史演化中成了"非接口层、非服务层的所有代码的容器"，而不是纯粹的"领域层"。`core/repositories/` 的存在是最明显的证据——仓储本质上是基础设施，但出于便利放在了 `core/` 里。

---

## 三、快速判断：新代码放哪里？

```
新代码要实现什么？
│
├─ 解析 HTTP 请求 / 返回响应 / 鉴权
│   └── → controllers/
│
├─ 编排多个步骤完成一个业务用例（调用多个领域服务）
│   └── → services/
│
├─ 异步执行某个用例
│   └── → tasks/
│
├─ 存取数据库的 ORM 模型
│   └── → models/
│
├─ Redis / 对象存储 / 向量库的具体实现
│   └── → extensions/ 或 repositories/
│
└─ 业务规则的实现（以下任一条件满足）
    ├─ 有不变式需要守护（某些状态不允许出现）
    ├─ 有独立生命周期（需要序列化/反序列化/持久化）
    ├─ 有领域特定的实体/值对象
    └── → core/
         │
         ├─ 属于现有子域（workflow/rag/model_runtime/app/agent/tools）？
         │   └── → 放入对应子域目录
         │
         ├─ 是被多个子域复用的"能力"（无聚合根）？
         │   └── → 放入 core/ 下新的支撑服务目录
         │
         └─ 是全新的、独立的业务子域？
             └── → 在 core/ 下新建子域目录（见第四节）
```

---

## 四、实战：如何新增功能和新领域

### 场景一：在现有子域内增加功能

**示例**：给 Workflow 子域新增一种节点类型 `SentimentNode`（情感分析节点）

```
api/core/workflow/nodes/
├── sentiment/                    ← 新建目录
│   ├── __init__.py
│   ├── entities.py               ← 节点的输入/输出 Pydantic 模型
│   ├── sentiment_node.py         ← 节点领域对象（继承 BaseNode）
│   └── exc.py                    ← 节点专有异常
└── node_mapping.py               ← 在这里注册新节点类型
```

节点实现模板：

```python
# core/workflow/nodes/sentiment/sentiment_node.py
from core.workflow.nodes.base.node import BaseNode
from core.workflow.node_events import NodeRunResult
from .entities import SentimentNodeData

class SentimentNode(BaseNode[SentimentNodeData]):
    """
    情感分析节点：对输入文本执行情感分析，输出 positive/negative/neutral。

    不变式：
    - 输入文本不能为空（由 run() 方法入口校验）
    - 输出 score 范围 [0.0, 1.0]（由 _analyze() 方法保证）
    """
    node_type = NodeType.SENTIMENT   # 先在 NodeType 枚举中注册

    def _run(self) -> NodeRunResult:
        # 业务规则在这里实现，不访问 HTTP/数据库（交给基础设施）
        ...
```

---

### 场景二：在现有子域内增加支撑能力（无需聚合根）

**示例**：新增一个"Prompt 压缩器"，被 app 和 agent 子域共用

```
api/core/prompt/
├── compressor.py         ← 新建，无状态的转换服务
└── entities/
    └── compress_config.py  ← 压缩配置的 Pydantic 模型
```

```python
# core/prompt/compressor.py
class PromptCompressor:
    """
    对超长 Prompt 进行语义压缩，使其适配模型 context window。
    无状态：每次调用独立，不持久化任何数据。
    """
    def compress(
        self,
        messages: list[PromptMessage],
        max_tokens: int,
        model_instance: ModelInstance,
    ) -> list[PromptMessage]:
        # 纯转换逻辑，没有成员变量，没有不变式守护
        ...
```

---

### 场景三：新增全新的业务子域

**示例**：新增"知识图谱"子域（GraphKB），独立于现有的 RAG 子域

#### 第一步：在 `core/` 下建立子域目录骨架

```
api/core/graph_kb/               ← 新子域根目录
├── __init__.py
├── entities/
│   ├── __init__.py
│   ├── graph_entities.py        ← 聚合根、实体、值对象的 Pydantic/dataclass 定义
│   └── query_entities.py
├── errors.py                    ← 子域专有异常（继承自 core/errors/）
├── graph_kb_service.py          ← 领域服务（核心业务逻辑入口）
├── query_engine.py              ← 图查询引擎（领域服务）
└── repositories/
    └── graph_repository.py      ← 仓储接口定义（抽象，实现在基础设施层）
```

#### 第二步：定义聚合根（`entities/graph_entities.py`）

```python
from pydantic import BaseModel, field_validator
from datetime import datetime

class KnowledgeGraph(BaseModel):
    """
    知识图谱聚合根。
    
    不变式：
    - node_count 始终等于 nodes 列表长度（由 add_node/remove_node 方法保证）
    - 图不允许存在孤立边（add_edge 前必须校验两端节点存在）
    """
    id: str
    tenant_id: str
    name: str
    nodes: list["GraphNode"] = []
    edges: list["GraphEdge"] = []

    def add_node(self, node: "GraphNode") -> None:
        # 守护不变式：节点 ID 唯一
        if any(n.id == node.id for n in self.nodes):
            raise DuplicateNodeError(f"Node {node.id} already exists")
        self.nodes.append(node)

    def add_edge(self, edge: "GraphEdge") -> None:
        # 守护不变式：边的两端节点必须存在
        node_ids = {n.id for n in self.nodes}
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            raise OrphanEdgeError("Edge endpoints must exist in graph")
        self.edges.append(edge)
```

#### 第三步：定义领域服务（`graph_kb_service.py`）

```python
# core/graph_kb/graph_kb_service.py
# 领域服务：只含业务逻辑，不直接访问数据库（通过仓储接口）

class GraphKBService:
    def __init__(self, graph_repo: GraphRepository) -> None:
        # 依赖倒置：依赖抽象接口，不依赖具体实现
        self._repo = graph_repo

    def build_graph_from_documents(
        self,
        tenant_id: str,
        documents: list[Document],
    ) -> KnowledgeGraph:
        """从文档列表构建知识图谱，提取实体和关系。"""
        graph = KnowledgeGraph(id=str(uuid4()), tenant_id=tenant_id, name="...")
        # 业务规则在这里...
        return graph
```

#### 第四步：在应用层（`services/`）新建用例协调器

```python
# services/graph_kb_service.py  ← 注意：这是应用层的 service
# 职责：协调领域服务、写数据库、触发事件

from core.graph_kb.graph_kb_service import GraphKBService as GraphKBDomainService

class GraphKBAppService:
    def create_knowledge_graph(
        self,
        tenant_id: str,
        app_id: str,
        document_ids: list[str],
    ) -> dict:
        # 1. 查询文档（基础设施）
        documents = self._load_documents(document_ids)
        # 2. 调用领域服务（核心逻辑）
        graph = GraphKBDomainService(repo=...).build_graph_from_documents(tenant_id, documents)
        # 3. 持久化（基础设施）
        self._save_graph(graph)
        # 4. 触发事件（副作用）
        graph_created.send(graph)
        return {"graph_id": graph.id}
```

#### 第五步：在接口层（`controllers/`）新建 API

```python
# controllers/console/app/graph_kb.py  ← 接口层
# 职责：HTTP 解析、鉴权、序列化，不含业务逻辑

class GraphKBApi(Resource):
    @login_required
    def post(self, app_id: str):
        parser = reqparse.RequestParser()
        parser.add_argument("document_ids", type=list, required=True)
        args = parser.parse_args()
        
        # 调用应用层（不直接调 core/）
        result = GraphKBAppService().create_knowledge_graph(
            tenant_id=current_user.current_tenant_id,
            app_id=app_id,
            document_ids=args["document_ids"],
        )
        return result, 201
```

---

## 五、三层对照总结表

| | 接口层 `controllers/` | 应用层 `services/` | 领域层 `core/` | 基础设施层 `models/` |
|---|---|---|---|---|
| **回答的问题** | "怎么接收请求？" | "这个用例怎么完成？" | "这个业务规则是什么？" | "数据怎么存取？" |
| **能否有业务规则** | ❌ | ❌（编排规则可以） | ✅ | ❌ |
| **能否直接访问数据库** | ❌ | ✅（通过 ORM/仓储） | ❌（通过仓储接口） | ✅ |
| **能否依赖 `core/`** | ✅ | ✅ | ✅（只能依赖自身和下层） | ✅ |
| **能否被 `core/` 依赖** | ❌（CI 强制拦截） | ❌（CI 强制拦截） | — | ✅ |
| **典型文件** | `workflow.py` controller | `workflow_service.py` | `graph_engine.py` | `models/workflow.py` |

---

## 六、最常见的错误放置模式

### ❌ 错误：业务规则写进 controller

```python
# controllers/console/app/workflow.py  ← 错误
class WorkflowPublishApi(Resource):
    def post(self, app_id):
        # ❌ 业务规则不应在接口层
        if workflow.node_count > 100:
            return {"error": "Too many nodes"}, 400
```

应该移入领域服务或应用层进行校验。

### ❌ 错误：直接在 `core/` 里查询数据库

```python
# core/workflow/graph_engine/graph_engine.py  ← 错误
from extensions.ext_database import db
from models.workflow import WorkflowRun

class GraphEngine:
    def execute(self):
        # ❌ 领域层不应直接访问 ORM
        run = db.session.query(WorkflowRun).filter_by(...).first()
```

应通过仓储接口（`repositories/`）访问，领域层只依赖抽象。

### ❌ 错误：在 `core/` 里 import `services/`

```python
# core/workflow/workflow_entry.py  ← 错误
from services.workflow_service import WorkflowService  # ❌ 反向依赖
```

这会触发 `import-linter` CI 报错。领域层永远不依赖应用层。

---

> **一句话记忆口诀**：
> `controllers/` 接请求 → `services/` 定剧本 → `core/` 讲规则 → `models/` 存数据
> 方向只有从左到右，从不反向。
