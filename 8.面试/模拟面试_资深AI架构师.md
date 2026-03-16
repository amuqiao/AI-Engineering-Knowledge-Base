# 模拟面试文档：资深 AI 架构师（Agent & Infra）

> **岗位**：资深 AI 架构师（Agent & Infra）| 35-60K · 15薪  
> **岗位核心**：企业级 AI 业务中台全栈架构设计，覆盖底层 GPU 调度、Dify 深度定制、Multi-Agent 协同  
> **本文结构**：候选人自我介绍 → 面试官问题 + 参考回答 → FAQ 技术专题

---

## 一、候选人自我介绍

> **提示**：自我介绍控制在 3~5 分钟，突出与岗位的高匹配点，按「人 → 经历 → 技术 → 亮点 → 期望」结构展开。

---

面试官您好，非常感谢这次面试机会。我叫 **[姓名]**，拥有近 **10 年**的后端与 AI 工程化经验，持有 **CCIE 思科认证**，目前在深圳东方大唐信息技术担任 **AI 架构师**。

我的技术成长路径可以分为三个阶段：

**第一阶段（2016-2019）——高并发后端基础建设**  
在永杨安风科技，我主导了一款社交直播 APP 的后台系统研发。基于 Tornado 异步框架 + Nginx 负载均衡，构建了支撑高并发的后端架构，完成了从代码提交到 AWS 云端上线的全流程自动化部署。这段经历让我积累了扎实的工程基础和云原生视野。

**第二阶段（2019-2024）——NLP 与 AI 工程化深耕**  
在深圳申朴信息技术，我聚焦平安系金融 AI 解决方案，涵盖以下几个核心方向：
- 基于 **BERT 微调**构建意图识别与命名实体识别模块，准确率突破 90%；
- 构建保险产品**知识图谱**，50 万+ 三元组，F1 达 88-92%；
- 为平安证券搭建基于 **Kafka + InfluxDB + Grafana** 的实时运维监控平台，系统异常响应时间从小时级压缩到分钟级；
- 主导智能客服平台微服务架构设计，支持 12+ 类业务意图，人工客服工作量降低 60%。

**第三阶段（2024-至今）——AI 架构全栈落地**  
加入东方大唐后，我主导了三大方向的技术落地：
1. **智能体平台**：基于 **Dify + AgentScope** 构建企业级多 Agent 协作与运营平台，自主设计「八层架构」，构建 LLMGateway 实现鉴权、限流、计费与降级，K8s 容器化部署，平台服务 3 个业务部门，智能体开发效率提升 60%；
2. **时序预测系统**：基于 **TimesNet/Informer** 构建电站派单 AI 决策系统，AI 派单占比从 30% 跃升至 80%；
3. **计算机视觉检测**：基于 **YOLOv5 + CBAM 注意力机制**构建光伏缺陷检测系统，准确率 95%，检测效率提升 8 倍。

结合本次岗位，我认为自己在 **Dify 二次开发、Multi-Agent 架构、LLMGateway 设计、RAG 全链路优化、K8s 工程落地**等方向上与岗位高度匹配。我期望加入贵公司，参与更大规模的企业级 AI 中台建设，在 GPU 调度、国产化算力适配等方向进一步深耕。

以上是我的自我介绍，请面试官指导！

---

## 二、面试官提问 + 参考回答

### 模块一：Dify 二次开发专项

---

#### Q1：你在 Dify 二次开发方面做了哪些工作？具体深入到源码哪个层面？

**参考回答：**

我在 Dify 的二次开发主要分为四个层面：

**① 私有化部署（华为云适配）**  
- 将 Dify 完整部署到华为云 CCE（Cloud Container Engine），调整了 `docker-compose` 配置适配华为云的 OBS 对象存储、VPC 网络策略；
- 处理了华为云 ARM 架构与 Dify 部分依赖的兼容性问题，修改了部分基础镜像的构建脚本；

**② 插件开发（源码级扩展）**  
- 基于 Dify 的 `ToolProvider` 抽象类开发了多个自定义工具插件，主要在 `api/core/tools/` 模块下新增工具定义；
- 实现了与内部数据平台 API 对接的插件，封装了认证、重试、流式返回等逻辑；
- 对插件的 `credential_schema` 进行了扩展，支持多租户下的动态凭证注入；

**③ MCP（Model Context Protocol）集成**  
- 接入了 MCP Server，在 Dify 工作流中以工具调用的方式集成了外部上下文协议服务；
- 修改了工作流引擎中的工具调用链路，支持 SSE 流式返回的 MCP 工具结果回调；

**④ 工作流引擎性能优化**  
- 针对节点并行执行的场景，优化了工作流调度逻辑，减少了中间节点等待时间；
- 对 LLM 调用节点增加了 fallback 机制：当主模型超时，自动降级到备用模型；

**面试追问：Dify 工作流引擎底层是如何调度节点的？**

Dify 工作流基于有向无环图（DAG）调度。每个节点对应一个 `BaseNode` 子类，通过 `node_run` 方法执行，引擎通过拓扑排序确定执行顺序。并行节点通过 Celery 的 `group()` 异步任务组实现并发。我在做性能优化时，主要在节点就绪条件判断和 Redis 状态同步两个环节做了缓存优化。

---

#### Q2：你构建的 LLMGateway 是如何设计的？解决了哪些核心问题？

**参考回答：**

LLMGateway 是我们智能体平台的核心基础组件，设计目标是**统一管控平台内所有模型调用**。主要解决了以下问题：

**架构设计：**
```
客户端请求 → LLMGateway（FastAPI） → 路由层 → 模型适配器层 → 各模型提供商（Qwen/GPT/Claude/本地部署）
```

**核心功能模块：**

| 模块 | 实现方式 |
|------|----------|
| **鉴权** | JWT Token + API Key 双模式，支持多租户隔离 |
| **限流** | 基于 Redis 的滑动窗口算法，按用户/部门维度限制 QPM |
| **计费** | 拦截模型返回的 `usage` 字段，异步写入计费流水表 |
| **降级** | 主模型超时/异常时，按优先级切换备用模型，熔断器模式 |
| **流式代理** | 透传 SSE 流，保证客户端流式体验 |
| **可观测** | 接入 Prometheus 暴露调用量、延迟、错误率指标 |

**关键技术点：**
- 限流使用 Redis `ZADD + ZCOUNT` 实现滑动窗口，相比固定窗口更平滑；
- 计费采用异步写入（Celery 任务），避免影响主链路延迟；
- 降级策略通过配置中心动态更新，无需重启服务；

---

#### Q3：你提到 RAG 全链路优化，具体做了哪些优化？效果如何？

**参考回答：**

RAG 优化我从以下五个环节入手：

**① 文档解析与分块（Chunking）**  
- 针对不同文档类型（PDF、Word、表格）设计了差异化的分块策略；
- 采用语义分块替代固定窗口分块，保证每个 chunk 的语义完整性；
- 对表格类数据单独提取，转换为 Markdown 格式再向量化；

**② 向量数据库选型**  
- 最终选用 Milvus：支持 HNSW/IVF_FLAT 索引、支持标量过滤、支持多租户 Collection 隔离；
- 对比测试过 Qdrant、Chroma，Milvus 在百万级向量下检索延迟最稳定；

**③ 混合搜索（Hybrid Search）**  
- 结合稠密向量（Embedding）+ 稀疏向量（BM25）的混合检索；
- 使用 RRF（Reciprocal Rank Fusion）融合两路召回结果；
- 对时效性强的问题加权稀疏检索，对语义模糊的问题加权向量检索；

**④ 重排序（Reranking）**  
- 引入 Cross-Encoder 重排序模型（BGE-Reranker）对 Top-K 结果精排；
- 将召回 Top-20，重排后取 Top-5 送入 LLM；

**⑤ 查询改写（Query Rewriting）**  
- 对用户输入进行意图识别，对多意图问题拆分为子问题分别检索；
- HyDE（Hypothetical Document Embeddings）：先让 LLM 生成假设答案，用假设答案的向量检索更相关的文档；

**效果：**整体回答准确率从基线的 62% 提升到 85%，幻觉率下降约 40%。

---

### 模块二：Agent 架构专项

---

#### Q4：你设计的多 Agent 协作平台是什么架构？AgentScope 如何在其中发挥作用？

**参考回答：**

我设计了一个「八层架构」的智能体平台：

```
┌─────────────────────────────────────┐
│  前端交互层  (Open WebUI / 自研UI)    │
├─────────────────────────────────────┤
│  应用服务层  (FastAPI API Gateway)    │
├─────────────────────────────────────┤
│  智能体层   (Agent 实例池)            │
├─────────────────────────────────────┤
│  技能工具层  (Skills / Tools)         │
├─────────────────────────────────────┤
│  平台核心层  (AgentScope 核心框架)     │
├─────────────────────────────────────┤
│  模型网关层  (LLMGateway)             │
├─────────────────────────────────────┤
│  基础设施层  (K8s / MySQL / Redis)    │
├─────────────────────────────────────┤
│  部署运维层  (监控 / 弹性伸缩 / 多环境)│
└─────────────────────────────────────┘
```

**AgentScope 的核心作用：**
- **消息总线**：Agent 间通过 AgentScope 的 Msg 对象传递结构化消息，解耦了 Agent 间的直接依赖；
- **生命周期管理**：AgentScope 的 Pipeline 和 MsgHub 管理 Agent 的启动、执行、挂起、销毁；
- **多 Agent 协作模式**：支持顺序流（Sequential Pipeline）、并行流（Parallel Pipeline）、路由（Router）三种协作模式；
- **可视化调试**：AgentScope-Studio 提供工作流可视化，方便调试多 Agent 的消息流；

**面试追问：你如何处理多 Agent 协作中的任务分解和结果聚合？**

任务分解采用 Orchestrator Agent（主控智能体）模式：
1. Orchestrator 接收用户任务，通过 LLM 调用 Planning 模块将任务分解为子任务 DAG；
2. 子任务按依赖关系分发给对应的 Worker Agent；
3. 各 Worker Agent 独立执行并将结果写回消息总线；
4. Orchestrator 通过 Aggregator 模块收集结果，进行二次综合推理，生成最终输出；

---

#### Q5：你如何实现 Agent 的长期记忆（Memory）？

**参考回答：**

我将 Agent 的记忆分为三个层次：

**① 短期记忆（Short-term Memory）**  
- 即当前会话的上下文窗口，直接存储在内存中；
- 超出 context window 时，采用滑动窗口 + 摘要压缩策略（让 LLM 对历史对话生成摘要，替换掉早期的原始对话）；

**② 长期记忆（Long-term Memory）**  
- 将重要的用户偏好、历史结论、个性化设置存储到 Milvus 向量数据库；
- 每次对话开始时，检索相关的长期记忆注入 System Prompt；
- 使用 Memory Entity 结构化存储（用户名、时间戳、重要程度、向量 embedding）；

**③ 工作记忆（Working Memory）**  
- 任务执行过程中的中间结果，存储在 Redis（TTL 管理），任务完成后决定是否持久化；

---

#### Q6：ReAct 框架你是如何落地的？遇到了哪些工程问题？

**参考回答：**

ReAct（Reasoning + Acting）的核心循环是：`Thought → Action → Observation → Thought → ...`

**落地方式：**
- 通过 System Prompt 约束 LLM 输出严格的 Thought/Action/Observation 格式；
- 解析 LLM 输出，提取 Action 名称和参数，调用对应的工具函数；
- 将工具返回结果作为 Observation 拼接回对话历史，触发下一轮推理；

**遇到的工程问题：**

| 问题 | 解决方案 |
|------|----------|
| LLM 输出格式不稳定，解析失败 | 使用 JSON Schema 强制约束输出格式，加 retry 机制 |
| 无限循环（Agent 卡在某个工具调用） | 设置最大迭代步数（max_steps=10），超过则强制中断并输出当前结果 |
| 工具调用耗时导致用户体验差 | 流式返回 Thought 部分，工具调用异步执行，完成后推送 Observation |
| 上下文膨胀（Observation 过长） | 对工具返回结果做截断和摘要，保留核心信息 |

---

### 模块三：云原生与基础设施专项

---

#### Q7：你的 K8s 实践达到什么深度？有没有写过 CRD 或 Operator？

**参考回答：**

**已有实践：**
- 在智能体平台中，基于 K8s 实现了容器化部署、水平自动伸缩（HPA）、滚动更新；
- 编写了 Helm Chart 管理多环境（dev/staging/prod）部署配置；
- 配置了基于 Prometheus + AlertManager 的监控告警体系；
- 使用 K8s ConfigMap + Secret 管理应用配置和敏感信息；

**CRD/Operator 方面：**  
我目前在 CRD/Operator 方面的实践深度相对有限，尚未独立编写完整的 Operator。但我理解其核心原理：
- CRD（Custom Resource Definition）用于扩展 K8s API，定义自定义资源类型；
- Operator 通过 `controller-runtime` 框架实现 Reconcile 控制循环，监听资源变更并收敛到期望状态；
- 例如，训练任务 Operator 可以监听 `TrainingJob` CRD，自动创建 Pod、分配 GPU 资源、处理失败重试；

**我的提升计划：**  
基于 `kubebuilder` 或 `operator-sdk` 快速构建 Operator，结合岗位需求，我可以在入职后重点攻关 GPU 算力调度 Operator 的开发。

---

#### Q8：你在华为云部署 Dify 时遇到了哪些坑？如何解决的？

**参考回答：**

**坑一：ARM 架构兼容性**  
华为云部分节点使用鲲鹏（ARM）架构，Dify 的部分 Python 依赖（如 `tiktoken`）在 ARM 上需要从源码编译。解决方案是在 Dockerfile 中增加条件编译逻辑，或改用预编译好的 ARM 版本镜像。

**坑二：OBS 对象存储适配**  
Dify 默认使用 S3 接口，华为云 OBS 虽然兼容 S3，但部分签名算法有差异。需要在环境变量中设置 `S3_ENDPOINT_URL` 指向 OBS 的 endpoint，并处理 SSL 证书验证问题。

**坑三：网络策略限制**  
华为云 VPC 内的 Pod 访问外部 LLM API 需要配置 NAT 网关，初次部署时因为忽略出站规则导致 LLM 调用超时。解决方案是梳理出站流量需求，统一配置 NAT 网关和安全组。

**坑四：Celery Worker 内存泄漏**  
在华为云 ECS 上长时间运行 Celery Worker 时，发现内存持续增长。排查发现是 Worker 处理文档解析任务时，大对象未及时释放。解决方案是配置 `CELERYD_MAX_TASKS_PER_CHILD=100`，定期重启 Worker 进程。

---

### 模块四：模型与算法专项

---

#### Q9：你在 TimesNet 时序预测项目中，如何处理数据质量问题？

**参考回答：**

电站派单数据质量问题是项目最大的挑战之一，50 万+ 条历史数据中存在大量噪声：

**数据问题及处理方案：**

| 数据问题 | 处理方案 |
|----------|----------|
| 时间戳不连续（设备离线导致） | 用线性插值填充短期缺失，超过 6 小时缺失打标记排除 |
| 传感器异常值（电流突变） | 基于 IQR 方法识别异常值，用滑动中位数替换 |
| 多个电站数据分布差异大 | 对每个电站独立做 Z-score 标准化，避免跨站污染 |
| 天气数据与电站数据时间粒度不匹配 | 将小时级天气数据用线性插值对齐到 15 分钟粒度 |

**特征工程：**
- 时间特征：小时、星期、月份、是否节假日（三角编码）；
- 气象特征：辐照度、温度、风速、云量；
- 设备特征：组串电流历史均值、设备型号编码；
- 运维特征：上次维修距今时间、历史派单频率；

**模型选择：**
- 对比了 TimesNet、Informer、PatchTST，在我们的数据集（序列长度 96，预测步长 24）上 TimesNet 综合表现最优；
- TimesNet 的核心优势是通过 2D 变换将时序数据映射到二维空间，利用 CNN 提取局部周期特征；

---

#### Q10：YOLOv5 缺陷检测项目中引入 CBAM 注意力机制，原理是什么？效果如何？

**参考回答：**

**CBAM（Convolutional Block Attention Module）原理：**

CBAM 是一个轻量级注意力模块，包含两个子模块串联：

1. **通道注意力（Channel Attention）**：对每个特征图通道计算全局平均池化和最大池化，通过 MLP 生成通道权重，增强有用特征通道、抑制无关通道；

2. **空间注意力（Spatial Attention）**：对通道维度做平均池化和最大池化，拼接后通过卷积生成空间权重图，让模型聚焦于关键空间区域；

**引入原因：**  
光伏组件的隐裂缺陷面积极小（有时只有几个像素宽），普通 YOLOv5 的特征提取对细小纹理不够敏感。CBAM 的空间注意力能让骨干网络更关注隐裂所在的局部区域。

**工程实现：**  
在 YOLOv5 的骨干网络（CSP Bottleneck）中插入 CBAM 模块，增加了约 0.3% 的参数量，推理速度几乎不受影响。

**效果：**
- 隐裂检测 AP 从 0.78 提升到 0.87；
- 总体 mAP@0.5 从 0.91 提升到 0.95；
- 小目标漏检率下降约 30%；

---

#### Q11：你用 BERT 做意图识别，为什么准确率能达到 90%？调优过程是什么？

**参考回答：**

从初版的 72% 提升到 90%，经历了以下几个关键调优步骤：

**① 数据质量提升**  
- 初版数据集只有 2000 条，通过数据增强（同义词替换、回译、EDA）扩充到 8000 条；
- 对边界样本（模糊意图）进行人工二次标注，统一标注规范；

**② 模型选型**  
- 从通用 BERT-base-Chinese 切换到金融领域预训练模型（FinBERT），领域适应性提升明显；
- 实验对比了 ALBERT、RoBERTa，最终 FinBERT 在保险场景表现最佳；

**③ 训练策略优化**  
- 分层学习率：底层 Transformer 层 lr=1e-5，分类头 lr=1e-4；
- 标签平滑（Label Smoothing）处理类别不均衡；
- 使用 Focal Loss 重点关注难分类样本；

**④ 后处理**  
- 对置信度低于阈值的样本触发「澄清对话」，让用户补充信息，避免错误分类；
- 对高频混淆意图对（如「保单查询」vs「保单变更」）增加判别式子分类器；

---

### 模块五：综合场景题

---

#### Q12：如果让你从零搭建一个企业级 AI 中台，你的整体方案是什么？

**参考回答：**

我会按照「底层算力 → 模型服务 → 平台能力 → 应用层」四层来规划：

```
┌──────────────────────────────────────────────────────────┐
│                    应用层                                  │
│  Dify工作流 | 智能体平台 | RAG知识库 | 业务垂类应用         │
├──────────────────────────────────────────────────────────┤
│                   平台能力层                               │
│  LLMGateway | 向量数据库(Milvus) | 知识库管理 | 插件生态    │
├──────────────────────────────────────────────────────────┤
│                   模型服务层                               │
│  推理引擎(vLLM/TGI) | 模型注册中心 | 微调平台(LoRA)        │
├──────────────────────────────────────────────────────────┤
│                   算力基础层                               │
│  K8s集群 | GPU调度(CRD/Operator) | 监控(Prometheus)       │
└──────────────────────────────────────────────────────────┘
```

**优先级排序：**
1. 先上 LLMGateway + Dify，快速支撑业务需求（2 周内）；
2. 建设向量数据库和 RAG 知识库，支撑知识密集型业务（1 个月）；
3. 搭建 vLLM 推理服务，降低外部 API 依赖（1.5 个月）；
4. 开发 K8s Operator 实现 GPU 精细调度（3 个月）；
5. 国产化硬件适配（华为昇腾）（持续推进）；

---

#### Q13：项目中让你印象最深刻的技术挑战是什么？

**参考回答：**

印象最深刻的是**电站派单 AI 系统从 30% 到 80% AI 占比的攻坚过程**。

初期 AI 派单只有 30% 覆盖率，根本原因是模型对「低效派单」和「正常派单」的边界判断太模糊，误报率高，运维人员不信任 AI 结果，人工干预比例大。

我做了三件关键事情：

**第一：从业务视角重新定义特征**  
与电站运维专家深度访谈，发现「电站历史派单后发电量恢复时长」是最强特征，但原始数据中没有直接记录，需要通过时序数据反向推算。这个特征引入后，模型 F1 提升了近 10 个百分点。

**第二：双模型协同（预测 + 审单）**  
将任务拆分为两个模型：TimesNet 负责预测「低效是否存在」，KNN 审单模型负责验证「发电量是否真正恢复」，形成决策闭环。单模型时，误报会直接传递到下游；双模型交叉验证后，误报率大幅下降。

**第三：渐进式上线建立信任**  
不是一次性切换，而是从置信度最高的 20% 样本开始自动派单，逐步扩大覆盖范围，同时提供可解释性报告（为什么 AI 这么派单），让运维人员理解 AI 的决策依据，信任度建立后自然扩大了 AI 自动化比例。

---

## 三、FAQ 技术专题

### 专题一：Dify 与 Agent 平台

---

**Q：Dify 和 LangChain 的核心区别是什么？什么场景选 Dify？**

| 维度 | Dify | LangChain |
|------|------|-----------|
| 定位 | 面向业务的 LLM 应用平台（有 UI） | 面向开发者的编程框架 |
| 上手门槛 | 低（可视化配置） | 较高（纯代码） |
| 可扩展性 | 通过插件和 API 扩展 | 极高（纯代码自定义） |
| 工作流 | 可视化 DAG 编排 | 代码定义 Chain/Graph |
| 适合场景 | 快速落地业务应用、非技术用户参与配置 | 复杂定制化 Agent 开发 |

**推荐策略**：用 Dify 承接业务侧快速交付，用 LangGraph 处理复杂的 Agent 逻辑，两者通过 API 集成。

---

**Q：AgentScope 与 LangGraph 相比有什么优势？**

- AgentScope 对**多 Agent 通信模式**（点对点、广播、群组）支持更原生，内置 MsgHub 消息总线；
- LangGraph 更擅长**有状态的工作流编排**，与 LangChain 生态深度集成；
- AgentScope 的可视化 Studio 对调试多 Agent 交互更友好；
- 实际项目中我同时使用两者：AgentScope 处理 Agent 间通信，LangGraph 处理复杂的任务编排逻辑。

---

**Q：如何防止 Agent 产生幻觉或执行危险操作？**

1. **工具层防护**：所有工具调用前增加参数校验，危险操作（删除、转账等）要求 Agent 二次确认；
2. **输出验证**：对 Agent 最终输出用规则引擎做合规性检查；
3. **沙箱执行**：代码执行类工具在容器沙箱中运行，限制系统调用；
4. **Human-in-the-loop**：对高风险操作强制加入人工审批节点；
5. **Prompt 护栏**：System Prompt 中明确禁止的行为边界，配合 LLM 的 Safety 能力；

---

**Q：RAG 和 Fine-tuning 如何选择？**

| 场景 | 推荐方案 |
|------|----------|
| 知识实时更新频繁 | RAG（向量数据库更新快） |
| 需要特定风格/格式输出 | Fine-tuning |
| 企业私有知识库问答 | RAG（知识隔离更安全） |
| 模型需要掌握特定领域推理能力 | Fine-tuning（SFT） |
| 预算有限 | RAG（无需训练成本） |

**最佳实践**：两者结合，用 Fine-tuning 让模型学会领域语言和推理范式，用 RAG 补充实时知识。

---

### 专题二：LLMGateway 与推理优化

---

**Q：vLLM 相比普通 Transformers 推理有什么优势？**

vLLM 的核心创新是 **PagedAttention**：
- 将 KV Cache 分页管理（类似操作系统的虚拟内存），消除内存碎片；
- 支持 Continuous Batching（动态批处理），GPU 利用率从 40% 提升到 80%+；
- 相同硬件下吞吐量提升 3-10 倍；

**适用场景**：高并发推理服务（QPS > 10），对延迟和吞吐量要求高的生产环境。

---

**Q：大模型流式输出（SSE/WebSocket）如何在网关层正确代理？**

```python
# FastAPI SSE 代理示例
from fastapi.responses import StreamingResponse
import httpx

async def proxy_stream(request_body: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", 
            upstream_url, 
            json=request_body,
            timeout=300
        ) as response:
            async def generate():
                async for chunk in response.aiter_bytes():
                    yield chunk
            return StreamingResponse(
                generate(), 
                media_type="text/event-stream"
            )
```

**注意事项**：
- 网关层不能缓冲流式响应，必须透传；
- 计费在流结束后（收到 `[DONE]` 信号）统计 token；
- 设置合理的 TCP keepalive，避免长连接被中间设备断开；

---

**Q：滑动窗口限流如何用 Redis 实现？**

```python
import redis
import time

def is_allowed(user_id: str, max_requests: int, window_seconds: int) -> bool:
    r = redis.Redis()
    key = f"rate_limit:{user_id}"
    now = time.time()
    window_start = now - window_seconds
    
    pipe = r.pipeline()
    # 清理窗口外的旧记录
    pipe.zremrangebyscore(key, 0, window_start)
    # 查询窗口内的请求数
    pipe.zcard(key)
    # 添加当前请求
    pipe.zadd(key, {str(now): now})
    # 设置过期时间
    pipe.expire(key, window_seconds)
    
    results = pipe.execute()
    current_count = results[1]
    
    return current_count < max_requests
```

---

### 专题三：K8s 与云原生

---

**Q：K8s HPA（水平自动伸缩）如何配置？有哪些注意事项？**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-platform-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**注意事项**：
- AI 推理服务的 CPU 使用率波动大，建议配合自定义指标（如 QPS）做扩缩容；
- 设置 `scaleDown.stabilizationWindowSeconds`（默认 300s）避免频繁缩容；
- GPU 资源不支持 HPA，需要手动或通过 KEDA 等扩展实现；

---

**Q：Celery Worker 如何做优雅停机（Graceful Shutdown）？**

```python
# celery worker 配置
app = Celery(...)
app.conf.update(
    # Worker 在接到停止信号后，等待当前任务完成再退出
    worker_cancel_long_running_tasks_on_connection_loss=True,
    # 任务超时时间
    task_soft_time_limit=300,
    task_time_limit=360,
)
```

```bash
# K8s Pod 配置 preStop hook
lifecycle:
  preStop:
    exec:
      command: ["celery", "-A", "app", "control", "shutdown"]
terminationGracePeriodSeconds: 400
```

---

### 专题四：模型训练与优化

---

**Q：LoRA 微调的原理是什么？相比全量微调有什么优势？**

**原理：**  
LoRA（Low-Rank Adaptation）在原始权重矩阵旁并联一个低秩分解矩阵 `ΔW = BA`（其中 B∈R^(d×r), A∈R^(r×k), r << d,k），只训练 A 和 B，冻结原始权重。

**优势：**
- 参数量仅为全量微调的 0.1%-1%，显存需求大幅降低；
- 多任务可以共享同一基础模型，只切换不同的 LoRA 适配器；
- 支持合并到原始权重，推理无额外开销；

**推荐配置（7B 模型）：**
- `r=16, lora_alpha=32, lora_dropout=0.05`；
- 目标模块：`q_proj, v_proj`（注意力层效果最好）；
- 使用 QLoRA（4-bit 量化 + LoRA）可在单张 24GB 显卡上微调 13B 模型；

---

**Q：向量数据库 HNSW 索引的原理是什么？如何调优？**

**HNSW（Hierarchical Navigable Small World）原理：**  
- 多层图结构，上层图连接较少节点（长距离跳转），下层图包含所有节点（精确搜索）；
- 查询从顶层开始，贪心地向目标向量靠近，逐层下降，最终在底层精确匹配；
- 时间复杂度 O(log N)，比暴力搜索快数量级；

**调优参数：**

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `M` | 每个节点的最大连接数 | 16-64，越大召回率越高但内存占用增加 |
| `ef_construction` | 构建时的候选集大小 | 200-400 |
| `ef_search` | 查询时的候选集大小 | 64-256，越大召回率越高但延迟增加 |

**权衡**：在我们的项目中，召回率 95%@ef_search=128 是性能和延迟的最佳平衡点，P99 延迟约 20ms（百万级向量）。

---

**Q：如何评估 RAG 系统的效果？有哪些指标？**

| 评估维度 | 指标 | 说明 |
|----------|------|------|
| 检索质量 | Recall@K | 相关文档是否被召回 |
| 检索质量 | MRR（平均倒数排名） | 相关文档排名越靠前越好 |
| 生成质量 | Faithfulness | 答案是否基于检索到的文档 |
| 生成质量 | Answer Relevance | 答案是否回答了用户问题 |
| 端到端 | Exact Match / F1 | 与标准答案的匹配度 |

**工具推荐**：使用 RAGAS 框架自动化评估，支持无参考答案的评估（利用 LLM 作为评判者）。

---

## 四、面试准备清单

### 重点准备项（与岗位高匹配）
- [x] Dify 源码架构讲解（工作流引擎、插件系统、API 扩展机制）
- [x] LLMGateway 设计细节（限流算法、降级策略、计费实现）
- [x] 多 Agent 协作架构（AgentScope 消息总线、任务分解、结果聚合）
- [x] RAG 全链路优化（分块策略、混合检索、重排序）
- [x] K8s 实践（HPA、Helm、监控体系）

### 需要补强的方向
- [ ] K8s CRD/Operator 开发（kubebuilder 框架实操）
- [ ] vLLM/TGI 推理引擎部署与性能调优实战
- [ ] 华为昇腾 NPU 适配基础知识
- [ ] GPU 显存切分技术（MPS、MIG）

### 反问面试官的好问题
1. 当前 AI 中台的技术架构现状是怎样的？最大的技术挑战是什么？
2. GPU 算力调度这块，目前是用开源方案（如 KubeRay）还是自研 Operator？
3. Dify 二次开发的深度如何？是做插件扩展还是涉及工作流引擎层的修改？
4. 团队技术栈偏好和代码规范是什么？
5. 国产化适配（华为昇腾）目前进展到哪个阶段？

---

> **文档版本**：v1.0 | 2026-03-16  
> **适用岗位**：资深 AI 架构师（Agent & Infra）35-60K · 15薪  
> **文档目的**：模拟面试备战，梳理技术深度与表达逻辑
