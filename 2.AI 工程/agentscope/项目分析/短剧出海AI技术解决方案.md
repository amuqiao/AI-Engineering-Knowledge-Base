# 短剧出海 AI 技术解决方案全景

> 面向 AI 出海短剧平台，从业务痛点出发，提供覆盖 **Agent 编排、模型训练/微调、推理工程、合规绕避** 的全栈技术解决方案。  
> 岗位背景：业务驱动型 AI 智能体研发工程师，要求同时具备算法与工程能力。

---

## 一、业务场景与痛点全景

```mermaid
flowchart TB
    classDef painStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef sceneStyle  fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef valueStyle  fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    subgraph PAIN["核心痛点"]
        direction LR
        P1["人工成本高<br>剪辑/翻译/配音<br>依赖大量人力"]:::painStyle
        P2["制作周期长<br>从剧本到上线<br>多周起步"]:::painStyle
        P3["多语种本地化难<br>文化差异/口型/配音<br>质量参差"]:::painStyle
        P4["合规风险大<br>各国政策不同<br>随时面临下架"]:::painStyle
        P5["投放效率低<br>素材生成慢<br>A/B测试成本高"]:::painStyle
        P6["线上模型受限<br>安全审查过滤<br>无法生成特定内容"]:::painStyle
    end
    class PAIN layerStyle

    subgraph SCENE["四大核心场景"]
        direction LR
        S1["本地化生产<br>Localization"]:::sceneStyle
        S2["内容生产<br>Content Production"]:::sceneStyle
        S3["合规审核<br>Compliance Review"]:::sceneStyle
        S4["运营投放<br>Marketing & Distribution"]:::sceneStyle
    end
    class SCENE layerStyle

    subgraph VALUE["核心价值目标"]
        direction LR
        V1["降本<br>人力成本↓60%+"]:::valueStyle
        V2["增效<br>产能↑5-10x"]:::valueStyle
        V3["合规<br>封号风险↓90%+"]:::valueStyle
        V4["规模化<br>多国快速扩张"]:::valueStyle
    end
    class VALUE layerStyle

    PAIN --> SCENE
    SCENE --> VALUE

    NOTE["技术解法不只有 Agent，还包括：<br>专属模型训练/微调（LoRA）、推理引擎优化、<br>私有化部署、合规策略绕避"]:::noteStyle
    NOTE -.- SCENE

    %% 边索引：0-1，共 2 条
    linkStyle 0,1 stroke:#374151,stroke-width:2px
```

---

## 二、场景一：多语种本地化

### 2.1 痛点分析

| 环节 | 当前问题 | 量化影响 |
|------|----------|---------|
| 字幕翻译 | 机器翻译质量差，俚语/文化梗错译 | 差评率高，用户流失 |
| AI 配音 | 公有 TTS 无小语种（泰语/印尼语等）或音色单调 | 无法覆盖东南亚核心市场 |
| 口型对齐 | 换语言后口型与音频不同步 | 观感差，出戏 |
| 文化适配 | 直译导致文化冲突（称谓/节日/禁忌） | 内容合规风险 |

### 2.2 解决方案

#### 方案 A：Agent 流水线（快速上线）

```mermaid
flowchart LR
    classDef userStyle    fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle   fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef toolStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef storeStyle   fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    INPUT["原始视频<br>中文剧集"]:::userStyle

    subgraph PIPELINE["sequential_pipeline 顺序流水线"]
        direction LR
        ASR["ASR Agent<br>语音转文字<br>Whisper/FunASR"]:::agentStyle
        TRANS["翻译 Agent<br>文化适配增强<br>LLM + 术语库"]:::agentStyle
        TTS["配音 Agent<br>多语种 TTS<br>CosyVoice/XTTS"]:::agentStyle
        ALIGN["口型对齐 Agent<br>Wav2Lip/SadTalker<br>视频合成"]:::agentStyle
        QC["质量检查 Agent<br>评分 + 人工复核<br>触发条件"]:::agentStyle
    end
    class PIPELINE layerStyle

    OUTPUT["本地化视频<br>多语种版本"]:::userStyle

    INPUT --> ASR
    ASR   -->|"原始字幕"| TRANS
    TRANS -->|"目标语字幕"| TTS
    TTS   -->|"配音音频"| ALIGN
    ALIGN -->|"合成视频"| QC
    QC    --> OUTPUT

    NOTE["AgentScope: sequential_pipeline([asr, trans, tts, align, qc])<br>每步输出作为下步输入，质量不达标时回流重做"]:::noteStyle
    NOTE -.- PIPELINE

    %% 边索引：0-5，共 6 条
    linkStyle 0,1,2,3,4,5 stroke:#0891b2,stroke-width:2px
```

```python
# AgentScope 实现
from agentscope.pipeline import sequential_pipeline

result = await sequential_pipeline(
    [asr_agent, translation_agent, tts_agent, lip_sync_agent, qc_agent],
    msg=Msg("user", video_path, "user"),
)
```

#### 方案 B：小语种专属模型训练（核心竞争力）

**问题**：公有 TTS（如 Azure / 阿里云）的泰语、印尼语、越南语音色数量少、音质差、无法自定义。

**解法**：基于开源 TTS 框架训练私有小语种音色模型：

```
数据准备：
  ├── 采集目标语种原声演员语料 2~5 小时（已授权）
  ├── 降噪处理 → 切片 → 文本对齐（MFA）
  └── 构建 phoneme 词典（泰语/越南语需专项处理声调）

模型选择：
  ├── CosyVoice2（阿里开源，支持多语种 zero-shot）
  ├── XTTS-v2（Coqui 开源，支持 17 种语言克隆）
  └── Kokoro（轻量，适合边缘推理）

训练策略：
  ├── 基座模型：预训练多语种 TTS
  ├── LoRA 微调：注入目标音色特征（显存 <16GB 可跑）
  └── 产出：专属音色模型，可无限复用
```

#### 方案 C：口型对齐（视频生成工程）

| 方案 | 原理 | 适用场景 | 代价 |
|------|------|---------|------|
| **Wav2Lip** | 基于音频生成嘴部区域 | 快速、轻量 | 脸部其他区域质量一般 |
| **SadTalker** | 3DMM 驱动全脸生成 | 表情更自然 | 计算量大 |
| **MuseTalk** | 扩散模型嘴部修复 | 高质量 | 需 A100 级 GPU |
| **商用 API** | HeyGen / D-ID | 快速集成 | 成本高，数据出境 |

**推荐**：本地部署 Wav2Lip + MuseTalk 的混合策略——普通镜头用 Wav2Lip（快），特写镜头用 MuseTalk（质量）。

---

## 三、场景二：音视频内容生产

### 3.1 痛点分析

- 剧本拆解为分镜、台词、情绪标注全靠人工，单集需 3~5 天
- 高光剪辑依赖剪辑师主观判断，效率低且难复制
- 投放素材（15s/30s 混剪）每天需数百条，人力瓶颈明显

### 3.2 解决方案

#### 方案 A：内容生产 Meta Planner（动态任务分解）

```mermaid
flowchart LR
    classDef userStyle   fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef planStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef workerStyle fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef toolStyle   fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    SCRIPT["原始剧本<br>Script"]:::userStyle

    PLANNER["剧本拆解 Planner<br>ReActAgent + PlanNotebook"]:::planStyle

    subgraph WORKERS["并发 Worker 层（fanout_pipeline）"]
        direction TB
        W1["情节分析 Worker<br>高潮点 / 冲突识别"]:::workerStyle
        W2["台词提取 Worker<br>关键金句 / 情绪标注"]:::workerStyle
        W3["镜头规划 Worker<br>场景切割 / 节奏建议"]:::workerStyle
    end
    class WORKERS layerStyle

    subgraph OUTPUT_LAYER["输出层（sequential_pipeline）"]
        direction LR
        CUT["自动剪辑 Agent<br>FFmpeg + 时间轴生成"]:::toolStyle
        MIX["混剪合成 Agent<br>BGM / 字幕 / 特效"]:::toolStyle
        EXPORT["多格式导出 Agent<br>竖屏9:16 / 方形 / 横屏"]:::toolStyle
    end
    class OUTPUT_LAYER layerStyle

    RESULT["投放素材包<br>多版本短视频"]:::userStyle

    SCRIPT  --> PLANNER
    PLANNER -->|"create_worker × 3"| WORKERS
    W1      -->|"ResultModel"| PLANNER
    W2      -->|"ResultModel"| PLANNER
    W3      -->|"ResultModel"| PLANNER
    PLANNER -->|"汇总分析结果"| CUT
    CUT     --> MIX
    MIX     --> EXPORT
    EXPORT  --> RESULT

    NOTE["WORKERS 层用 fanout_pipeline 并发执行（互不依赖）<br>OUTPUT 层用 sequential_pipeline 串行（前序结果驱动后续）"]:::noteStyle
    NOTE -.- OUTPUT_LAYER

    %% 边索引：0-8，共 9 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1 stroke:#dc2626,stroke-width:2px
    linkStyle 2,3,4 stroke:#0891b2,stroke-width:2px
    linkStyle 5 stroke:#dc2626,stroke-width:2px
    linkStyle 6,7,8 stroke:#d97706,stroke-width:2px
```

#### 方案 B：视频理解模型（亮点技术点）

公有大模型（GPT-4V / Qwen-VL）处理视频成本极高，且无法处理完整集数（1集≈20~40分钟）：

```
长视频处理工程方案：
  ├── 均匀采帧：每秒 1~2 帧 → 降低处理量
  ├── 场景切割：PySceneDetect 自动切片
  ├── 关键帧提取：CLIP 向量相似度去重
  ├── 分片并行理解：fanout_pipeline 分发各段给视频理解 Agent
  └── 结果聚合：时间轴对齐 + 叙事摘要生成

模型选择：
  ├── Qwen2.5-VL（开源，支持长视频，可本地部署）
  ├── InternVL2（多模态理解能力强）
  └── LLaVA-Video（专为视频优化）
```

#### 方案 C：剪辑模型微调（降低通用模型的业务偏差）

通用 LLM 对"什么是短剧高光"的判断与真实用户喜好存在偏差，可以用业务数据微调：

```
训练数据构建：
  ├── 标注素材：历史高播放量片段（正样本）vs 低播放量片段（负样本）
  ├── 特征：完播率、评论情绪、弹幕密度、点赞率
  └── 标注维度：冲突强度、反转节点、情绪峰值、节奏感

微调策略：
  ├── 基座：Qwen2.5-7B 或 LLaMA-3-8B
  ├── 方法：LoRA（rank=16，target=q_proj/v_proj）
  ├── 任务：剧本片段打分回归 + 高光点定位分类
  └── 显存：单卡 A100 40G，训练约 4~8 小时
```

---

## 四、场景三：多模态合规审核

### 4.1 痛点分析

出海内容合规是最高优先级风险，不同市场规则差异巨大：

| 市场 | 主要限制 | 典型风险 |
|------|---------|---------|
| 东南亚（泰/印尼） | 宗教敏感、皇室相关、暴力 | 账号封禁、内容下架 |
| 欧美（Meta/TikTok） | 仇恨言论、裸露、版权 | DMCA 投诉、广告限流 |
| 中东 | 宗教禁忌、性别相关 | 市场准入被拒 |
| 国内出口 | 政治敏感、历史 | 内容无法通过审查 |

### 4.2 解决方案

#### 方案 A：多模态合规审核 Agent 集群（MsgHub 协商模式）

```mermaid
flowchart LR
    classDef userStyle   fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle  fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef judgeStyle  fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef hubStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    VIDEO["待审核视频<br>+ 字幕 + 音频"]:::userStyle

    subgraph FANOUT["并发多模态检测（fanout_pipeline）"]
        direction TB
        VA["视觉审核 Agent<br>NSFW / 暴力 / 版权水印"]:::agentStyle
        AA["音频审核 Agent<br>ASR + 敏感词 / 违禁音乐"]:::agentStyle
        TA["文本审核 Agent<br>字幕 / 台词 LLM 审核"]:::agentStyle
        CA["文化合规 Agent<br>地区规则引擎 + RAG 政策库"]:::agentStyle
    end
    class FANOUT layerStyle

    subgraph HUB_DEBATE["MsgHub 协商裁决"]
        direction LR
        HUB["MsgHub<br>广播各方审核意见"]:::hubStyle
        JUDGE["裁决 Agent<br>综合评分 + 结构化输出"]:::judgeStyle
    end
    class HUB_DEBATE layerStyle

    RESULT["审核报告<br>通过 / 打回 / 需人审"]:::userStyle

    VIDEO --> FANOUT
    VA    -->|"视觉风险报告"| HUB
    AA    -->|"音频风险报告"| HUB
    TA    -->|"文本风险报告"| HUB
    CA    -->|"合规建议"| HUB
    HUB   -->|"广播所有意见"| JUDGE
    JUDGE -->|"AuditResult(passed, risks, actions)"| RESULT

    NOTE["四路并发（fanout），结果汇入 MsgHub 供裁决 Agent 综合判断<br>风险等级低→自动通过，中→标注复核，高→直接打回"]:::noteStyle
    NOTE -.- HUB_DEBATE

    %% 边索引：0-6，共 7 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1,2,3,4 stroke:#0891b2,stroke-width:2px
    linkStyle 5 stroke:#d97706,stroke-width:2px
    linkStyle 6 stroke:#dc2626,stroke-width:2.5px
```

#### 方案 B：私有合规检测模型（精准 + 不依赖公有 API）

公有 API（阿里云内容安全 / Google Vision SafeSearch）对短剧特有内容（如古装打斗、言情场景）误判率高，且无法细化到地区规则。

```
自建合规模型方案：

1. 视觉 NSFW 检测
   ├── 基座：CLIP-ViT-L-14 或 EVA-CLIP
   ├── 微调数据：业务标注数据集（按地区分类）
   ├── 方法：LoRA + 二分类/多标签头
   └── 部署：TensorRT 量化为 INT8，单帧推理 < 5ms

2. 音频敏感内容检测
   ├── 基座：Whisper Large-v3（ASR）+ 分类头
   ├── 任务：转录文本后接敏感词 NER 模型
   └── 本地词库：按国家维护敏感词/实体黑名单

3. 地区规则引擎（RAG + 规则库）
   ├── 建立各国合规政策文档向量库（Qdrant）
   ├── 新内容审核时检索相关政策
   └── Agent 结合检索结果给出合规建议
```

#### 方案 C：针对公有 LLM 法律法规限制的解决方案

**核心问题**：出海短剧常涉及打斗、言情、紧张对峙等场景，公有 LLM API（GPT/Claude/Qwen）的安全过滤器会错误拒绝生成相关内容描述、剧情续写、对白等。

```
解决路径（由低到高风险，按需选择）：

路径 ①：Prompt 工程绕避（适合轻度场景）
  ├── 使用虚构化框架：「为小说创作场景描述，用隐喻表达」
  ├── 角色扮演声明：「以下是专业影视剧本，内容面向成年观众」
  └── 分步生成：先生成场景背景，再补充情节细节

路径 ②：自研小模型微调（适合中高频需求）
  ├── 选用未经 RLHF 对齐的基座：
  │   ├── LLaMA-3-8B/70B（Meta 开源，安全限制最少）
  │   ├── Mistral-7B（欧洲开源，对言情内容相对宽松）
  │   └── Qwen2.5（阿里开源，中文支持强）
  ├── 移除/绕过安全限制：
  │   ├── 不加载 safety_checker 组件（SD 类模型）
  │   ├── system_prompt 中明确声明创作场景
  │   └── DPO 微调：用业务数据替换 RLHF 的拒绝倾向
  └── LoRA 注入：注入短剧写作风格，无需全量训练

路径 ③：私有化部署开源模型（最高自由度）
  ├── 模型：LLaMA-3-70B / Qwen2.5-72B / DeepSeek-V3
  ├── 推理引擎：vLLM（高吞吐）/ SGLang（低延迟）
  ├── 量化：AWQ INT4（显存减半，精度损失 <1%）
  └── 部署：多卡 Tensor Parallel，支持并发 100+ 请求
```

**量化与推理引擎选型对比**：

| 推理引擎 | 吞吐量 | 延迟 | 量化支持 | 适用场景 |
|---------|--------|------|---------|---------|
| **vLLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | GPTQ / AWQ / FP8 | 批量内容生产（高并发） |
| **SGLang** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | GPTQ / AWQ | 实时对话（低延迟优先） |
| **llama.cpp** | ⭐⭐ | ⭐⭐⭐ | GGUF Q4/Q8 | CPU 部署 / 边缘设备 |
| **TensorRT-LLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | INT4/INT8/FP8 | NVIDIA GPU 极致优化 |
| **MLC-LLM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 多后端 | 跨平台（含移动端） |

---

## 五、场景四：运营投放辅助

### 5.1 痛点分析

- 每日需要生成数百条不同国家/语种的封面图文素材
- A/B 测试需快速产出多版本标题/封面组合
- 投放数据反馈（CTR/完播率）无法自动驱动素材优化

### 5.2 解决方案

#### 方案 A：素材生产 Agent 流水线

```mermaid
flowchart LR
    classDef userStyle   fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef agentStyle  fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef genStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef abStyle     fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    DRAMA["短剧元数据<br>标题/人物/情节"]:::userStyle

    subgraph GEN["素材并发生成（fanout_pipeline）"]
        direction TB
        TITLE["标题生成 Agent<br>× 10 变体<br>不同情绪/语气"]:::genStyle
        COVER["封面文案 Agent<br>× 5 变体<br>悬念/情感/福利款"]:::genStyle
        COPY["广告文案 Agent<br>适配各平台风格<br>TikTok/Meta/YouTube"]:::genStyle
    end
    class GEN layerStyle

    subgraph AB["A/B 测试管理"]
        direction LR
        SELECT["组合选优 Agent<br>历史 CTR 数据<br>RAG 检索优质模板"]:::abStyle
        MONITOR["效果监控 Agent<br>实时 CTR/完播率<br>→ 触发再生成"]:::abStyle
    end
    class AB layerStyle

    OUTPUT["投放素材包<br>多版本 × 多语种"]:::userStyle

    DRAMA   --> GEN
    TITLE   -->|"标题候选集"| SELECT
    COVER   -->|"封面候选集"| SELECT
    COPY    -->|"文案候选集"| SELECT
    SELECT  -->|"最优组合"| OUTPUT
    OUTPUT  -->|"投放后数据回流"| MONITOR
    MONITOR -->|"低效素材触发重生成"| GEN

    NOTE["闭环：投放数据 → 效果评估 → 触发重生成<br>SELECT Agent 用 RAG 检索历史高 CTR 素材模板"]:::noteStyle
    NOTE -.- AB

    %% 边索引：0-6，共 7 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1,2,3 stroke:#d97706,stroke-width:2px
    linkStyle 4 stroke:#dc2626,stroke-width:2px
    linkStyle 5 stroke:#1e40af,stroke-width:2px
    linkStyle 6 stroke:#dc2626,stroke-width:1.5px,stroke-dasharray:4 3
```

#### 方案 B：图像生成模型私有化（封面图生成）

公有图像生成 API（Midjourney/DALL-E/Stable Diffusion API）存在：
- 人物肖像相似度要求无法满足（需与真实演员匹配）
- 跨境数据合规风险（演员面部数据出境）
- 成本高（商业用量下单张 $0.02~0.1）

```
私有图像生成方案：

1. 基础能力：SDXL / FLUX.1 本地部署
   ├── 量化：INT8 / FP16，24G 显存可跑 SDXL
   └── 推理加速：TensorRT 优化，生成速度 < 2s/张

2. 演员风格 LoRA 训练（核心）
   ├── 数据：15~30 张演员图片（已授权）
   ├── 方法：Dreambooth + LoRA（rank=8~16）
   ├── 训练时长：约 30~60 分钟（A100）
   └── 效果：生成图与演员相似度 > 85%

3. 文案叠加与风格化
   ├── 工具：PIL / ComfyUI 工作流自动化
   └── 模板库：按平台/地区维护封面模板
```

---

## 六、横向技术能力：模型训练与工程体系

### 6.1 LoRA 微调在短剧业务的系统应用

```
短剧业务 LoRA 矩阵：

                  LLM 类                    视觉类
              ┌──────────────────┬──────────────────────┐
  文本生成    │ 剧情续写风格适配  │                      │
  任务        │ 台词风格微调      │       N/A            │
              │ 合规审核专属分类  │                      │
              ├──────────────────┼──────────────────────┤
  多模态      │ 视频内容描述      │ 演员风格 LoRA         │
  任务        │ 封面文案生成      │ 封面风格统一化        │
              │                  │ NSFW 检测微调         │
              └──────────────────┴──────────────────────┘

LoRA 参数建议：
  - rank：8（轻量风格迁移）/ 16（能力注入）/ 32（任务专属）
  - alpha：rank × 2（通用经验值）
  - target_modules：q_proj, v_proj（最小）/ 全 attention（最强）
  - 数据量：500~2000 条高质量样本即可（质量 > 数量）
```

### 6.2 推理引擎部署架构

```mermaid
flowchart TB
    classDef clientStyle fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef svcStyle    fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef modelStyle  fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef infraStyle  fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle   fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle  fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    subgraph CLIENT["业务层（Agent 编排）"]
        direction LR
        AGT["AgentScope<br>Planner / Worker"]:::clientStyle
        API["对外 REST API<br>/ gRPC"]:::clientStyle
    end
    class CLIENT layerStyle

    subgraph ROUTER["路由层（按任务分流）"]
        direction LR
        RT["智能路由<br>延迟敏感 → 小模型<br>质量优先 → 大模型<br>合规相关 → 私有模型"]:::svcStyle
    end
    class ROUTER layerStyle

    subgraph INFERENCE["推理层"]
        direction LR
        VLLM["vLLM 集群<br>高并发批量生产<br>AWQ INT4 量化"]:::modelStyle
        SGLANG["SGLang 集群<br>实时交互场景<br>低延迟优先"]:::modelStyle
        TRT["TensorRT-LLM<br>极致吞吐<br>NVIDIA GPU"]:::modelStyle
    end
    class INFERENCE layerStyle

    subgraph MODELS["模型仓库"]
        direction LR
        PRIV["私有微调模型<br>LoRA Adapter<br>合规/风格专属"]:::infraStyle
        PUB["公有 API 兜底<br>GPT-4o / Claude<br>敏感度低任务"]:::infraStyle
    end
    class MODELS layerStyle

    CLIENT --> ROUTER
    ROUTER --> VLLM
    ROUTER --> SGLANG
    ROUTER --> TRT
    VLLM   --> PRIV
    SGLANG --> PRIV
    TRT    --> PRIV
    ROUTER -->|"兜底"| PUB

    NOTE["核心原则：合规敏感任务必须走私有模型<br>公有 API 只处理低风险、低频、非核心任务"]:::noteStyle
    NOTE -.- ROUTER

    %% 边索引：0-7，共 8 条
    linkStyle 0 stroke:#1d4ed8,stroke-width:2px
    linkStyle 1,2,3 stroke:#0891b2,stroke-width:2px
    linkStyle 4,5,6 stroke:#059669,stroke-width:2px
    linkStyle 7 stroke:#374151,stroke-width:1.5px,stroke-dasharray:4 3
```

### 6.3 数据飞轮：用业务数据持续改进模型

```
运营数据 → 训练数据 → 更好的模型 → 更好的运营效果

具体实现：
  ├── 素材 CTR/完播率 → 标注"好/坏"生成样本 → 微调文案生成模型
  ├── 人工审核纠错记录 → 合规模型训练数据 → 减少误判
  ├── 用户评论情感 → 高光剪辑质量标签 → 剪辑模型优化
  └── 多语种用户反馈 → 翻译质量评分 → TTS/翻译模型迭代

工具链：
  ├── 数据标注：Label Studio / 内部标注平台
  ├── 实验管理：MLflow / Weights & Biases
  ├── 模型训练：LLaMA-Factory（LoRA 一键训练）/ Axolotl
  └── 模型评估：自定义业务指标（非仅通用 benchmark）
```

---

## 七、合规策略专题：公有 API 限制的系统性解决

### 7.1 限制分类与对策矩阵

| 限制类型 | 典型场景 | 公有 API 行为 | 推荐解法 |
|---------|---------|-------------|---------|
| 暴力/打斗描写 | 武侠/古装剧对白续写 | 拒绝生成，或输出被稀释 | 私有 LLaMA-3/Qwen2.5，不加安全限制 |
| 言情/亲密场景 | 言情剧台词、剧情描写 | 过度审查，内容失真 | Mistral + 业务 LoRA |
| 演员面部生成 | 封面图、营销海报 | 版权/肖像权过滤 | SD/FLUX 私有部署 + 演员 LoRA |
| 政治敏感内容审核 | 出海内容的政治检查 | 各平台规则不透明 | 本地规则库 + RAG + 专属审核模型 |
| 版权音乐检测 | 背景音乐合规 | 无此能力 | Dejavu/AcoustID 本地部署 |
| 未成年保护检测 | 各国 COPPA/GDPR 要求 | 通用模型精度不足 | 专项训练年龄估计模型 |

### 7.2 私有化部署的合规收益

```
公有 API 的隐性合规风险：
  ├── 用户数据/演员肖像上传至境外服务器 → GDPR / 中国数据安全法
  ├── 剧本内容泄露给第三方 → 知识产权风险
  └── API 服务不可控下线 → 业务连续性风险

私有化部署解决上述全部问题：
  ├── 数据不出境：推理全部在自有服务器/云 VPC 内
  ├── 内容不泄露：模型权重私有，无调用日志外传
  └── 服务可控：SLA 自定义，不受第三方影响

成本估算（70B 量化模型）：
  ├── 硬件：4× A100 80G，约 ¥40 万/套（租用约 ¥8/小时）
  ├── 量化后吞吐：AWQ INT4 下约 3000 tokens/s
  └── 成本对比：百万 token 费用约 ¥0.5（vs GPT-4 的 ¥100+）
```

---

## 八、技术选型速查

| 能力域 | 推荐方案 | 备注 |
|--------|---------|------|
| Agent 编排框架 | AgentScope（ReActAgent + Pipeline） | 已有深度分析 |
| 文本 LLM | Qwen2.5-72B（AWQ INT4）+ vLLM | 中文强，推理效率高 |
| 视觉理解 | Qwen2.5-VL-7B / InternVL2 | 开源，支持本地部署 |
| TTS 配音 | CosyVoice2 + LoRA 音色微调 | 支持小语种，可克隆 |
| ASR 转写 | FunASR / Whisper Large-v3 | 中文/多语种均支持 |
| 口型对齐 | Wav2Lip（快）+ MuseTalk（精） | 按镜头类型混合使用 |
| 图像生成 | SDXL / FLUX.1 + Dreambooth LoRA | 演员风格定制 |
| LoRA 训练 | LLaMA-Factory / Axolotl | 一键训练，支持全系列模型 |
| 推理引擎 | vLLM（高并发）/ TensorRT-LLM（极致性能） | 按场景选择 |
| 向量库（RAG） | Qdrant（合规政策库） | 开源，支持本地部署 |
| 视频处理 | FFmpeg + PySceneDetect + CLIP | 标准工具链 |
| 合规检测 | 自训 ViT 分类器 + 本地规则引擎 | 不依赖公有 API |

---

## 九、核心价值主张（VP 面试话术逻辑）

```
我不是来做纯技术研发的。

我的目标是用三层技术体系，帮公司把短剧出海的全链路自动化：

  第一层（Agent 编排）：
    用 AgentScope 的 sequential / fanout / MsgHub 管道，
    把本地化、剪辑、审核、投放四条业务线各自串成自动化流水线，
    可以快速接入新的工具和模型，不重复造轮子。

  第二层（模型能力）：
    通用 API 解决不了的问题（言情台词、演员肖像、小语种配音、地区合规），
    用 LoRA 微调 + 私有化部署来补齐，
    同时用业务数据飞轮持续改进模型，形成自己的技术壁垒。

  第三层（工程基础）：
    vLLM + TensorRT-LLM + AWQ 量化，
    把推理成本打下来，支撑规模化出海，
    私有部署还解决了跨境数据合规问题。

最终目标：降本 60%+，产能提升 5~10 倍，封号风险降低 90%，
          用一套系统支撑多国多语种快速扩张。
```

---

> **文档关联：**
> - `docs/项目分析/agentscope.pipeline四种编排模式详解.md` — Pipeline 编排原语参考
> - `docs/项目分析/AgentScope 任务拆解模式.md` — Meta Planner / PlanNotebook 详解
> - `docs/项目分析/短剧业务痛点.md` — 业务背景与 VP 面试准备
