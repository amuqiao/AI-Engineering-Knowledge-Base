# AI 漫剧系统架构说明

## 1. 文档目标

本文档用于帮助快速理解 AI 漫剧系统在“创作、生成、审核、发布”主线上的整体结构、关键服务协作方式以及核心状态迁移。

本文档包含 5 张主图：

1. 平台分层图
2. 分层能力结构图
3. 流程图
4. 技术架构图
5. 状态图

建议阅读顺序：

1. 先看平台分层图，建立全局认知
2. 再看分层能力结构图，理解产品级分层总览
3. 再看流程图，理解主业务链路
4. 再看技术架构图，理解服务协作
5. 最后看状态图，理解生命周期和异常回路

---

## 2. 平台分层图

这张图回答：系统通常分成哪些层，每层负责什么，主线如何穿过这些层。

```mermaid
flowchart TB
    classDef clientStyle  fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef gatewayStyle fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef svcStyle     fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef aiStyle      fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef reviewStyle  fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef dbStyle      fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef cacheStyle   fill:#ea580c,stroke:#7c2d12,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px
    classDef infraStyle   fill:#fafaf9,stroke:#a8a29e,stroke-width:1.5px

    subgraph ENTRY["产品入口层"]
        direction LR
        CREATOR["创作工作台<br>选题 / 剧本 / 分镜"]:::clientStyle
        OPS["审核运营后台<br>审核 / 发布 / 回退"]:::clientStyle
        CHANNEL["分发渠道<br>App / 小程序 / 视频平台"]:::clientStyle
    end
    class ENTRY layerStyle

    subgraph ORCH["业务编排层"]
        direction LR
        PROJECT["项目中心<br>作品 / 章节 / 版本"]:::gatewayStyle
        WORKFLOW["工作流编排<br>任务拆解 / 重试 / 调度"]:::gatewayStyle
        CMS["内容中心<br>稿件管理 / 上架配置"]:::gatewayStyle
    end
    class ORCH layerStyle

    subgraph AI["AI 生成层"]
        direction LR
        SCRIPT["剧本生成服务<br>剧情扩写 / 台词生成"]:::aiStyle
        STORYBOARD["分镜生成服务<br>镜头拆分 / 画面提示"]:::aiStyle
        IMAGE["图像生成服务<br>角色 / 场景 / 画格"]:::aiStyle
        AUDIO["音频生成服务<br>配音 / 音效 / BGM"]:::aiStyle
        RENDER["成片合成服务<br>字幕 / 镜头 / 音画合成"]:::aiStyle
    end
    class AI layerStyle

    subgraph REVIEW["审核治理层"]
        direction LR
        SAFETY["内容安全服务<br>涉政 / 色情 / 暴恐 / 版权"]:::reviewStyle
        QA["质量校验服务<br>角色一致性 / 镜头完整性"]:::reviewStyle
        MANUAL["人工复核<br>打回 / 修订 / 放行"]:::reviewStyle
    end
    class REVIEW layerStyle

    subgraph INFRA["数据与基础设施层"]
        direction LR
        ASSET[("媒资库 / 对象存储<br>图片 / 音频 / 视频")]:::dbStyle
        DB[("业务数据库<br>项目 / 版本 / 发布记录")]:::dbStyle
        CACHE[("缓存<br>会话态 / 热数据")]:::cacheStyle
        MQ[("消息队列<br>异步任务 / 回调事件")]:::cacheStyle
        MODEL[("模型网关 / 外部模型<br>LLM / TTS / 图像模型")]:::dbStyle
    end
    class INFRA infraStyle

    CREATOR -->|"创作请求"| PROJECT
    OPS -->|"审核/发布操作"| CMS

    PROJECT -->|"任务编排"| WORKFLOW
    WORKFLOW -->|"调用生成能力"| SCRIPT
    WORKFLOW -->|"调用生成能力"| STORYBOARD
    WORKFLOW -->|"调用生成能力"| IMAGE
    WORKFLOW -->|"调用生成能力"| AUDIO
    WORKFLOW -->|"触发合成"| RENDER

    SCRIPT -->|"模型调用"| MODEL
    STORYBOARD -->|"模型调用"| MODEL
    IMAGE -->|"模型调用"| MODEL
    AUDIO -->|"模型调用"| MODEL

    SCRIPT -->|"脚本产物"| ASSET
    STORYBOARD -->|"分镜产物"| ASSET
    IMAGE -->|"画面产物"| ASSET
    AUDIO -->|"音频产物"| ASSET
    RENDER -->|"成片产物"| ASSET

    RENDER -->|"待审内容"| SAFETY
    SAFETY -->|"通过后质检"| QA
    SAFETY -->|"高风险转人工"| MANUAL
    QA -->|"合格稿件"| CMS
    MANUAL -->|"修订/放行"| CMS

    PROJECT -->|"项目元数据"| DB
    CMS -->|"发布记录"| DB
    WORKFLOW -->|"任务状态"| CACHE
    WORKFLOW -->|"异步事件"| MQ
    MQ -->|"审核回调 / 发布回调"| CMS
    CMS -->|"内容分发"| CHANNEL

    NOTE["分层要点<br>① 产品入口层负责操作入口，不承载生成逻辑<br>② 业务编排层负责把创作、生成、审核、发布串成主线<br>③ AI 生成层专注内容生产，审核治理层专注风险与质量兜底<br>④ 数据与基础设施层为全链路提供存储、缓存、消息和模型接入"]:::noteStyle
    NOTE -.- ORCH

    linkStyle 0,1 stroke:#374151,stroke-width:2px
    linkStyle 2,3,4,5,6 stroke:#1d4ed8,stroke-width:2px
    linkStyle 7,8,9,10 stroke:#dc2626,stroke-width:2px,stroke-dasharray:4 3
    linkStyle 11,12,13,14,15 stroke:#059669,stroke-width:2px
    linkStyle 16,17,18,19,20 stroke:#d97706,stroke-width:2px
    linkStyle 21,22 stroke:#059669,stroke-width:2px
    linkStyle 23,24 stroke:#ea580c,stroke-width:2px
    linkStyle 25 stroke:#1d4ed8,stroke-width:2px
```

**阅读重点**

- 先看中间三层：业务编排层、AI 生成层、审核治理层。
- `工作流编排` 是全链路中枢。
- `审核治理层` 是发布前闸口。
- `数据与基础设施层` 为全链路提供支撑，但不承载业务决策。

---

## 3. 分层能力结构图

这张图回答：AI 漫剧系统按哪些阶段或能力域分层展开，每层里有哪些并列能力块，以及主生产链路如何从上游推进到下游。

```mermaid
flowchart LR
    classDef entryStyle   fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef scriptStyle  fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef visualStyle  fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef videoStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef editStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef storeStyle   fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    subgraph ENTRY["入口层 Entry"]
        direction LR
        USER["用户输入<br>主题 / 故事梗概 / 角色设定"]:::entryStyle
        STUDIO["创作平台<br>Web / App<br>创作者工作台"]:::entryStyle
        API["API 接入<br>第三方调用<br>批量生成任务"]:::entryStyle
        ASSET_IN["素材导入<br>自定义角色<br>参考图 / 风格图"]:::entryStyle
    end
    class ENTRY layerStyle

    subgraph SCRIPT["剧本生成层 Script Generation"]
        direction LR
        STORY["故事规划<br>主题分析<br>世界观设定"]:::scriptStyle
        BOARD_SCRIPT["分镜脚本<br>剧情拆分<br>分镜描述生成"]:::scriptStyle
        DIALOG["台词生成<br>角色对话<br>情绪与语气控制"]:::scriptStyle
        SCENE_OUT["分镜表输出<br>Scene List<br>镜头编号 / 时长"]:::scriptStyle
    end
    class SCRIPT layerStyle

    subgraph VISUAL["视觉生成层 Visual Generation"]
        direction LR
        CHAR["角色生成<br>角色形象设计<br>表情 / 姿态示例"]:::visualStyle
        SCENE["场景生成<br>背景图生成<br>多风格场景切换"]:::visualStyle
        SHOT["分镜画面生成<br>按分镜描述生成<br>保持角色一致性"]:::visualStyle
        IMG_POST["图像后处理<br>超分辨率放大<br>风格统一 / 修复"]:::visualStyle
    end
    class VISUAL layerStyle

    subgraph VIDEO["视频合成层 Video Composition"]
        direction LR
        ANIM["动画 / 运镜生成<br>关键帧补间<br>动态镜头切换"]:::videoStyle
        TTS["语音合成 TTS<br>角色配音<br>情感语气 / 口型适配"]:::videoStyle
        MUSIC["配乐音效<br>背景音乐生成<br>环境音循环 / 增强"]:::videoStyle
        SUB["字幕与特效<br>自动字幕生成<br>滤镜特效 / 转场"]:::videoStyle
    end
    class VIDEO layerStyle

    subgraph EDIT["编辑与优化层 Editing & Optimize"]
        direction LR
        EDIT_TIME["时间轴编辑<br>分镜顺序调整<br>时长裁剪 / 节奏控制"]:::editStyle
        EDIT_IMAGE["画面微调<br>镜像翻转<br>风格滤镜 / 色彩调整"]:::editStyle
        EDIT_AUDIO["声音调节<br>音量 / 音效平衡<br>背景音乐替换"]:::editStyle
        EXPORT["一键导出<br>预览播放<br>多格式视频输出"]:::editStyle
    end
    class EDIT layerStyle

    subgraph STORE["存储与分发层 Storage & Distribution"]
        direction LR
        LIB["素材资产库<br>角色 / 场景缓存<br>可复用素材管理"]:::storeStyle
        WORK["作品管理<br>版本记录<br>草稿与成片管理"]:::storeStyle
        PUBLISH["分发发布<br>多平台发布<br>封面 / 标签 / 简介生成"]:::storeStyle
        ANALYTICS["数据分析<br>播放量 / 完播率<br>用户互动数据"]:::storeStyle
    end
    class STORE layerStyle

    ENTRY -->|"创作输入"| SCRIPT
    SCRIPT -->|"脚本与分镜"| VISUAL
    VISUAL -->|"视觉素材"| VIDEO
    VIDEO -->|"视频草稿"| EDIT
    EDIT -->|"导出成片"| STORE

    NOTE["整体主线<br>入口层 -> 剧本生成层 -> 视觉生成层 -> 视频合成层 -> 编辑与优化层 -> 存储与分发层<br><br>结构重点<br>① 每一层表示一组并列能力块，不表示服务调用拓扑<br>② 层内节点强调能力归属，不强调先后顺序<br>③ 层间箭头只表达主流程推进，不展开技术细节"]:::noteStyle
    NOTE -.- STORE
```

**阅读重点**

- 先按“层”看，不要先按“服务调用”看。
- 每一层中的节点是并列能力块，不是严格时序步骤。
- 层与层之间的箭头只表示主流程推进，不表示完整的数据流细节。

这张图和平台分层图的区别是：

- 平台分层图更偏职责边界和层间协作。
- 分层能力结构图更偏产品能力总览和阶段式组织。
- AI 漫剧这类“创作层 -> 生成层 -> 合成层 -> 分发层”的表达，更适合用分层能力结构图承载。

---

## 4. 流程图

这张图回答：AI 漫剧从立项到发布，主流程怎么走。

```mermaid
flowchart LR
    classDef userStyle     fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef createStyle   fill:#4f46e5,stroke:#3730a3,stroke-width:2px,color:#fff
    classDef genStyle      fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef reviewStyle   fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef publishStyle  fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef dbStyle       fill:#374151,stroke:#111827,stroke-width:2px,color:#fff
    classDef noteStyle     fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle    fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    USER["创作者发起项目<br>选题 / 角色 / 风格"]:::userStyle

    subgraph CREATE["创作阶段"]
        direction LR
        C1["剧本策划<br>世界观 / 分集大纲"]:::createStyle
        C2["分镜设计<br>镜头 / 画格 / 台词"]:::createStyle
        C3["素材确认<br>角色 / 场景 / 风格约束"]:::createStyle
    end
    class CREATE layerStyle

    subgraph GENERATE["生成阶段"]
        direction LR
        G1["文本生成<br>剧情扩写 / 对白润色"]:::genStyle
        G2["画面生成<br>角色 / 场景 / 分镜图"]:::genStyle
        G3["音频生成<br>配音 / 音效 / BGM"]:::genStyle
        G4["成片合成<br>字幕 / 音画拼接"]:::genStyle
    end
    class GENERATE layerStyle

    subgraph REVIEW["审核阶段"]
        direction LR
        R1["机器审核<br>安全 / 版权 / 敏感内容"]:::reviewStyle
        R2["质量校验<br>角色一致性 / 完整性"]:::reviewStyle
        R3["人工复核<br>打回 / 修订 / 放行"]:::reviewStyle
    end
    class REVIEW layerStyle

    subgraph PUBLISH["发布阶段"]
        direction LR
        P1["内容入库<br>版本封板 / 上架配置"]:::publishStyle
        P2["渠道发布<br>App / 小程序 / 视频平台"]:::publishStyle
        P3["数据回流<br>播放 / 留存 / 举报反馈"]:::publishStyle
    end
    class PUBLISH layerStyle

    STORE[("媒资与版本存档<br>脚本 / 图片 / 音频 / 成片")]:::dbStyle

    USER --> CREATE
    CREATE --> GENERATE
    GENERATE -->|"生成产物"| STORE
    GENERATE --> REVIEW
    REVIEW -->|"审核通过"| PUBLISH
    REVIEW -.->|"打回重做"| CREATE
    REVIEW -.->|"局部返工"| GENERATE
    PUBLISH -->|"成品归档"| STORE
    PUBLISH --> P3
    P3 -->|"反哺选题/风格/模板"| CREATE

    NOTE["主流程要点<br>① 创作阶段先确定剧本、分镜和风格约束<br>② 生成阶段通常并行产出文本、画面和音频，再统一合成<br>③ 审核采用机器初筛 + 人工兜底<br>④ 发布后的反馈继续影响下一轮创作"]:::noteStyle
    NOTE -.- REVIEW

    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1 stroke:#4f46e5,stroke-width:2px
    linkStyle 2,3 stroke:#dc2626,stroke-width:2px
    linkStyle 4 stroke:#d97706,stroke-width:2px
    linkStyle 5,6 stroke:#d97706,stroke-width:1.5px,stroke-dasharray:4 3
    linkStyle 7,8 stroke:#059669,stroke-width:2px
    linkStyle 9 stroke:#059669,stroke-width:2px
    linkStyle 10 stroke:#4f46e5,stroke-width:2px
```

**阅读重点**

- 主线是：创作 → 生成 → 审核 → 发布。
- 审核阶段是最大分叉点。
- 发布后的数据回流会反哺下一轮创作，不是终点。

---

## 5. 技术架构图

这张图回答：关键服务如何协作，哪些组件在关键路径上。

```mermaid
flowchart TB
    classDef clientStyle  fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef gatewayStyle fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef svcStyle     fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef aiStyle      fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef reviewStyle  fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef mqStyle      fill:#ea580c,stroke:#7c2d12,stroke-width:2px,color:#fff
    classDef dbStyle      fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px
    classDef infraStyle   fill:#fafaf9,stroke:#a8a29e,stroke-width:1.5px

    subgraph CLIENT["客户端层"]
        direction LR
        STUDIO["创作工作台"]:::clientStyle
        OPS["审核运营后台"]:::clientStyle
        CHANNEL["内容渠道"]:::clientStyle
    end
    class CLIENT layerStyle

    subgraph ACCESS["接入与控制层"]
        direction LR
        API["API Gateway<br>鉴权 / 路由 / 限流"]:::gatewayStyle
        PROJECT["项目服务<br>作品 / 章节 / 版本"]:::gatewayStyle
        CMS["内容 CMS<br>入库 / 封板 / 发布配置"]:::gatewayStyle
    end
    class ACCESS layerStyle

    subgraph SERVICES["业务与编排层"]
        direction LR
        ORCH["工作流编排服务<br>任务拆解 / 重试 / 调度"]:::svcStyle
        PROMPT["Prompt / 模板服务<br>角色卡 / 风格模板"]:::svcStyle
        ASSET["素材资产服务<br>角色 / 场景 / 画风"]:::svcStyle
        FEEDBACK["反馈分析服务<br>播放 / 留存 / 举报"]:::svcStyle
    end
    class SERVICES layerStyle

    subgraph AI["AI 生成层"]
        direction LR
        TEXT["文本生成服务"]:::aiStyle
        BOARD["分镜生成服务"]:::aiStyle
        IMAGE["图像生成服务"]:::aiStyle
        AUDIO["音频生成服务"]:::aiStyle
        RENDER["成片合成服务"]:::aiStyle
    end
    class AI layerStyle

    subgraph REVIEW["审核治理层"]
        direction LR
        SAFETY["机器审核服务"]:::reviewStyle
        QUALITY["质量校验服务"]:::reviewStyle
        MANUAL["人工复核服务"]:::reviewStyle
    end
    class REVIEW layerStyle

    subgraph INFRA["基础设施层"]
        direction LR
        MODEL[("模型网关<br>LLM / 图像 / TTS")]:::dbStyle
        MQ[("消息队列<br>任务 / 回调 / 事件")]:::mqStyle
        CACHE[("缓存<br>会话态 / 热数据")]:::mqStyle
        DB[("业务数据库<br>项目 / 审核 / 发布记录")]:::dbStyle
        OSS[("对象存储<br>脚本 / 图片 / 音频 / 成片")]:::dbStyle
    end
    class INFRA infraStyle

    STUDIO -->|"HTTPS 请求"| API
    OPS -->|"HTTPS 请求"| API
    API -->|"项目管理"| PROJECT
    API -->|"发布配置"| CMS
    API -->|"生成任务"| ORCH

    PROJECT -->|"读取项目上下文"| ORCH
    PROMPT -->|"提示模板"| ORCH
    ASSET -->|"角色/场景约束"| ORCH

    ORCH -->|"调用文本能力"| TEXT
    ORCH -->|"调用分镜能力"| BOARD
    ORCH -->|"调用图像能力"| IMAGE
    ORCH -->|"调用音频能力"| AUDIO
    ORCH -->|"触发合成"| RENDER

    TEXT -->|"模型调用"| MODEL
    BOARD -->|"模型调用"| MODEL
    IMAGE -->|"模型调用"| MODEL
    AUDIO -->|"模型调用"| MODEL

    TEXT -->|"脚本产物"| OSS
    BOARD -->|"分镜产物"| OSS
    IMAGE -->|"图像产物"| OSS
    AUDIO -->|"音频产物"| OSS
    RENDER -->|"视频成片"| OSS

    RENDER -->|"送审"| SAFETY
    SAFETY -->|"通过后质检"| QUALITY
    SAFETY -->|"高风险升级"| MANUAL
    QUALITY -->|"通过稿件"| CMS
    MANUAL -->|"修订/放行结果"| CMS

    ORCH -->|"异步任务事件"| MQ
    MQ -->|"审核/发布回调"| CMS
    ORCH -->|"任务态缓存"| CACHE
    PROJECT -->|"项目元数据"| DB
    CMS -->|"审核/发布记录"| DB
    CHANNEL -->|"消费内容"| FEEDBACK
    FEEDBACK -->|"优化建议"| PROJECT

    NOTE["技术协作要点<br>① 工作流编排服务是全链路调度中枢<br>② Prompt 与素材服务为多模型生成提供统一约束<br>③ 审核治理层作为发布前闸口，与 CMS 紧耦合<br>④ MQ、缓存、对象存储共同支撑异步生成与大文件产物管理"]:::noteStyle
    NOTE -.- SERVICES
```

**阅读重点**

- `工作流编排服务` 是技术中枢。
- `Prompt / 模板服务` 和 `素材资产服务` 为生成提供统一约束。
- `CMS + 审核治理层` 共同构成发布控制面。
- `对象存储 + MQ + 缓存` 是高频基础设施组合。

---

## 6. 状态图

这张图回答：作品或章节在生产链路中会经历哪些状态，什么事件会推进或回退。

```mermaid
stateDiagram-v2
    [*] --> Draft: 创建项目/章节

    Draft --> Creating: 提交创作信息
    Creating --> ReadyToGenerate: 剧本/分镜/素材确认

    ReadyToGenerate --> Generating: 发起生成任务
    Generating --> GenerateFailed: 模型失败/超时/资源不足
    Generating --> Generated: 文本/图像/音频/成片生成完成

    GenerateFailed --> ReadyToGenerate: 调整参数后重试
    Generated --> Reviewing: 提交审核

    Reviewing --> ReviewRejected: 机器审核拦截
    Reviewing --> ManualReview: 高风险/低质量升级人工
    Reviewing --> Approved: 机器审核通过

    ManualReview --> ReviewRejected: 人工打回
    ManualReview --> Approved: 人工放行

    ReviewRejected --> Creating: 重写剧本/分镜/素材
    Approved --> ReadyToPublish: 进入发布配置

    ReadyToPublish --> Publishing: 发起渠道发布
    Publishing --> PublishFailed: 渠道失败/回调异常
    Publishing --> Published: 渠道发布成功

    PublishFailed --> ReadyToPublish: 修正配置后重发
    Published --> Archived: 下线/归档
    Archived --> [*]
```

**阅读重点**

- 主线状态是：`Draft → Creating → ReadyToGenerate → Generating → Generated → Reviewing → Approved → ReadyToPublish → Publishing → Published`
- `Reviewing` 是最大分流状态。
- `GenerateFailed` 和 `PublishFailed` 是系统常见失败回路。

---

## 7. 总结

这 5 张图分别回答不同问题：

- 平台分层图：系统分几层，每层负责什么
- 分层能力结构图：系统按哪些阶段或能力域分层展开
- 流程图：创作到发布主线如何推进
- 技术架构图：关键服务如何协作
- 状态图：作品生命周期如何迁移

如果后续继续扩展，优先建议补两类图：

1. `时序图`
   适合看“一次生成任务谁先调谁”
2. `模块依赖图`
   适合看“编排模块、审核模块、CMS 模块之间的代码依赖关系”
