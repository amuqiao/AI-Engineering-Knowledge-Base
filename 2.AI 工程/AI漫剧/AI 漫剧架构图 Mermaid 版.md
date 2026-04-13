# AI 漫剧架构图 Mermaid 版

## 说明

本文件基于原图 [AI 漫剧架构图.jpg](./AI%20漫剧架构图.jpg) 改写为 Mermaid 版本。

原图的表达方式本质上是“层级结构图 + 主流程贯穿”，不是技术调用架构图。

这里保留同样的核心结构：

- 入口层
- 剧本生成层
- 视觉生成层
- 视频合成层
- 编辑与优化层
- 存储与分发层

同时保留原图的主链路：`入口层 -> 剧本生成层 -> 视觉生成层 -> 视频合成层 -> 编辑与优化层 -> 存储与分发层`

## Mermaid 图

```mermaid
flowchart LR
    %% ── 配色定义 ───────────────────────────────────────────────
    classDef entryStyle   fill:#1f2937,stroke:#111827,stroke-width:2px,color:#f9fafb
    classDef scriptStyle  fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef visualStyle  fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef videoStyle   fill:#dc2626,stroke:#991b1b,stroke-width:2px,color:#fff
    classDef editStyle    fill:#d97706,stroke:#92400e,stroke-width:2px,color:#fff
    classDef storeStyle   fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle    fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle   fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    %% ── 入口层 ─────────────────────────────────────────────────
    subgraph ENTRY["入口层 Entry"]
        direction LR
        USER["用户输入<br>主题 / 故事梗概 / 角色设定"]:::entryStyle
        STUDIO["创作平台<br>Web / App<br>创作者工作台"]:::entryStyle
        API["API 接入<br>第三方调用<br>批量生成任务"]:::entryStyle
        ASSET_IN["素材导入<br>自定义角色<br>参考图 / 风格图"]:::entryStyle
    end
    class ENTRY layerStyle

    %% ── 剧本生成层 ─────────────────────────────────────────────
    subgraph SCRIPT["剧本生成层 Script Generation"]
        direction LR
        STORY["故事规划<br>主题分析<br>世界观设定"]:::scriptStyle
        BOARD_SCRIPT["分镜脚本<br>剧情拆分<br>分镜描述生成"]:::scriptStyle
        DIALOG["台词生成<br>角色对话<br>情绪与语气控制"]:::scriptStyle
        SCENE_OUT["分镜表输出<br>Scene List<br>镜头编号 / 时长"]:::scriptStyle
    end
    class SCRIPT layerStyle

    %% ── 视觉生成层 ─────────────────────────────────────────────
    subgraph VISUAL["视觉生成层 Visual Generation"]
        direction LR
        CHAR["角色生成<br>角色形象设计<br>表情 / 姿态示例"]:::visualStyle
        SCENE["场景生成<br>背景图生成<br>多风格场景切换"]:::visualStyle
        SHOT["分镜画面生成<br>按分镜描述生成<br>保持角色一致性"]:::visualStyle
        IMG_POST["图像后处理<br>超分辨率放大<br>风格统一 / 修复"]:::visualStyle
    end
    class VISUAL layerStyle

    %% ── 视频合成层 ─────────────────────────────────────────────
    subgraph VIDEO["视频合成层 Video Composition"]
        direction LR
        ANIM["动画 / 运镜生成<br>关键帧补间<br>动态镜头切换"]:::videoStyle
        TTS["语音合成 TTS<br>角色配音<br>情感语气 / 口型适配"]:::videoStyle
        MUSIC["配乐音效<br>背景音乐生成<br>环境音循环 / 增强"]:::videoStyle
        SUB["字幕与特效<br>自动字幕生成<br>滤镜特效 / 转场"]:::videoStyle
    end
    class VIDEO layerStyle

    %% ── 编辑与优化层 ───────────────────────────────────────────
    subgraph EDIT["编辑与优化层 Editing & Optimize"]
        direction LR
        EDIT_TIME["时间轴编辑<br>分镜顺序调整<br>时长裁剪 / 节奏控制"]:::editStyle
        EDIT_IMAGE["画面微调<br>镜像翻转<br>风格滤镜 / 色彩调整"]:::editStyle
        EDIT_AUDIO["声音调节<br>音量 / 音效平衡<br>背景音乐替换"]:::editStyle
        EXPORT["一键导出<br>预览播放<br>多格式视频输出"]:::editStyle
    end
    class EDIT layerStyle

    %% ── 存储与分发层 ───────────────────────────────────────────
    subgraph STORE["存储与分发层 Storage & Distribution"]
        direction LR
        LIB["素材资产库<br>角色 / 场景缓存<br>可复用素材管理"]:::storeStyle
        WORK["作品管理<br>版本记录<br>草稿与成片管理"]:::storeStyle
        PUBLISH["分发发布<br>多平台发布<br>封面 / 标签 / 简介生成"]:::storeStyle
        ANALYTICS["数据分析<br>播放量 / 完播率<br>用户互动数据"]:::storeStyle
    end
    class STORE layerStyle

    %% ── 主链路：按层级向下推进 ─────────────────────────────────
    ENTRY -->|"创作输入"| SCRIPT
    SCRIPT -->|"脚本与分镜"| VISUAL
    VISUAL -->|"视觉素材"| VIDEO
    VIDEO -->|"视频草稿"| EDIT
    EDIT -->|"导出成片"| STORE

    %% ── 注记 ───────────────────────────────────────────────────
    NOTE["整体主线<br>入口层 -> 剧本生成层 -> 视觉生成层 -> 视频合成层 -> 编辑与优化层 -> 存储与分发层<br><br>结构重点<br>① 每一层表示一组并列能力块，不表示服务调用拓扑<br>② 层内节点强调能力归属，不强调先后顺序<br>③ 层间箭头只表达主流程推进，不展开技术细节"]:::noteStyle
    NOTE -.- STORE

    %% 边索引：0-5，共 6 条
    linkStyle 0,1,2,3,4 stroke:#374151,stroke-width:2px
    linkStyle 5 stroke:#1f2937,stroke-width:2px
```

## 图解说明

这张图回答的是：AI 漫剧系统通常按哪些能力层来组织，以及主生产链路如何自上而下推进。

建议先看纵向层级：

`入口层 -> 剧本生成层 -> 视觉生成层 -> 视频合成层 -> 编辑与优化层 -> 存储与分发层`

这是原图最核心的层级结构。

最关键的理解方式有 3 点：

- 先按“层”看，不要先按“服务调用”看。
- 每一层中的节点是并列能力块，不是严格时序步骤。
- 层与层之间的箭头只表示主流程推进，不表示完整的数据流细节。

## 没覆盖的内容

这张图主要保留了原图的“层级结构表达”，没有展开以下内容：

- 具体技术服务之间的调用关系
- 审核、风控、内容安全链路
- 模型网关、消息队列、对象存储等基础设施细节
- 单次生成任务的时序和状态迁移

如果后续需要，可以继续补两张图：

1. `技术架构图`
   - 展开服务之间的调用关系、存储和异步链路
2. `流程图`
   - 展开一次 AI 漫剧生成任务的详细处理步骤
