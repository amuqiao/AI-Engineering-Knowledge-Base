**LSTM和BiLSTM模型架构图**（序列建模经典模型，严格贴合核心机制：**门控机制、记忆单元、双向信息融合**），风格和全套深度学习架构完全统一，可直接用于笔记/PPT。

# LSTM 完整架构流程图（基础版）

```mermaid
flowchart LR
    %% 统一样式（和Transformer/TimesNet/YOLO/SimCLR风格一致）
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px,rounded:8px
    classDef embedStyle fill:#ffd700,stroke:#333,stroke-width:2px,rounded:8px
    classDef blockStyle fill:#9ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef gateStyle fill:#9f9,stroke:#333,stroke-width:2px,rounded:8px
    classDef cellStyle fill:#ff9,stroke:#333,stroke-width:2px,rounded:8px
    classDef headStyle fill:#f2e6ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px,rounded:10px
    classDef coreNoteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px,rounded:8px

    %% 1. 输入层
    subgraph InputLayer["输入层"]
        A["输入序列<br/>Input Sequence"]:::inputStyle
    end
    class InputLayer subgraphStyle

    %% 2. 嵌入层
    subgraph EmbedLayer["嵌入层"]
        B["Token Embedding<br/>词元嵌入"]:::embedStyle
    end
    class EmbedLayer subgraphStyle

    %% 3. LSTM层
    subgraph LSTMLayer["LSTM层（N层堆叠）"]
        %% 单个LSTM单元
        subgraph LSTMCell["LSTM单元"]
            C["遗忘门<br/>Forget Gate"]:::gateStyle
            D["输入门<br/>Input Gate"]:::gateStyle
            E["候选记忆<br/>Candidate Cell"]:::cellStyle
            F["输出门<br/>Output Gate"]:::gateStyle
            G["记忆单元<br/>Cell State"]:::cellStyle
            H["隐藏状态<br/>Hidden State"]:::blockStyle
        end
        class LSTMCell subgraphStyle
    end
    class LSTMLayer subgraphStyle

    %% 4. 输出层
    subgraph OutputLayer["输出层"]
        I["线性层<br/>Linear Layer"]:::headStyle
        J["预测结果<br/>Prediction Output"]:::inputStyle
    end
    class OutputLayer subgraphStyle

    %% 核心注释（LSTM创新点）
    Note["LSTM 核心创新：<br/>1. 门控机制（遗忘门、输入门、输出门）<br/>2. 记忆单元（长期记忆）<br/>3. 隐藏状态（短期记忆）<br/>4. 解决RNN梯度消失问题<br/>5. 适合处理长序列依赖"]:::coreNoteStyle

    %% 数据流
    A --> B
    B --> C
    B --> D
    B --> E
    H --> C
    H --> D
    H --> F
    C --> G
    D --> E
    E --> G
    G --> F
    F --> H
    H --> I
    I --> J

    %% 连接线样式
    linkStyle 0,1 stroke:#ff7043,stroke-width:1.5px
    linkStyle 2,3,4,5,6,7,8,9 stroke:#4299e1,stroke-width:2px
    linkStyle 10 stroke:#9c27b0,stroke-width:2px

    %% 可选：将注释节点定位到合适位置（提升可读性）
    Note -.-> LSTMCell
```

# BiLSTM 完整架构流程图（基础版）

```mermaid
flowchart LR
    %% 统一样式（和Transformer/TimesNet/YOLO/SimCLR风格一致）
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px,rounded:8px
    classDef embedStyle fill:#ffd700,stroke:#333,stroke-width:2px,rounded:8px
    classDef blockStyle fill:#9ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef gateStyle fill:#9f9,stroke:#333,stroke-width:2px,rounded:8px
    classDef cellStyle fill:#ff9,stroke:#333,stroke-width:2px,rounded:8px
    classDef headStyle fill:#f2e6ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px,rounded:10px
    classDef coreNoteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px,rounded:8px

    %% 1. 输入层
    subgraph InputLayer["输入层"]
        A["输入序列<br/>Input Sequence"]:::inputStyle
    end
    class InputLayer subgraphStyle

    %% 2. 嵌入层
    subgraph EmbedLayer["嵌入层"]
        B["Token Embedding<br/>词元嵌入"]:::embedStyle
    end
    class EmbedLayer subgraphStyle

    %% 3. BiLSTM层
    subgraph BiLSTMLayer["BiLSTM层（N层堆叠）"]
        %% 前向LSTM
        subgraph ForwardLSTM["前向LSTM"]
            C1["遗忘门<br/>Forget Gate"]:::gateStyle
            D1["输入门<br/>Input Gate"]:::gateStyle
            E1["候选记忆<br/>Candidate Cell"]:::cellStyle
            F1["输出门<br/>Output Gate"]:::gateStyle
            G1["记忆单元<br/>Cell State"]:::cellStyle
            H1["前向隐藏状态<br/>Forward Hidden State"]:::blockStyle
        end
        class ForwardLSTM subgraphStyle

        %% 后向LSTM
        subgraph BackwardLSTM["后向LSTM（从序列末尾到开头）"]
            C2["遗忘门<br/>Forget Gate"]:::gateStyle
            D2["输入门<br/>Input Gate"]:::gateStyle
            E2["候选记忆<br/>Candidate Cell"]:::cellStyle
            F2["输出门<br/>Output Gate"]:::gateStyle
            G2["记忆单元<br/>Cell State"]:::cellStyle
            H2["后向隐藏状态<br/>Backward Hidden State"]:::blockStyle
        end
        class BackwardLSTM subgraphStyle

        %% 融合层
        I["隐藏状态融合<br/>Hidden State Concatenation"]:::blockStyle
    end
    class BiLSTMLayer subgraphStyle

    %% 4. 输出层
    subgraph OutputLayer["输出层"]
        J["线性层<br/>Linear Layer"]:::headStyle
        K["预测结果<br/>Prediction Output"]:::inputStyle
    end
    class OutputLayer subgraphStyle

    %% 核心注释（BiLSTM创新点）
    Note["BiLSTM 核心创新：<br/>1. 双向信息融合<br/>2. 同时捕获上下文信息<br/>3. 前向LSTM处理过去信息<br/>4. 后向LSTM处理未来信息<br/>5. 更全面的序列表示"]:::coreNoteStyle

    %% 数据流
    A --> B
    B --> C1
    B --> D1
    B --> E1
    H1 --> C1
    H1 --> D1
    H1 --> F1
    C1 --> G1
    D1 --> E1
    E1 --> G1
    G1 --> F1
    F1 --> H1

    B --> C2
    B --> D2
    B --> E2
    H2 --> C2
    H2 --> D2
    H2 --> F2
    C2 --> G2
    D2 --> E2
    E2 --> G2
    G2 --> F2
    F2 --> H2

    H1 --> I
    H2 --> I
    I --> J
    J --> K

    %% 连接线样式
    linkStyle 0,1 stroke:#ff7043,stroke-width:1.5px
    linkStyle 2,3,4,5,6,7,8,9,10,11,12,13,14,15 stroke:#4299e1,stroke-width:2px
    linkStyle 16,17 stroke:#9c27b0,stroke-width:2px

    %% 可选：将注释节点定位到合适位置（提升可读性）
    Note -.-> BiLSTMLayer
```

---

# LSTM 和 BiLSTM 极简核心总结

## LSTM
1. **定位**：**序列建模**经典模型，解决RNN梯度消失问题
2. **核心Backbone**：**门控机制**结构，包含遗忘门、输入门、输出门
3. **最大创新**
    - **门控机制**：控制信息的流入和流出
    - **记忆单元**：长期记忆存储
    - **隐藏状态**：短期记忆传递
    - **解决梯度消失**：适合处理长序列
4. **结构范式**
输入 → 嵌入 → LSTM单元（门控机制+记忆单元）→ 线性层 → 输出

## BiLSTM
1. **定位**：**双向序列建模**增强版，捕获上下文信息
2. **核心Backbone**：**双向LSTM**结构，包含前向和后向LSTM
3. **最大创新**
    - **双向信息融合**：同时处理过去和未来信息
    - **更全面的上下文**：捕获完整的序列语义
    - **隐藏状态拼接**：丰富特征表示
4. **结构范式**
输入 → 嵌入 → 前向LSTM + 后向LSTM → 隐藏状态融合 → 线性层 → 输出
