**标准 GRU 模型架构图**（循环神经网络变体，核心：**门控机制、更新门、重置门**），风格和项目全套深度学习架构完全统一，可直接用于笔记/PPT。

# GRU 完整架构流程图（基础版）

```mermaid
flowchart LR
    %% 统一样式
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px,rounded:8px
    classDef embedStyle fill:#ffd700,stroke:#333,stroke-width:2px,rounded:8px
    classDef blockStyle fill:#9ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef gateStyle fill:#9f9,stroke:#333,stroke-width:2px,rounded:8px
    classDef hiddenStyle fill:#ff9,stroke:#333,stroke-width:2px,rounded:8px
    classDef headStyle fill:#f2e6ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px,rounded:10px

    %% 1. 输入层
    subgraph InputLayer["输入层"]
        A["输入序列<br/>Input Sequence"]:::inputStyle
    end
    class InputLayer subgraphStyle

    %% 2. 嵌入层
    subgraph EmbedLayer["嵌入层"]
        B["Token Embedding<br/>词元嵌入"]:::embedStyle
        C["Positional Encoding<br/>位置编码"]:::embedStyle
        D["Embedding Sum<br/>特征融合"]:::embedStyle
    end
    class EmbedLayer subgraphStyle

    %% 3. GRU层
    subgraph GRULayer["GRU层（N层堆叠）"]
        subgraph GRUCell["GRU单元"]
            E["重置门<br/>Reset Gate"]:::gateStyle
            F["更新门<br/>Update Gate"]:::gateStyle
            G["候选隐藏状态<br/>Candidate Hidden State"]:::hiddenStyle
            H["隐藏状态更新<br/>Hidden State Update"]:::blockStyle
            P["历史隐藏状态<br/>Previous Hidden State"]:::hiddenStyle
        end
        class GRUCell subgraphStyle
    end
    class GRULayer subgraphStyle

    %% 4. 输出层
    subgraph OutputLayer["输出层"]
        I["线性层<br/>Linear Layer"]:::headStyle
        J["Softmax<br/>概率分布"]:::headStyle
        K["预测结果<br/>Prediction Output"]:::inputStyle
    end
    class OutputLayer subgraphStyle

    %% 数据流
    A --> B
    B --> D
    A --> C
    C --> D
    D --> E
    D --> F
    D --> G
    P --> E
    P --> F
    P --> H
    E --> G
    F --> H
    G --> H
    H --> I
    I --> J
    J --> K
    H --> P  

    %% 核心注释
    Note["GRU核心创新：<br/>1.门控机制控制信息流动<br/>2.更新门保留历史信息<br/>3.重置门忽略历史信息<br/>4.简化LSTM减少参数<br/>5.缓解梯度消失问题"]
    class Note fill:#fff8e6,stroke:#ffb74d,stroke-width:1px,rounded:8px
    Note -.-> GRULayer

    %% 连接线样式
    linkStyle 0,1,2,3 stroke:#ff7043,stroke-width:1.5px
    linkStyle 4,5,6,7,8,9,10,11,12 stroke:#4299e1,stroke-width:2px
    linkStyle 13,14,15 stroke:#9c27b0,stroke-width:2px
    linkStyle 16 stroke:#888,stroke-width:1px,stroke-dasharray:5,5
```
---

# GRU 极简核心总结

1. **定位**：**循环神经网络**变体，解决长序列依赖和梯度消失问题
2. **核心Backbone**：**门控循环单元**，包含重置门、更新门和历史隐藏状态
3. **最大创新**
    - **门控机制**：通过更新门和重置门控制信息流动
    - **简化结构**：相比LSTM减少了一个门和记忆单元
    - **参数效率**：参数数量少于LSTM，训练更快
    - **梯度传播**：缓解了RNN的梯度消失问题
    - **循环连接**：隐藏状态的循环更新机制
4. **结构范式**
输入 → 嵌入+位置编码 → GRU层（重置门+更新门+候选隐藏状态+历史隐藏状态）→ 线性层+Softmax → 输出
5. **核心公式**
    - 重置门：r_t = σ(W_r · [h_{t-1}, x_t])
    - 更新门：z_t = σ(W_z · [h_{t-1}, x_t])
    - 候选隐藏状态：ĥ_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])
    - 隐藏状态更新：h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t