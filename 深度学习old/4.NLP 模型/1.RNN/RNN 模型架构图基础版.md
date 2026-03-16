**标准 RNN 模型架构图**（循环神经网络基础版，核心：**循环连接、隐藏状态**），风格和项目全套深度学习架构完全统一，可直接用于笔记/PPT。

# RNN 完整架构流程图（基础版）

```mermaid
flowchart LR
    %% 统一样式（优化整合）
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px,rounded:8px
    classDef blockStyle fill:#9ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef hiddenStyle fill:#ff9,stroke:#333,stroke-width:2px,rounded:8px
    classDef headStyle fill:#f2e6ff,stroke:#333,stroke-width:2px,rounded:8px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px,rounded:10px

    %% 1. 输入层
    subgraph InputLayer["输入层"]
        InputXt["时间步t输入<br/>Input x_t"]:::inputStyle
    end
    class InputLayer subgraphStyle

    %% 2. RNN层（优化整合版）
    subgraph RNNLayer["RNN层（基础单元）"]
        subgraph RNNCell["RNN核心单元"]
            PrevHiddenState["历史隐藏状态<br/>h_{t-1}"]:::hiddenStyle
            InputLinearTransform["输入线性变换<br/>W_hx·x_t"]:::blockStyle
            HiddenLinearTransform["隐藏状态变换<br/>W_hh·h_{t-1}"]:::blockStyle
            SumWithBias["线性组合+偏置<br/>W_hx·x_t + W_hh·h_{t-1} + b_h"]:::blockStyle
            TanhActivation["激活函数tanh<br/>tanh(SumWithBias)"]:::hiddenStyle
            CurrentHiddenState["当前隐藏状态<br/>h_t"]:::hiddenStyle
        end
        class RNNCell subgraphStyle
    end
    class RNNLayer subgraphStyle

    %% 3. 输出层
    subgraph OutputLayer["输出层"]
        OutputLinearLayer["输出线性层<br/>W_oh·h_t + b_o"]:::headStyle
        SoftmaxLayer["Softmax<br/>概率分布 ŷ_t"]:::headStyle
        PredictionResult["预测结果（argmax）<br/>y_t"]:::headStyle
    end
    class OutputLayer subgraphStyle

    %% 数据流（明确清晰）
    InputXt --> InputLinearTransform         
    PrevHiddenState --> HiddenLinearTransform         
    InputLinearTransform --> SumWithBias         
    HiddenLinearTransform --> SumWithBias         
    SumWithBias --> TanhActivation         
    TanhActivation --> CurrentHiddenState         
    CurrentHiddenState --> OutputLinearLayer         
    OutputLinearLayer --> SoftmaxLayer         
    SoftmaxLayer --> PredictionResult         
    CurrentHiddenState --> PrevHiddenState         

    %% 核心注释（完整准确）
    Note["RNN核心特点：<br/>1. 循环连接传递时序信息<br/>2. 隐藏状态存储序列上下文<br/>3. 天然捕获序列位置依赖<br/>4. 参数共享：所有时间步复用同一套权重<br/>5. 结构简洁：基础架构简单直观<br/>局限性：<br/>1. 易出现梯度消失/爆炸<br/>2. 长序列依赖建模能力弱"]
    class Note fill:#fff8e6,stroke:#ffb74d,stroke-width:1px,rounded:8px
    Note -.-> RNNLayer  

    %% 连接线样式（清晰区分）
    linkStyle 0,1 stroke:#ff7043,stroke-width:1.5px  
    linkStyle 2,3,4,5,6,7,8,9 stroke:#4299e1,stroke-width:2px  
    linkStyle 10 stroke:#888,stroke-width:1px,stroke-dasharray:5,5
```

---

# RNN 极简核心总结

1. **定位**：**循环神经网络**基础模型，处理序列数据的经典架构
2. **核心Backbone**：**循环连接机制**，包含输入层、RNN核心单元和输出层
3. **最大创新**
    - **循环连接**：隐藏状态的循环更新，传递时序信息
    - **序列建模**：天然捕获序列数据的位置依赖关系
    - **上下文存储**：隐藏状态存储序列的历史上下文信息
    - **参数共享**：所有时间步复用同一套权重参数
    - **结构简洁**：基础架构简单直观，易于理解
4. **结构范式**
输入 → 输入线性变换 → 隐藏状态变换 → 线性组合+偏置 → 激活函数tanh → 当前隐藏状态 → 输出线性层 → Softmax → 预测结果
5. **核心公式**
    - 隐藏状态更新：h_t = tanh(W_hh · h_{t-1} + W_hx · x_t + b_h)
    - 输出计算：y_t = W_oh · h_t + b_o