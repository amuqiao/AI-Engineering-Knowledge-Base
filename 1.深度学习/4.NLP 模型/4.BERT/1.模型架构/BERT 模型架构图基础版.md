# BERT 模型架构图基础版

## 📝 基础版架构概览

### 核心组件

```mermaid
flowchart LR
    %% 样式定义
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef embStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef encoderStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef outputStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px,rounded:10px
    
    %% 输入层（修复：节点文本加双引号，避免特殊字符解析错误）
    subgraph inputLayer["输入层"]
        A["Token输入<br/>(序列长度: L)"]:::inputStyle
    end
    class inputLayer subgraphStyle
    
    %% 嵌入层（所有节点文本加双引号）
    subgraph embeddingLayer["嵌入层"]
        B["Token Embedding<br/>(维度: L × H)"]:::embStyle
        C["Position Embedding<br/>(维度: L × H)"]:::embStyle
        D["Segment Embedding<br/>(维度: L × H)"]:::embStyle
        E["Embedding 总和<br/>(维度: L × H)"]:::embStyle
    end
    class embeddingLayer subgraphStyle
    
    %% Transformer Encoder（参考堆叠表示方式）
    subgraph encoderLayer["编码器（N层编码器层堆叠）"]
        %% 单个编码器层
        subgraph encoderBlock["编码器层"]
            F["Multi-Head Attention<br/>(维度: L × H)"]:::encoderStyle
            G["Add & Norm<br/>(维度: L × H)"]:::encoderStyle
            H["Feed Forward Network<br/>(维度: L × H)"]:::encoderStyle
            I["Add & Norm<br/>(维度: L × H)"]:::encoderStyle
        end
    end
    class encoderLayer subgraphStyle
    class encoderBlock subgraphStyle
    
    %% 输出层（修复节点文本引号）
    subgraph outputLayer["输出层"]
        J["[CLS] 输出<br/>(维度: H)"]:::outputStyle
        K["Token 输出<br/>(维度: L × H)"]:::outputStyle
    end
    class outputLayer subgraphStyle
    
    %% 数据流转（共11条连接线，索引0-10）
    A --> B
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I -->|循环 N 次| F
    I --> J
    I --> K
    
    %% 连接线样式（精准匹配索引）
    linkStyle 0,1,2,3,4,5,6,7,8,9,10 stroke:#666,stroke-width:1.5px,arrowheadStyle:filled
```


### 关键节点说明

1. **输入层**：接收token序列，长度为L
2. **嵌入层**：
   - Token Embedding：将token映射到高维空间
   - Position Embedding：编码位置信息
   - Segment Embedding：区分不同句子
   - 三者相加得到最终嵌入表示
3. **Transformer Encoder**：
   - 堆叠N层（BERT-base: 12层，BERT-large: 24层）
   - 每层包含Multi-Head Attention和Feed Forward Network
   - 每层都有Add & Norm操作
4. **输出层**：
   - [CLS]位置输出：用于分类任务
   - 所有token输出：用于序列标注等任务

### 关键参数
- **隐藏层维度 (H)**：BERT-base: 768，BERT-large: 1024
- **注意力头数**：BERT-base: 12，BERT-large: 16
- **序列长度 (L)**：最大512
- **层数 (N)**：BERT-base: 12，BERT-large: 24