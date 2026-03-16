# RAG 全链路优化设计与实现指南
# 向量数据库 · 重排序 · 混合搜索

> 本文系统讲解 RAG（Retrieval-Augmented Generation）全链路的核心优化技术，覆盖向量数据库选型、重排序提升精度、混合搜索融合策略三大方向，结合原理、代码示例与工程实践，帮助读者构建高质量的知识检索系统。

---

## 目录

1. [RAG 全链路概览](#一rag-全链路概览)
2. [向量数据库](#二向量数据库)
3. [重排序（Reranking）](#三重排序reranking)
4. [混合搜索（Hybrid Search）](#四混合搜索hybrid-search)
5. [三项技术的协同整合](#五三项技术的协同整合)
6. [面试常见问题（FAQ）](#六面试常见问题faq)

---

## 一、RAG 全链路概览

### 1.1 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识库与大语言模型结合的技术范式：在生成答案之前，先从知识库中检索最相关的文档片段，再将其注入 LLM 的上下文，从而让模型回答基于真实数据而非幻觉。

### 1.2 全链路流程图

```mermaid
flowchart TD
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef indexStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef retrieveStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef rerankStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef generateStyle fill:#ff9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef coreStyle fill:#e8f4f8,stroke:#4299e1,stroke-width:1.5px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    %% 离线索引阶段
    subgraph indexPhase["① 离线索引阶段（构建知识库）"]
        D1[原始文档<br/>PDF / Word / HTML]:::inputStyle
        D2[文档切分<br/>Chunking]:::indexStyle
        D3[向量化<br/>Embedding Model]:::indexStyle
        D4[写入向量数据库<br/>+ 倒排索引]:::indexStyle
    end
    class indexPhase subgraphStyle

    %% 在线检索阶段
    subgraph retrievePhase["② 在线检索阶段（混合搜索）"]
        Q[用户问题]:::inputStyle
        R1[向量检索<br/>ANN Search]:::retrieveStyle
        R2[关键词检索<br/>BM25 / TF-IDF]:::retrieveStyle
        R3[融合排序<br/>RRF / 加权]:::retrieveStyle
    end
    class retrievePhase coreStyle

    %% 重排序阶段
    subgraph rerankPhase["③ 重排序阶段（精排）"]
        RR1[Cross-Encoder<br/>精细评分]:::rerankStyle
        RR2[Top-K 截取<br/>选最相关片段]:::rerankStyle
    end
    class rerankPhase subgraphStyle

    %% 生成阶段
    subgraph generatePhase["④ 生成阶段"]
        G1[Prompt 组装<br/>问题 + 上下文]:::generateStyle
        G2[LLM 生成<br/>最终答案]:::generateStyle
    end
    class generatePhase subgraphStyle

    D1 --> D2 --> D3 --> D4
    Q --> R1 & R2
    R1 & R2 --> R3 --> RR1 --> RR2 --> G1 --> G2

    linkStyle 0,1,2 stroke:#666,stroke-width:1.5px
    linkStyle 3,4,5,6,7,8,9 stroke:#333,stroke-width:1.5px

    Note["优化重点：<br/>向量数据库决定检索天花板<br/>混合搜索扩大召回覆盖<br/>重排序提升最终精度"]:::noteStyle
    Note -.-> rerankPhase
```

### 1.3 三项技术的分工

| 技术 | 所处阶段 | 解决的问题 | 优化目标 |
|------|---------|---------|---------|
| **向量数据库** | 索引 + 检索 | 如何高效存储和查询语义向量 | 检索速度与精度的平衡 |
| **混合搜索** | 检索 | 单一检索方式召回不足 | 提高召回率（Recall） |
| **重排序** | 检索后处理 | 粗排结果顺序不准确 | 提高精确率（Precision） |

---

## 二、向量数据库

### 2.1 基本原理

#### 2.1.1 为什么需要向量数据库

传统关系型数据库基于精确匹配（`WHERE name = 'xxx'`），无法处理语义相似性查询。向量数据库的核心能力是：**在高维向量空间中，快速找到与查询向量最相近的 K 个向量**（KNN / ANN 搜索）。

文本经过 Embedding 模型转化为稠密向量后，语义相似的文本在向量空间中距离更近：

```
"苹果手机" → [0.12, 0.87, -0.34, ...]  ← 距离近
"iPhone"  → [0.15, 0.83, -0.31, ...]  ← 距离近
"香蕉"    → [-0.67, 0.21, 0.89, ...]  ← 距离远
```

#### 2.1.2 相似度度量方式

```mermaid
flowchart TD
    classDef metricStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef useStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    subgraph cosine["余弦相似度（Cosine Similarity）"]
        C1[衡量向量夹角<br/>值域 -1 ~ 1]:::metricStyle
        C2[✅ 文本语义匹配<br/>✅ 归一化向量场景]:::useStyle
    end
    class cosine subgraphStyle

    subgraph dot["点积（Dot Product）"]
        D1[向量模长 × 余弦<br/>考虑幅度信息]:::metricStyle
        D2[✅ OpenAI Embedding<br/>✅ 已归一化的模型]:::useStyle
    end
    class dot subgraphStyle

    subgraph euclidean["欧氏距离（L2 Distance）"]
        E1[直线距离<br/>值越小越相似]:::metricStyle
        E2[✅ 图像特征向量<br/>✅ 需要绝对距离场景]:::useStyle
    end
    class euclidean subgraphStyle

    C1 --> C2
    D1 --> D2
    E1 --> E2

    Note["推荐：文本 RAG 场景优先用余弦相似度<br/>OpenAI text-embedding-3 系列用点积"]:::noteStyle
    Note -.-> dot

    linkStyle 0,1,2 stroke:#333,stroke-width:1.5px
```

#### 2.1.3 ANN 索引算法

暴力搜索（精确 KNN）复杂度为 O(N×D)，百万级向量时耗时不可接受。主流 ANN（近似最近邻）算法：

| 算法 | 英文全称 | 中文名 | 原理 | 速度 | 精度 | 内存 | 适用场景 |
|------|---------|--------|------|------|------|------|---------|
| **HNSW** | Hierarchical Navigable Small World | 分层导航小世界图 | 多层图结构，上层稀疏捷径 + 下层密集精搜 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 | 通用场景首选 |
| **IVF-Flat** | Inverted File Index - Flat | 倒排文件索引（精确） | K-Means 聚类分区，查询时只搜最近簇 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 大规模数据集 |
| **IVF-PQ** | Inverted File Index - Product Quantization | 倒排文件索引（乘积量化） | IVF 分区 + 向量压缩编码，大幅降低内存 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 低 | 内存受限场景 |
| **ScaNN** | Scalable Nearest Neighbors | 可扩展近邻搜索 | 各向异性量化，Google 自研，兼顾速度与精度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | Google 生产环境 |

### 2.2 主流向量数据库对比

```mermaid
flowchart TD
    classDef pgStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef weaviateStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef qdrantStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef milvusStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef chromaStyle fill:#ff9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef coreStyle fill:#e8f4f8,stroke:#4299e1,stroke-width:1.5px

    subgraph pgvector["PgVector（PostgreSQL 插件）"]
        PG1[索引：HNSW / IVF]:::pgStyle
        PG2[存储：行存 + 向量列]:::pgStyle
        PG3[适合：已有 PG 基础设施<br/>中小规模 < 500万]:::pgStyle
    end
    class pgvector subgraphStyle

    subgraph weaviate["Weaviate"]
        W1[索引：HNSW]:::weaviateStyle
        W2[存储：对象存储 + 向量]:::weaviateStyle
        W3[适合：多模态 + GraphQL<br/>中大规模]:::weaviateStyle
    end
    class weaviate subgraphStyle

    subgraph qdrant["Qdrant"]
        Q1[索引：HNSW + 量化]:::qdrantStyle
        Q2[存储：Segment 分段存储]:::qdrantStyle
        Q3[适合：高性能低内存<br/>Rust 实现，延迟极低]:::qdrantStyle
    end
    class qdrant coreStyle

    subgraph milvus["Milvus"]
        M1[索引：HNSW/IVF/ScaNN]:::milvusStyle
        M2[存储：列存 + 对象存储]:::milvusStyle
        M3[适合：超大规模亿级<br/>云原生分布式]:::milvusStyle
    end
    class milvus subgraphStyle

    subgraph chroma["Chroma"]
        C1[索引：HNSW（hnswlib）]:::chromaStyle
        C2[存储：本地 SQLite]:::chromaStyle
        C3[适合：原型开发<br/>本地快速验证]:::chromaStyle
    end
    class chroma subgraphStyle

    PG1 --> PG2 --> PG3
    W1 --> W2 --> W3
    Q1 --> Q2 --> Q3
    M1 --> M2 --> M3
    C1 --> C2 --> C3

    linkStyle 0,1,2,3,4,5,6,7,8,9 stroke:#666,stroke-width:1px
```

**详细对比表：**

| 维度 | PgVector | Weaviate | Qdrant | Milvus | Chroma |
|------|---------|---------|--------|--------|--------|
| **开源协议** | Apache 2.0 | BSD-3 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **实现语言** | C（PG 插件） | Go | Rust | Go + C++ | Python |
| **最大规模** | ~500 万 | ~1 亿 | ~1 亿 | 100 亿+ | ~100 万 |
| **混合搜索** | ✅（需手写） | ✅ 原生支持 | ✅ 原生支持 | ✅ 原生支持 | ❌ |
| **元数据过滤** | ✅ SQL WHERE | ✅ GraphQL | ✅ JSON Filter | ✅ 表达式 | ✅ 简单过滤 |
| **持久化** | PostgreSQL | 本地磁盘 | 本地磁盘 | 对象存储 | SQLite |
| **部署复杂度** | 低（PG 插件） | 中 | 低 | 高（K8s） | 极低 |
| **Dify 支持** | ✅ | ✅ | ✅ | ✅ | ✅ |

### 2.3 应用步骤

```mermaid
flowchart LR
    classDef stepStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef codeStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px

    subgraph step1["步骤一：文档切分（Chunking）"]
        S1[固定长度切分<br/>512 tokens]:::stepStyle
        S2[语义切分<br/>按段落/标题]:::stepStyle
        S3[递归切分<br/>LangChain RecursiveCharacterTextSplitter]:::stepStyle
    end
    class step1 subgraphStyle

    subgraph step2["步骤二：向量化（Embedding）"]
        E1[调用 Embedding 模型<br/>OpenAI / BGE / M3E]:::stepStyle
        E2[批量处理<br/>batch_size=32]:::stepStyle
    end
    class step2 subgraphStyle

    subgraph step3["步骤三：写入向量数据库"]
        W1[创建 Collection / Index]:::codeStyle
        W2[批量 upsert<br/>向量 + 元数据]:::codeStyle
    end
    class step3 subgraphStyle

    subgraph step4["步骤四：查询检索"]
        Q1[查询向量化]:::stepStyle
        Q2[ANN Search<br/>top_k=20]:::stepStyle
        Q3["元数据过滤<br/>WHERE category='技术'"]:::stepStyle
    end
    class step4 subgraphStyle

    step1 --> step2 --> step3 --> step4

    linkStyle 0,1,2 stroke:#333,stroke-width:2px
```

### 2.4 示例代码

#### 示例一：使用 Qdrant（推荐生产环境）

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION = "knowledge_base"
VECTOR_DIM  = 1536  # text-embedding-3-small 维度

# ① 创建集合（指定向量维度和距离度量）
qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

def embed(texts: list[str]) -> list[list[float]]:
    """批量向量化"""
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in resp.data]

def index_documents(chunks: list[dict]):
    """② 批量写入文档块"""
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
            },
        )
        for vec, chunk in zip(vectors, chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"已写入 {len(points)} 条向量")

def vector_search(query: str, top_k: int = 10, source_filter: str | None = None):
    """③ 向量检索（支持元数据过滤）"""
    query_vec = embed([query])[0]

    # 元数据过滤（可选）
    query_filter = None
    if source_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        {"text": r.payload["text"], "score": r.score, "source": r.payload["source"]}
        for r in results
    ]

# 使用示例
chunks = [
    {"text": "Transformer 架构由 Encoder 和 Decoder 组成...", "source": "attention.pdf", "page": 1},
    {"text": "Self-Attention 机制计算 Q、K、V 矩阵...", "source": "attention.pdf", "page": 2},
    {"text": "BERT 模型使用双向 Transformer Encoder...", "source": "bert.pdf", "page": 1},
]
index_documents(chunks)

results = vector_search("什么是 Self-Attention？", top_k=3)
for r in results:
    print(f"[{r['score']:.4f}] {r['text'][:60]}...")
```

#### 示例二：使用 PgVector（已有 PostgreSQL 时）

```python
import psycopg2
import numpy as np
from openai import OpenAI

client = OpenAI()
conn = psycopg2.connect("postgresql://user:pass@localhost/rag_db")

def setup_pgvector():
    """初始化 pgvector 扩展和表结构"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          BIGSERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                source      TEXT,
                embedding   vector(1536),          -- 向量列
                created_at  TIMESTAMP DEFAULT NOW()
            );
        """)
        # 创建 HNSW 索引（推荐）
        cur.execute("""
            CREATE INDEX IF NOT EXISTS docs_embedding_hnsw_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        conn.commit()

def pgvector_search(query: str, top_k: int = 5) -> list[dict]:
    """余弦相似度向量检索"""
    vec = client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

    # pgvector 语法：<=> 余弦距离，<#> 负点积，<-> L2 距离
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vec, vec, top_k))
        rows = cur.fetchall()

    return [{"text": r[0], "source": r[1], "score": float(r[2])} for r in rows]
```

---

## 三、重排序（Reranking）

### 3.1 基本原理

#### 3.1.1 为什么需要重排序

向量检索（ANN Search）是**双塔模型**：Query 和 Document 各自独立编码，只在最后做向量内积，速度快但精度有损。

重排序使用 **Cross-Encoder** 模型：Query 和 Document **拼接后**一起输入模型，模型直接输出相关性分数，精度远高于双塔，但计算量大，只适合对少量候选文档精排。

```mermaid
flowchart LR
    classDef biStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef crossStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef coreStyle fill:#e8f4f8,stroke:#4299e1,stroke-width:1.5px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    subgraph biEncoder["双塔模型（Bi-Encoder）— 粗排"]
        B1["Query Encoder<br/>q = E(query)"]:::biStyle
        B2["Doc Encoder<br/>d = E(doc)"]:::biStyle
        B3[相似度 = q · d<br/>向量内积]:::biStyle
        B1 & B2 --> B3
    end
    class biEncoder subgraphStyle

    subgraph crossEncoder["Cross-Encoder — 精排"]
        C1["[CLS] query [SEP] doc [SEP]<br/>拼接输入"]:::crossStyle
        C2[BERT / Transformer<br/>联合编码]:::crossStyle
        C3[线性层输出<br/>相关性分数 0~1]:::crossStyle
        C1 --> C2 --> C3
    end
    class crossEncoder coreStyle

    subgraph comparison["对比"]
        CM1["双塔：快 O(1)<br/>可预计算 Doc 向量"]:::biStyle
        CM2["Cross：慢 O(N)<br/>无法预计算，实时计算"]:::crossStyle
        CM3[典型用法：<br/>双塔粗排 top-100<br/>Cross精排 top-5]:::noteStyle
    end
    class comparison subgraphStyle

    biEncoder --> crossEncoder
    crossEncoder --> comparison

    linkStyle 0,1,2,3,4,5 stroke:#333,stroke-width:1.5px
```

#### 3.1.2 主流重排序模型

| 模型 | 来源 | 特点 | 适用语言 |
|------|------|------|---------|
| `BAAI/bge-reranker-v2-m3` | 智源研究院 | 多语言，性能最强 | 中英文 |
| `BAAI/bge-reranker-large` | 智源研究院 | 中文最优，速度快 | 中文为主 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace | 英文，轻量极快 | 英文 |
| `Cohere Rerank` | Cohere API | 商业 API，无需部署 | 多语言 |
| `Jina Reranker` | Jina AI | 开源+API 双模式 | 多语言 |

### 3.2 重排序工作流

```mermaid
flowchart LR
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef retrieveStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef rerankStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef outputStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef coreStyle fill:#e8f4f8,stroke:#4299e1,stroke-width:1.5px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    Q["用户问题<br/>「BERT 和 GPT 的区别」"]:::inputStyle

    subgraph recall["粗排召回（向量检索 top-20）"]
        R1[Doc1: BERT 使用双向 Encoder<br/>score=0.82]:::retrieveStyle
        R2[Doc2: GPT 是自回归模型<br/>score=0.79]:::retrieveStyle
        R3[Doc3: Transformer 基础架构<br/>score=0.76]:::retrieveStyle
        R4[Doc4: BERT 的预训练任务<br/>score=0.71]:::retrieveStyle
        RN[... Doc20]:::retrieveStyle
    end
    class recall subgraphStyle

    subgraph rerank["精排（Cross-Encoder 重排）"]
        RE["Cross-Encoder 联合计算<br/>[CLS] 问题 [SEP] 文档 [SEP]"]:::rerankStyle
        SC[重新评分<br/>精度更高]:::rerankStyle
        RE --> SC
    end
    class rerank coreStyle

    subgraph output["最终 Top-K（top-3）"]
        O1[Doc1: BERT 使用双向 Encoder<br/>新分数=0.95 ✅]:::outputStyle
        O2[Doc2: GPT 是自回归模型<br/>新分数=0.91 ✅]:::outputStyle
        O3[Doc4: BERT 的预训练任务<br/>新分数=0.87 ✅ 排名上升]:::outputStyle
    end
    class output subgraphStyle

    Q --> recall --> rerank --> output

    Note["重排序价值：Doc4 在向量检索中排第4<br/>经 Cross-Encoder 精排后排到第3<br/>语义理解更准确"]:::noteStyle
    Note -.-> rerank

    linkStyle 0,1,2 stroke:#333,stroke-width:2px
```

### 3.3 应用步骤与示例代码

#### 示例一：使用 BGE Reranker（本地部署）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class BGEReranker:
    """本地部署 BGE Cross-Encoder 重排序"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        """
        对候选文档重排序
        :param query: 查询问题
        :param documents: 候选文档列表（粗排结果）
        :param top_k: 返回最相关的 K 篇
        """
        # 构造 [query, doc] 对
        pairs = [[query, doc] for doc in documents]

        # Tokenize（批量处理）
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # 前向推理
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(scores).cpu().numpy()  # 归一化到 0~1

        # 按分数降序排列
        ranked = sorted(
            [{"text": doc, "score": float(score)} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )
        return ranked[:top_k]


# 使用示例
reranker = BGEReranker("BAAI/bge-reranker-v2-m3")

query = "BERT 和 GPT 有什么区别？"
candidates = [
    "Transformer 是 2017 年 Google 提出的基础架构。",          # 相关但不直接
    "BERT 使用双向 Transformer Encoder，通过 MLM 任务预训练。", # 高度相关
    "GPT 系列是自回归语言模型，只使用 Decoder。",               # 高度相关
    "BERT 和 GPT 都基于 Transformer，但编码方向不同。",         # 直接回答
    "卷积神经网络在图像识别领域表现优异。",                      # 不相关
]

results = reranker.rerank(query, candidates, top_k=3)
for i, r in enumerate(results, 1):
    print(f"Top{i} [{r['score']:.4f}]: {r['text']}")

# 输出：
# Top1 [0.9821]: BERT 和 GPT 都基于 Transformer，但编码方向不同。
# Top2 [0.9634]: BERT 使用双向 Transformer Encoder，通过 MLM 任务预训练。
# Top3 [0.9412]: GPT 系列是自回归语言模型，只使用 Decoder。
```

#### 示例二：使用 Cohere Rerank API（无需部署）

```python
import cohere

co = cohere.Client("your-api-key")

def cohere_rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    """使用 Cohere Rerank API 重排序"""
    response = co.rerank(
        model="rerank-multilingual-v3.0",   # 支持中文
        query=query,
        documents=documents,
        top_n=top_k,
        return_documents=True,
    )
    return [
        {
            "text": result.document.text,
            "score": result.relevance_score,
            "original_rank": result.index,  # 原始排名
        }
        for result in response.results
    ]

# 使用示例
results = cohere_rerank(
    query="什么是 RAG？",
    documents=["...", "...", "..."],
    top_k=3,
)
```

#### 示例三：完整 RAG 管道（向量检索 + 重排序）

```python
def rag_with_rerank(
    query: str,
    vector_top_k: int = 20,  # 粗排数量，较大
    rerank_top_k: int = 5,   # 精排数量，较小
) -> str:
    """
    两阶段 RAG：向量检索粗排 → Cross-Encoder 精排 → LLM 生成
    """
    # 第一阶段：向量检索（粗排，top_k 较大）
    candidates = vector_search(query, top_k=vector_top_k)
    candidate_texts = [c["text"] for c in candidates]

    # 第二阶段：重排序（精排，top_k 较小）
    reranked = reranker.rerank(query, candidate_texts, top_k=rerank_top_k)

    # 第三阶段：组装 Prompt 调用 LLM
    context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(reranked)])
    prompt = f"""根据以下参考资料回答问题：

{context}

问题：{query}
答案："""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

---

## 四、混合搜索（Hybrid Search）

### 4.1 基本原理

#### 4.1.1 单一检索方式的局限

| 检索方式 | 优势 | 劣势 | 典型失败场景 |
|---------|------|------|------------|
| **向量检索** | 语义理解，同义词友好 | 无法匹配精确词，对专有名词弱 | 搜索 "CVE-2024-1234" 漏洞编号 |
| **关键词检索（BM25）** | 精确匹配，可解释性强 | 无法理解同义词和语义相似性 | 搜索 "苹果手机" 找不到含 "iPhone" 的文档 |

**混合搜索**将两者结合，互补长短，同时提升召回率和准确性。

#### 4.1.2 混合搜索架构

```mermaid
flowchart LR
    classDef queryStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef vectorStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef bm25Style fill:#9ff,stroke:#333,stroke-width:2px
    classDef fusionStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef outputStyle fill:#ff9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef coreStyle fill:#e8f4f8,stroke:#4299e1,stroke-width:1.5px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    Q[用户查询]:::queryStyle

    subgraph vectorBranch["语义检索分支"]
        V1[Embedding 向量化]:::vectorStyle
        V2[ANN 向量搜索<br/>top-20]:::vectorStyle
        V3[结果列表 A<br/>含相似度分数]:::vectorStyle
        V1 --> V2 --> V3
    end
    class vectorBranch subgraphStyle

    subgraph bm25Branch["关键词检索分支"]
        B1[分词 + 词干处理]:::bm25Style
        B2[BM25 倒排索引<br/>top-20]:::bm25Style
        B3[结果列表 B<br/>含 BM25 分数]:::bm25Style
        B1 --> B2 --> B3
    end
    class bm25Branch subgraphStyle

    subgraph fusion["融合排序层"]
        F1[分数归一化<br/>Min-Max / Z-Score]:::fusionStyle
        F2[RRF 互惠排名融合<br/>或 加权融合]:::fusionStyle
        F3[合并去重]:::fusionStyle
        F1 --> F2 --> F3
    end
    class fusion coreStyle

    subgraph output["融合后 top-K"]
        O1[最终候选文档列表<br/>准确率 ↑ 召回率 ↑]:::outputStyle
    end
    class output subgraphStyle

    Q --> vectorBranch & bm25Branch
    V3 & B3 --> fusion --> output

    Note["RRF 公式：score(d) = Σ 1/(k + rank_i(d))<br/>k=60 为平滑参数<br/>无需归一化，健壮性强"]:::noteStyle
    Note -.-> fusion

    linkStyle 0,1,2,3,4,5,6,7,8 stroke:#333,stroke-width:1.5px
```

### 4.2 融合算法详解

#### 4.2.1 RRF（互惠排名融合）— 推荐

RRF（Reciprocal Rank Fusion）是目前最流行的混合搜索融合算法，无需对分数做归一化：

```
RRF_score(d) = Σᵢ  1 / (k + rankᵢ(d))
```

- `k = 60`：平滑参数，防止排名靠前的文档过度主导
- `rankᵢ(d)`：文档 d 在第 i 路检索结果中的排名（从 1 开始）
- 若文档在某路检索中未出现，则不计入该路得分

**示例计算：**

| 文档 | 向量检索排名 | BM25 排名 | RRF 分数 |
|------|-----------|---------|---------|
| Doc A | 1 | 3 | 1/61 + 1/63 = 0.0323 |
| Doc B | 2 | 1 | 1/62 + 1/61 = 0.0325 ← 最高 |
| Doc C | 3 | 未出现 | 1/63 = 0.0159 |

#### 4.2.2 加权线性融合

```
final_score(d) = α × normalize(vector_score) + (1-α) × normalize(bm25_score)
```

- α 为权重参数，通常取 0.7（偏重语义）
- 需要对两路分数各自做归一化（Min-Max）
- 适合对权重有业务直觉时使用

### 4.3 应用步骤与示例代码

#### 示例一：RRF 融合实现

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
    id_key: str = "id",
) -> list[dict]:
    """
    互惠排名融合（RRF）
    :param results_list: 多路检索结果，每路是按相关性排序的文档列表
    :param k: 平滑参数，默认 60
    :param id_key: 文档唯一标识字段名
    :return: 融合后按 RRF 分数排序的文档列表
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_store: dict[str, dict] = {}

    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_store[doc_id] = doc  # 保存文档内容

    # 按 RRF 分数降序排列
    fused = sorted(
        [{"rrf_score": score, **doc_store[doc_id]} for doc_id, score in rrf_scores.items()],
        key=lambda x: x["rrf_score"],
        reverse=True,
    )
    return fused


# 使用示例
vector_results = [
    {"id": "doc_a", "text": "Transformer 注意力机制...", "vector_score": 0.92},
    {"id": "doc_b", "text": "Self-Attention 详解...", "vector_score": 0.88},
    {"id": "doc_c", "text": "BERT 模型架构...", "vector_score": 0.81},
]

bm25_results = [
    {"id": "doc_b", "text": "Self-Attention 详解...", "bm25_score": 12.4},
    {"id": "doc_d", "text": "注意力机制计算 Q K V...", "bm25_score": 10.8},
    {"id": "doc_a", "text": "Transformer 注意力机制...", "bm25_score": 9.5},
]

fused = reciprocal_rank_fusion([vector_results, bm25_results])
for r in fused[:3]:
    print(f"[RRF={r['rrf_score']:.4f}] {r['text'][:50]}")

# doc_b 同时出现在两路，RRF 分数最高
```

#### 示例二：完整混合搜索（Qdrant 原生支持）

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, NamedSparseVector, NamedVector,
    SearchRequest, FusionQuery, Prefetch, Fusion,
)

qdrant = QdrantClient(host="localhost", port=6333)

def hybrid_search_qdrant(
    query: str,
    top_k: int = 10,
    collection: str = "hybrid_kb",
) -> list[dict]:
    """
    Qdrant 原生混合搜索（Dense + Sparse 向量）
    需要在写入时同时存储 dense 和 sparse 向量
    """
    # 1. Dense 向量（语义）
    dense_vec = embed([query])[0]

    # 2. Sparse 向量（BM25 / SPLADE）
    sparse_vec = bm25_encode(query)  # 返回 {indices: [], values: []}

    # 3. Qdrant Query API 混合搜索（内置 RRF 融合）
    results = qdrant.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",  # 使用 dense 向量字段
                limit=20,
            ),
            Prefetch(
                query=SparseVector(**sparse_vec),
                using="sparse",  # 使用 sparse 向量字段
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # RRF 融合
        limit=top_k,
        with_payload=True,
    )

    return [
        {"text": r.payload["text"], "score": r.score}
        for r in results.points
    ]
```

#### 示例三：基于 Elasticsearch 的混合搜索

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
INDEX = "knowledge_base"

def es_hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Elasticsearch 混合搜索：kNN + BM25
    Elasticsearch 8.x 原生支持 knn + query 融合
    """
    query_vec = embed([query])[0]

    body = {
        "size": top_k,
        # BM25 全文检索
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["content^2", "title"],  # title 字段权重 x2
            }
        },
        # kNN 向量检索（Elasticsearch 8.x）
        "knn": {
            "field": "embedding",
            "query_vector": query_vec,
            "k": 20,
            "num_candidates": 100,
        },
        # 两路结果的权重（线性融合）
        "rank": {
            "rrf": {  # Elasticsearch 8.9+ 支持 RRF
                "window_size": 20,
                "rank_constant": 60,
            }
        },
        "_source": ["content", "source"],
    }

    resp = es.search(index=INDEX, body=body)
    return [
        {"text": hit["_source"]["content"], "score": hit["_score"]}
        for hit in resp["hits"]["hits"]
    ]
```

---

## 五、三项技术的协同整合

### 5.1 完整 RAG 优化管道

```mermaid
flowchart LR
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef processStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef hybridStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef rerankStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef outputStyle fill:#ff9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef coreStyle fill:#e8f4f8,stroke:#4299e1,stroke-width:1.5px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    Q[用户问题]:::inputStyle

    subgraph hybridPhase["① 混合检索（召回 top-30）"]
        H1[向量检索<br/>top-20]:::hybridStyle
        H2[BM25 检索<br/>top-20]:::hybridStyle
        H3[RRF 融合<br/>→ top-30 候选]:::hybridStyle
        H1 & H2 --> H3
    end
    class hybridPhase subgraphStyle

    subgraph filterPhase["② 元数据过滤（可选）"]
        F1[按文档来源/时间过滤]:::processStyle
        F2[按权限/租户过滤]:::processStyle
    end
    class filterPhase subgraphStyle

    subgraph rerankPhase["③ 重排序（精排 top-5）"]
        R1[Cross-Encoder<br/>联合评分]:::rerankStyle
        R2[最终 top-5 文档]:::rerankStyle
        R1 --> R2
    end
    class rerankPhase coreStyle

    subgraph generatePhase["④ 生成"]
        G1[Prompt 组装]:::outputStyle
        G2[LLM 回答]:::outputStyle
        G1 --> G2
    end
    class generatePhase subgraphStyle

    Q --> hybridPhase --> filterPhase --> rerankPhase --> generatePhase

    Note["效果提升路径：<br/>仅向量检索 → 加混合搜索（召回+15%）<br/>→ 再加重排序（精度+20%）"]:::noteStyle
    Note -.-> rerankPhase

    linkStyle 0,1,2,3,4,5,6,7 stroke:#333,stroke-width:2px
```

### 5.2 端到端代码

```python
from dataclasses import dataclass

@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    retrieval_stats: dict

class OptimizedRAGPipeline:
    """完整优化 RAG 管道：混合检索 + 重排序 + 生成"""

    def __init__(
        self,
        vector_db,         # 向量数据库客户端
        bm25_index,        # BM25 索引
        reranker,          # Cross-Encoder 重排序模型
        llm_client,        # LLM 客户端
        hybrid_top_k: int = 30,   # 混合检索返回数量
        rerank_top_k: int = 5,    # 重排序后保留数量
    ):
        self.vector_db = vector_db
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.llm = llm_client
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: str) -> list[dict]:
        """混合检索"""
        # 并行发起两路检索（实际可用 asyncio 并发）
        vector_results = self.vector_db.search(query, top_k=20)
        bm25_results   = self.bm25_index.search(query, top_k=20)

        # RRF 融合
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=60,
        )
        return fused[:self.hybrid_top_k]

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-Encoder 重排序"""
        texts = [c["text"] for c in candidates]
        return self.reranker.rerank(query, texts, top_k=self.rerank_top_k)

    def generate(self, query: str, context_docs: list[dict]) -> str:
        """LLM 生成"""
        context = "\n\n".join(
            [f"[参考{i+1}] {doc['text']}" for i, doc in enumerate(context_docs)]
        )
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"根据以下参考资料回答问题：\n\n{context}\n\n问题：{query}\n答案："
            }],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def query(self, question: str) -> RAGResult:
        """完整查询管道"""
        # ① 混合检索
        candidates = self.retrieve(question)

        # ② 重排序
        reranked = self.rerank(question, candidates)

        # ③ 生成
        answer = self.generate(question, reranked)

        return RAGResult(
            answer=answer,
            sources=reranked,
            retrieval_stats={
                "hybrid_candidates": len(candidates),
                "rerank_top_k": len(reranked),
            },
        )
```

---

## 六、面试常见问题（FAQ）

### Q1：向量检索和关键词检索各自的优缺点是什么？什么时候用哪种？

**A：**

| 场景 | 推荐方式 |
|------|---------|
| 语义相似查询（同义词、释义） | 向量检索 |
| 精确词匹配（编号、专有名词、代码片段） | 关键词检索 |
| 生产 RAG 系统 | 混合搜索（两者都用） |
| 数据量 < 10 万，快速验证 | 向量检索 + Chroma |
| 企业级，已有 ES 基础设施 | ES 混合搜索 |

---

### Q2：HNSW 索引的原理是什么？为什么它比 IVF 更快、更准？

**A：** HNSW（Hierarchical Navigable Small World）是一种多层图结构：

- 最底层包含所有节点，上层是采样的"捷径层"
- 查询时从顶层稀疏图快速定位大概位置，逐层下沉精确搜索
- 时间复杂度接近 O(log N)，精度（Recall）通常能达到 99%+

IVF（倒排文件）先对向量做 K-Means 聚类，查询时只搜索最近的几个聚类中心内的向量，当聚类边界上的向量容易被漏掉（nprobe 参数需要手动调优），Recall 通常比 HNSW 低 3-5%。

**工程选择**：内存充足选 HNSW；内存紧张、数据量超亿级选 IVF-PQ（以精度换内存）。

---

### Q3：重排序模型（Cross-Encoder）为什么比双塔模型（Bi-Encoder）精度高？

**A：** 核心区别在于**交互时机**：

- **双塔**：Query 和 Document 分别独立编码，两个向量只在最后做一次内积，无法捕获词汇级别的精细交互（如"苹果"在不同上下文的歧义）
- **Cross-Encoder**：Query 和 Document 拼接后输入同一个 Transformer，通过自注意力机制让 Query 中每个 Token 都能"看到" Document 中所有 Token，交互充分，精度高

代价是无法预计算文档向量（每次推理都要把 Query 和 Document 重新拼接），适合对 top-20~100 的候选文档做精排，不适合全库扫描。

---

### Q4：RRF 为什么不需要分数归一化？

**A：** RRF 只依赖**排名（rank）**，不依赖**分数（score）**的绝对值。向量检索分数是余弦相似度（0.8~0.99），BM25 分数是对数概率（可能是 5~50），两者量纲完全不同，无法直接相加。

RRF 将每个文档的排名转化为 `1/(k+rank)`，这是一个归一化的值（最大约 1/61），不受原始分数量纲影响，因此可以直接累加不同路的 RRF 分数，健壮性极强，是生产环境最推荐的融合方式。

---

### Q5：文档切分（Chunking）策略如何影响 RAG 效果？

**A：** Chunking 是 RAG 的源头，切分不好会导致后续所有优化都白费：

| 切分策略 | 适用场景 | 注意事项 |
|---------|---------|---------|
| **固定长度（512 token）** | 快速实现 | 可能截断语义完整的句子 |
| **按段落/标题切分** | 结构化文档（技术手册）| 段落长度不均匀 |
| **递归切分** | 通用文档 | 配置合适的 overlap（如 50 token）防止上下文丢失 |
| **小块检索+大块生成** | 精度要求高 | 检索用小块（128 token），生成时返回对应的大块（父块）|
| **语义切分** | 长文档 | 用相邻句子的向量相似度判断切分边界 |

**最佳实践**：chunk_size=512，chunk_overlap=50，并保存父块 ID，检索到小块后扩展到父块给 LLM。

---

### Q6：如何评估 RAG 系统的检索质量？

**A：** 主要指标：

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Recall@K** | top-K 结果中包含正确文档的比例 | 命中数 / 总相关文档数 |
| **Precision@K** | top-K 结果中正确文档的比例 | 命中数 / K |
| **MRR（Mean Reciprocal Rank）** | 第一个正确文档的排名倒数的均值 | avg(1 / first_correct_rank) |
| **NDCG@K** | 考虑排名权重的归一化折损累积增益 | 综合考虑排名和相关度 |

**工具推荐**：
- `RAGAS`：专为 RAG 设计的评估框架，可评估 faithfulness、answer_relevancy、context_recall
- `LlamaIndex`：内置 RetrieverEvaluator，支持自动化评测

---

### Q7：混合搜索中 α 权重（向量 vs BM25）如何调优？

**A：** 两种方法：

**方法一：经验值**
- 通用 RAG：α=0.7（偏重语义，向量占 70%）
- 法律/合规文档（精确条款重要）：α=0.4
- 代码搜索（函数名、API 名精确匹配重要）：α=0.3

**方法二：自动调优（推荐）**
```python
from scipy.optimize import minimize_scalar
from sklearn.metrics import ndcg_score

def objective(alpha):
    """目标函数：最大化验证集 NDCG"""
    fused_results = weighted_fusion(vector_results, bm25_results, alpha=alpha)
    return -ndcg_score(y_true, get_scores(fused_results))

result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
best_alpha = result.x
print(f"最优权重 α = {best_alpha:.3f}")
```

如果使用 RRF，则无需调权重，直接使用即可。

---

*文档版本：v1.0 | 最后更新：2026-03*
