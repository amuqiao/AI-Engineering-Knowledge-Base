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
6. [Query 优化技术](#六query-优化技术)
7. [高级检索策略](#七高级检索策略)
8. [RAG 评估体系](#八rag-评估体系)
9. [生产部署优化](#九生产部署优化)
10. [面试常见问题（FAQ）](#十面试常见问题faq)

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

## 六、Query 优化技术

用户原始查询往往模糊、简短或存在歧义，直接用其检索效果欠佳。Query 优化在"提问"这一源头提升检索质量。

### 6.1 查询改写（Query Rewriting）

让 LLM 将用户问题改写成更利于检索的形式：

```mermaid
flowchart LR
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef llmStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef outputStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    Q["原始问题<br/>「这个东西怎么用？」"]:::inputStyle

    subgraph rewrite["LLM 查询改写"]
        R1["消歧义<br/>「这个」指代什么？"]:::llmStyle
        R2["扩展关键词<br/>加入同义词/上位词"]:::llmStyle
        R3["生成改写后问题<br/>「Qdrant 向量数据库<br/>Python SDK 使用教程」"]:::llmStyle
        R1 --> R2 --> R3
    end
    class rewrite subgraphStyle

    Q --> rewrite

    Note["适用场景：<br/>对话历史上下文补全<br/>口语化问题规范化"]:::noteStyle
    Note -.-> rewrite
```

```python
from openai import OpenAI

client = OpenAI()

REWRITE_PROMPT = """你是一个搜索查询优化助手。将用户的问题改写为更适合文档检索的查询语句。
要求：
1. 消除代词指代歧义（如「它」「这个」）
2. 补充关键术语和同义词
3. 保持语义不变，输出 1 条改写后的查询
只输出改写结果，不要解释。

用户问题：{query}
对话历史：{history}
改写后的查询："""

def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """使用 LLM 改写查询，消歧义并补充关键词"""
    history_text = ""
    if history:
        history_text = "\n".join(
            [f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in history[-4:]]
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": REWRITE_PROMPT.format(query=query, history=history_text)
        }],
        temperature=0.1,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# 使用示例
history = [
    {"role": "user", "content": "我在用 Qdrant 做 RAG"},
    {"role": "assistant", "content": "好的，Qdrant 是一个高性能向量数据库..."},
]
original = "它支持混合搜索吗？怎么配置？"
rewritten = rewrite_query(original, history)
print(f"原始：{original}")
print(f"改写：{rewritten}")
# 改写：Qdrant 向量数据库混合搜索（Hybrid Search）配置方法与示例
```

### 6.2 HyDE（假设文档嵌入）

**HyDE（Hypothetical Document Embeddings）** 是一种反直觉但高效的技术：让 LLM 先"凭空"生成一个假设答案文档，然后用这个假设文档的向量去检索，而非用问题的向量检索。

**原理：** 问题的向量空间分布与答案文档的向量空间分布不一致。假设文档与真实文档在向量空间中更近。

```mermaid
flowchart TD
    classDef inputStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef llmStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef vectorStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef dbStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    Q["用户问题<br/>「BERT 预训练任务有哪些？」"]:::inputStyle

    subgraph hyde["HyDE 处理"]
        H1["LLM 生成假设文档<br/>（不依赖知识库）<br/>「BERT 的预训练任务包括：\n1. MLM 掩码语言模型...\n2. NSP 下一句预测...」"]:::llmStyle
        H2["对假设文档做 Embedding<br/>得到向量 v_hyp"]:::vectorStyle
        H3["用 v_hyp 检索向量数据库<br/>（而不是用问题向量）"]:::dbStyle
        H1 --> H2 --> H3
    end
    class hyde subgraphStyle

    Q --> hyde

    Note["效果：假设文档的向量分布<br/>比问题向量更接近真实答案文档<br/>尤其对短问题效果提升显著"]:::noteStyle
    Note -.-> H2
```

```python
def hyde_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    HyDE：用假设文档向量检索，提升短问题检索精度
    """
    # ① LLM 生成假设文档
    hyde_prompt = f"""请根据以下问题，生成一段简短的参考文档（2-3句话），
就像这个问题的标准答案文档一样。直接输出文档内容，不要前缀。

问题：{query}
参考文档："""

    hyp_doc = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.7,
        max_tokens=300,
    ).choices[0].message.content.strip()

    # ② 对假设文档做 Embedding（而非问题本身）
    hyp_vec = embed([hyp_doc])[0]

    # ③ 用假设文档向量检索（也可与问题向量平均融合）
    query_vec = embed([query])[0]
    fused_vec = [(h + q) / 2 for h, q in zip(hyp_vec, query_vec)]  # 平均融合

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=fused_vec,
        limit=top_k,
        with_payload=True,
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]
```

### 6.3 多路查询扩展（Multi-Query）

从不同角度生成多个查询，分别检索后合并，显著提升召回覆盖率：

```python
MULTI_QUERY_PROMPT = """从不同角度将以下问题改写为 {n} 个查询变体，
角度包括：同义表达、上位概念、子问题分解。
每行输出一个查询，不要编号。

原始问题：{query}
查询变体："""

def multi_query_retrieve(query: str, n_queries: int = 3, top_k: int = 5) -> list[dict]:
    """
    多路查询：生成多个查询变体，合并检索结果，用 RRF 融合
    """
    # ① 生成多个查询变体
    variants_text = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": MULTI_QUERY_PROMPT.format(n=n_queries, query=query)
        }],
        temperature=0.8,
    ).choices[0].message.content.strip()

    queries = [query] + [q.strip() for q in variants_text.split("\n") if q.strip()]
    queries = queries[:n_queries + 1]  # 原始 + n_queries 个变体

    # ② 每个查询独立检索
    all_results = []
    for q in queries:
        results = vector_search(q, top_k=top_k)
        # 加入文档 ID（用 text hash 模拟）
        for r in results:
            r["id"] = str(hash(r["text"]))
        all_results.append(results)

    # ③ RRF 融合多路结果
    fused = reciprocal_rank_fusion(all_results, k=60)
    return fused[:top_k]


# 使用示例
results = multi_query_retrieve(
    "BERT 模型怎么做文本分类？",
    n_queries=3,
)
# 内部生成的查询变体可能包括：
# - "BERT fine-tuning 文本分类任务"
# - "预训练语言模型文本分类方法"
# - "如何用 BERT 做 sentiment analysis"
```

### 6.4 Step-Back Prompting（退后提问）

对于具体问题，先提炼出其背后的通用原理问题，检索原理性文档，再结合具体问题回答：

```python
STEP_BACK_PROMPT = """给定一个具体问题，生成一个更通用的、后退一步的问题，
用于检索更基础的背景知识。只输出后退后的问题。

具体问题：{query}
后退后的问题："""

def step_back_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Step-Back：先检索通用原理，再检索具体内容，两路合并"""
    # ① 生成退后问题
    step_back_q = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": STEP_BACK_PROMPT.format(query=query)}],
        temperature=0.1,
    ).choices[0].message.content.strip()

    # ② 两路检索
    specific_results = vector_search(query, top_k=top_k)
    general_results  = vector_search(step_back_q, top_k=top_k)

    for r in specific_results: r["id"] = str(hash(r["text"]))
    for r in general_results:  r["id"] = str(hash(r["text"]))

    # ③ RRF 融合（具体问题权重稍高）
    return reciprocal_rank_fusion([specific_results, general_results])[:top_k]
```

---

## 七、高级检索策略

### 7.1 父子块检索（Parent-Child Chunking）

**核心思路**：检索用小块（精准匹配），生成用大块（丰富上下文）。

```mermaid
flowchart LR
    classDef parentStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef childStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef queryStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef outputStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px
    classDef noteStyle fill:#fff8e6,stroke:#ffb74d,stroke-width:1px

    subgraph indexPhase["索引阶段"]
        P1["父块 1（512 token）<br/>第1章：BERT 简介..."]:::parentStyle
        P2["父块 2（512 token）<br/>第2章：GPT 架构..."]:::parentStyle
        C1["子块 1-1（128 token）<br/>BERT 使用双向 Encoder"]:::childStyle
        C2["子块 1-2（128 token）<br/>BERT 预训练任务：MLM、NSP"]:::childStyle
        C3["子块 2-1（128 token）<br/>GPT 使用单向 Decoder"]:::childStyle
        P1 --> C1 & C2
        P2 --> C3
    end
    class indexPhase subgraphStyle

    subgraph retrievePhase["检索阶段"]
        Q["查询：BERT 预训练任务"]:::queryStyle
        VEC["向量检索子块<br/>命中：子块 1-2（精准）"]:::childStyle
        EXP["扩展至父块 1<br/>（512 token 完整上下文）"]:::parentStyle
        LLM["LLM 生成答案<br/>（基于完整父块内容）"]:::outputStyle
        Q --> VEC --> EXP --> LLM
    end
    class retrievePhase subgraphStyle

    Note["小块检索精度高<br/>父块生成上下文丰富<br/>两全其美"]:::noteStyle
    Note -.-> EXP
```

```python
import hashlib
from dataclasses import dataclass, field

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    parent_id: str | None
    chunk_type: str  # "parent" | "child"
    metadata: dict = field(default_factory=dict)

class ParentChildChunker:
    """父子块切分与索引"""

    def __init__(
        self,
        parent_size: int = 512,
        child_size: int = 128,
        overlap: int = 20,
    ):
        self.parent_size = parent_size
        self.child_size  = child_size
        self.overlap     = overlap

    def _split_by_tokens(self, text: str, size: int, overlap: int) -> list[str]:
        """按 token 数量切分（简化版，实际用 tiktoken）"""
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i: i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks

    def chunk(self, document: str, source: str) -> list[DocumentChunk]:
        """生成父块和子块"""
        chunks: list[DocumentChunk] = []

        parent_texts = self._split_by_tokens(document, self.parent_size, self.overlap)
        for p_idx, p_text in enumerate(parent_texts):
            parent_id = hashlib.md5(f"{source}_{p_idx}".encode()).hexdigest()
            chunks.append(DocumentChunk(
                chunk_id=parent_id,
                text=p_text,
                parent_id=None,
                chunk_type="parent",
                metadata={"source": source, "parent_idx": p_idx},
            ))

            # 对父块进一步切分为子块（仅子块进入向量索引）
            child_texts = self._split_by_tokens(p_text, self.child_size, 10)
            for c_idx, c_text in enumerate(child_texts):
                child_id = hashlib.md5(f"{parent_id}_{c_idx}".encode()).hexdigest()
                chunks.append(DocumentChunk(
                    chunk_id=child_id,
                    text=c_text,
                    parent_id=parent_id,
                    chunk_type="child",
                    metadata={"source": source, "parent_idx": p_idx, "child_idx": c_idx},
                ))

        return chunks


def parent_child_retrieve(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> list[DocumentChunk]:
    """
    父子块检索：用子块精确匹配，返回对应父块的完整内容
    """
    # 只对子块做向量索引
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    parent_map   = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}

    child_texts = [c.text for c in child_chunks]
    child_vecs  = embed(child_texts)

    query_vec = embed([query])[0]

    # 余弦相似度排序（简化，实际使用向量数据库）
    import numpy as np
    sims = [
        float(np.dot(query_vec, cv) / (np.linalg.norm(query_vec) * np.linalg.norm(cv)))
        for cv in child_vecs
    ]

    # 取 top-k 子块，扩展为父块（去重）
    top_children = sorted(
        zip(sims, child_chunks), key=lambda x: x[0], reverse=True
    )[:top_k * 2]

    seen_parent_ids: set[str] = set()
    result_parents: list[DocumentChunk] = []
    for _, child in top_children:
        if child.parent_id and child.parent_id not in seen_parent_ids:
            seen_parent_ids.add(child.parent_id)
            result_parents.append(parent_map[child.parent_id])
        if len(result_parents) >= top_k:
            break

    return result_parents
```

### 7.2 Sentence Window 检索

**思路**：以单句为最小检索单元，命中后扩展到前后 N 句的滑动窗口，兼顾精度和上下文：

```python
from nltk.tokenize import sent_tokenize

def build_sentence_window_index(
    document: str,
    window_size: int = 3,
) -> list[dict]:
    """
    构建句子窗口索引
    :param window_size: 每个索引单元包含的句子数（中心句 ± (window_size-1)//2）
    """
    sentences = sent_tokenize(document)
    half = (window_size - 1) // 2
    indexed_units = []

    for i, sent in enumerate(sentences):
        # 窗口文本（用于生成，包含上下文）
        start = max(0, i - half)
        end   = min(len(sentences), i + half + 1)
        window_text = " ".join(sentences[start:end])

        indexed_units.append({
            "id": f"sent_{i}",
            "sentence": sent,          # 向量化用（精准匹配）
            "window": window_text,     # LLM 生成用（丰富上下文）
            "sentence_idx": i,
        })

    return indexed_units


def sentence_window_retrieve(
    query: str,
    indexed_units: list[dict],
    top_k: int = 3,
) -> list[str]:
    """检索最相关句子，返回对应窗口文本"""
    import numpy as np

    # 仅对 sentence 字段做向量检索
    sentences = [u["sentence"] for u in indexed_units]
    sent_vecs  = embed(sentences)
    query_vec  = embed([query])[0]

    sims = [
        float(np.dot(query_vec, sv) / (np.linalg.norm(query_vec) * np.linalg.norm(sv)))
        for sv in sent_vecs
    ]

    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]

    # 返回窗口文本（比单句有更丰富的上下文）
    return [indexed_units[i]["window"] for i in top_indices]
```

### 7.3 上下文压缩（Contextual Compression）

检索到的文档可能包含大量噪音，上下文压缩让 LLM 只保留与问题相关的句子，降低 Token 消耗同时减少干扰：

```python
COMPRESS_PROMPT = """请从以下文档中提取与问题相关的句子，去除无关内容。
如果文档完全无关，输出 "IRRELEVANT"。
只输出相关内容，不要解释。

问题：{query}
文档：{document}
相关内容："""

def compress_document(query: str, document: str) -> str | None:
    """
    LLM 上下文压缩：从文档中提取与问题最相关的句子
    返回 None 表示文档与问题完全不相关
    """
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": COMPRESS_PROMPT.format(query=query, document=document)
        }],
        temperature=0.0,
        max_tokens=500,
    ).choices[0].message.content.strip()

    return None if result == "IRRELEVANT" else result


def retrieve_and_compress(
    query: str,
    top_k_retrieve: int = 10,
    top_k_final: int = 5,
) -> list[str]:
    """
    两阶段：向量检索 top-10 → 上下文压缩过滤 → 保留最终 top-5
    """
    candidates = vector_search(query, top_k=top_k_retrieve)

    compressed = []
    for doc in candidates:
        result = compress_document(query, doc["text"])
        if result:  # 过滤完全无关文档
            compressed.append(result)
        if len(compressed) >= top_k_final:
            break

    return compressed
```

### 7.4 RAPTOR（层次化文档摘要树）

RAPTOR 对文档做递归聚类和摘要，构建多层语义树，支持从细节到全局的多粒度检索：

```mermaid
flowchart TD
    classDef chunkStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef clusterStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef summaryStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef rootStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px

    subgraph L0["第 0 层（原始文档块）"]
        C1[chunk1]:::chunkStyle
        C2[chunk2]:::chunkStyle
        C3[chunk3]:::chunkStyle
        C4[chunk4]:::chunkStyle
        C5[chunk5]:::chunkStyle
        C6[chunk6]:::chunkStyle
    end
    class L0 subgraphStyle

    subgraph L1["第 1 层（聚类摘要）"]
        S1["摘要 A<br/>（chunk1+2+3 的摘要）"]:::summaryStyle
        S2["摘要 B<br/>（chunk4+5+6 的摘要）"]:::summaryStyle
    end
    class L1 subgraphStyle

    subgraph L2["第 2 层（全局摘要）"]
        Root["根摘要<br/>（整个文档的摘要）"]:::rootStyle
    end
    class L2 subgraphStyle

    C1 & C2 & C3 --> S1
    C4 & C5 & C6 --> S2
    S1 & S2 --> Root

    Note["检索时从各层同时检索<br/>细节问题命中 L0<br/>全局问题命中 L2"]
```

---

## 八、RAG 评估体系

### 8.1 核心评估指标

```mermaid
flowchart LR
    classDef metricStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef retrievalStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef generationStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef e2eStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px

    subgraph retrieval["检索质量指标"]
        R1["Recall@K<br/>Top-K 中正确文档占比"]:::retrievalStyle
        R2["Precision@K<br/>Top-K 中相关文档比例"]:::retrievalStyle
        R3["MRR<br/>第一个正确文档排名倒数均值"]:::retrievalStyle
        R4["NDCG@K<br/>带排名权重的归一化增益"]:::retrievalStyle
        R5["Context Recall<br/>上下文覆盖真实答案的程度"]:::retrievalStyle
    end
    class retrieval subgraphStyle

    subgraph generation["生成质量指标"]
        G1["Faithfulness<br/>答案与上下文的一致性<br/>（防幻觉）"]:::generationStyle
        G2["Answer Relevancy<br/>答案与问题的相关性"]:::generationStyle
        G3["Answer Correctness<br/>答案与参考答案的准确度"]:::generationStyle
    end
    class generation subgraphStyle

    subgraph e2e["端到端指标"]
        E1["RAGAS Score<br/>综合评分"]:::e2eStyle
        E2["Human Eval<br/>人工评分（Gold Standard）"]:::e2eStyle
    end
    class e2e subgraphStyle

    retrieval --> e2e
    generation --> e2e
```

**指标说明：**

| 指标 | 计算公式 | 值域 | 侧重 |
|------|---------|------|------|
| **Recall@K** | 命中相关文档数 / 全部相关文档数 | 0~1 | 检索覆盖率 |
| **Precision@K** | Top-K 中相关文档数 / K | 0~1 | 检索准确率 |
| **MRR** | avg(1 / rank_first_correct) | 0~1 | 首个正确结果排名 |
| **NDCG@K** | DCG@K / IDCG@K | 0~1 | 综合排序质量 |
| **Faithfulness** | 答案中有上下文支撑的陈述比例 | 0~1 | 防幻觉 |
| **Answer Relevancy** | 答案向量与问题向量余弦相似度 | 0~1 | 答非所问检测 |

### 8.2 RAGAS 框架使用

RAGAS 是专为 RAG 设计的自动化评估框架，无需人工标注，用 LLM 作为评判者：

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
from datasets import Dataset

def evaluate_rag_pipeline(
    questions: list[str],
    rag_pipeline,
    ground_truths: list[str] | None = None,
) -> dict:
    """
    使用 RAGAS 评估 RAG 管道质量
    :param questions: 测试问题列表
    :param rag_pipeline: RAG 管道对象（有 query 方法）
    :param ground_truths: 参考答案（可选，有则计算 answer_correctness）
    """
    answers, contexts = [], []

    for q in questions:
        result = rag_pipeline.query(q)
        answers.append(result.answer)
        contexts.append([doc["text"] for doc in result.sources])

    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        eval_data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(eval_data)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    if ground_truths:
        metrics.append(answer_correctness)

    result = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas().to_dict(orient="records")


# 使用示例
test_questions = [
    "BERT 和 GPT 的主要区别是什么？",
    "什么是 Self-Attention 机制？",
    "Transformer 的时间复杂度是多少？",
]
ground_truths = [
    "BERT 使用双向 Encoder，GPT 使用单向 Decoder...",
    "Self-Attention 通过 Q/K/V 矩阵计算注意力权重...",
    "标准 Transformer Self-Attention 时间复杂度为 O(n²d)...",
]

scores = evaluate_rag_pipeline(test_questions, pipeline, ground_truths)
print(f"Faithfulness:      {scores[0]['faithfulness']:.3f}")
print(f"Answer Relevancy:  {scores[0]['answer_relevancy']:.3f}")
print(f"Context Recall:    {scores[0]['context_recall']:.3f}")
print(f"Answer Correctness:{scores[0]['answer_correctness']:.3f}")
```

### 8.3 离线评估集构建

```python
import json
from pathlib import Path

def generate_eval_dataset(
    documents: list[str],
    n_questions: int = 50,
) -> list[dict]:
    """
    用 LLM 自动生成评估数据集（问题 + 参考答案 + 来源文档）
    """
    eval_data = []

    GEN_PROMPT = """基于以下文档，生成 {n} 个测试问题和对应的标准答案。
格式：每行一对，用 ||| 分隔：问题|||答案
文档：{doc}"""

    for doc in documents[:10]:  # 取前10篇文档
        n = max(1, n_questions // len(documents))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": GEN_PROMPT.format(n=n, doc=doc[:2000])
            }],
            temperature=0.7,
        ).choices[0].message.content

        for line in response.strip().split("\n"):
            if "|||" in line:
                q, a = line.split("|||", 1)
                eval_data.append({
                    "question": q.strip(),
                    "ground_truth": a.strip(),
                    "source_doc": doc[:500],
                })

    return eval_data
```

---

## 九、生产部署优化

### 9.1 语义缓存（Semantic Cache）

相似问题命中缓存，避免重复检索和 LLM 调用，显著降低延迟和成本：

```python
import numpy as np
import time
from dataclasses import dataclass, field

@dataclass
class CacheEntry:
    query: str
    query_vec: list[float]
    answer: str
    sources: list[dict]
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600  # 缓存有效期（秒）

class SemanticCache:
    """
    语义缓存：用向量相似度判断是否命中缓存
    命中条件：新查询向量与缓存查询向量余弦相似度 > threshold
    """

    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000):
        self.threshold = similarity_threshold
        self.max_size  = max_size
        self._entries: list[CacheEntry] = []

    def _cosine_sim(self, v1: list[float], v2: list[float]) -> float:
        a, b = np.array(v1), np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str, query_vec: list[float]) -> CacheEntry | None:
        """查找语义相似的缓存条目"""
        now = time.time()
        for entry in self._entries:
            if now - entry.created_at > entry.ttl:
                continue  # 跳过过期缓存
            sim = self._cosine_sim(query_vec, entry.query_vec)
            if sim >= self.threshold:
                print(f"[Cache HIT] sim={sim:.4f}, cached_query='{entry.query}'")
                return entry
        return None

    def set(self, query: str, query_vec: list[float], answer: str, sources: list[dict]):
        """写入缓存，超出 max_size 时淘汰最旧条目"""
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)  # FIFO 淘汰
        self._entries.append(CacheEntry(
            query=query, query_vec=query_vec, answer=answer, sources=sources
        ))


class CachedRAGPipeline:
    """带语义缓存的 RAG 管道"""

    def __init__(self, base_pipeline, cache: SemanticCache):
        self.pipeline = base_pipeline
        self.cache    = cache
        self.stats    = {"hits": 0, "misses": 0}

    def query(self, question: str) -> dict:
        query_vec = embed([question])[0]

        # 尝试命中缓存
        cached = self.cache.get(question, query_vec)
        if cached:
            self.stats["hits"] += 1
            return {"answer": cached.answer, "sources": cached.sources, "cached": True}

        # 缓存未命中，走完整管道
        self.stats["misses"] += 1
        result = self.pipeline.query(question)

        # 写入缓存
        self.cache.set(question, query_vec, result.answer, result.sources)
        return {"answer": result.answer, "sources": result.sources, "cached": False}
```

### 9.2 异步并行检索

向量检索和 BM25 检索可并行，结合异步 LLM 调用大幅降低端到端延迟：

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def async_vector_search(query: str, top_k: int = 20) -> list[dict]:
    """异步向量检索"""
    loop = asyncio.get_event_loop()
    # Qdrant 同步调用放入线程池
    results = await loop.run_in_executor(None, lambda: vector_search(query, top_k))
    return results

async def async_bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """异步 BM25 检索"""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: bm25_index.search(query, top_k))
    return results

async def async_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """异步重排序"""
    loop = asyncio.get_event_loop()
    texts = [c["text"] for c in candidates]
    return await loop.run_in_executor(None, lambda: reranker.rerank(query, texts, top_k))

async def async_rag_pipeline(question: str) -> dict:
    """
    完整异步 RAG 管道
    向量检索 & BM25 并行 → RRF 融合 → 重排序 → LLM 生成
    """
    import time
    t0 = time.perf_counter()

    # ① 并行发起向量检索 + BM25 检索
    vector_results, bm25_results = await asyncio.gather(
        async_vector_search(question, top_k=20),
        async_bm25_search(question, top_k=20),
    )
    t1 = time.perf_counter()

    # ② RRF 融合
    for r in vector_results: r.setdefault("id", str(hash(r["text"])))
    for r in bm25_results:   r.setdefault("id", str(hash(r["text"])))
    fused = reciprocal_rank_fusion([vector_results, bm25_results])[:30]
    t2 = time.perf_counter()

    # ③ 重排序
    reranked = await async_rerank(question, fused, top_k=5)
    t3 = time.perf_counter()

    # ④ 异步 LLM 生成
    context = "\n\n".join([f"[参考{i+1}] {doc['text']}" for i, doc in enumerate(reranked)])
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"根据参考资料回答：\n\n{context}\n\n问题：{question}"}],
        temperature=0.1,
    )
    t4 = time.perf_counter()

    return {
        "answer": response.choices[0].message.content,
        "timing": {
            "retrieval_ms":  round((t1 - t0) * 1000),
            "fusion_ms":     round((t2 - t1) * 1000),
            "rerank_ms":     round((t3 - t2) * 1000),
            "generation_ms": round((t4 - t3) * 1000),
            "total_ms":      round((t4 - t0) * 1000),
        }
    }


# 运行示例
async def main():
    result = await async_rag_pipeline("什么是 HNSW 索引？")
    print(result["answer"])
    print(f"总耗时: {result['timing']['total_ms']}ms")
    # 典型输出：检索并行后总耗时可从 800ms 降至 400ms

asyncio.run(main())
```

### 9.3 索引更新策略

| 场景 | 策略 | 实现方式 |
|------|------|---------|
| **文档新增** | 增量写入 | upsert 新向量，不重建索引 |
| **文档修改** | 先删后写 | delete old_id → upsert new_id |
| **文档删除** | 软删除 + 定期清理 | payload 标记 `deleted=true`，过滤查询 |
| **Embedding 模型升级** | 全量重建 | 后台异步重建新集合，完成后切流量 |
| **批量导入** | 批量 upsert | batch_size=256，并发多线程写入 |

```python
import asyncio
from qdrant_client.models import Filter, FieldCondition, MatchValue

class VectorIndexManager:
    """向量索引增量更新管理器"""

    def __init__(self, qdrant_client, collection: str):
        self.client     = qdrant_client
        self.collection = collection

    def upsert_documents(self, docs: list[dict], batch_size: int = 256):
        """批量增量写入（新增或更新）"""
        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            texts = [d["text"] for d in batch]
            vecs  = embed(texts)

            points = [
                PointStruct(
                    id=d["id"],
                    vector=vec,
                    payload={**d, "deleted": False, "updated_at": time.time()},
                )
                for d, vec in zip(batch, vecs)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            print(f"已写入 {i + len(batch)}/{len(docs)}")

    def soft_delete(self, doc_ids: list[str]):
        """软删除：打标记，不物理删除（避免索引碎片）"""
        from qdrant_client.models import SetPayload
        self.client.set_payload(
            collection_name=self.collection,
            payload={"deleted": True},
            points=doc_ids,
        )

    def search_active(self, query_vec: list[float], top_k: int = 10) -> list:
        """检索时自动过滤已删除文档"""
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="deleted", match=MatchValue(value=False))]
            ),
            with_payload=True,
        )

    def rebuild_collection(self, new_collection: str, all_docs: list[dict]):
        """全量重建（Embedding 模型升级时使用）"""
        # 新建集合
        self.client.recreate_collection(
            collection_name=new_collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        # 批量写入新向量
        mgr = VectorIndexManager(self.client, new_collection)
        mgr.upsert_documents(all_docs)
        print(f"重建完成，共 {len(all_docs)} 条，请切换流量到 {new_collection}")
```

### 9.4 性能调优 Checklist

```mermaid
flowchart LR
    classDef checkStyle fill:#9ff,stroke:#333,stroke-width:2px
    classDef warnStyle fill:#ffd700,stroke:#333,stroke-width:2px
    classDef okStyle fill:#9f9,stroke:#333,stroke-width:2px
    classDef subgraphStyle fill:#f5f5f5,stroke:#666,stroke-width:1px

    subgraph indexOpt["索引层优化"]
        I1["✅ 使用 HNSW 索引<br/>m=16, ef_construction=128"]:::okStyle
        I2["✅ 向量维度压缩<br/>int8 量化减少 75% 内存"]:::okStyle
        I3["✅ 按租户分 Collection<br/>避免元数据过滤全扫"]:::okStyle
    end
    class indexOpt subgraphStyle

    subgraph retrieveOpt["检索层优化"]
        R1["✅ 语义缓存<br/>相似度>0.95 命中缓存"]:::okStyle
        R2["✅ 并行双路检索<br/>asyncio.gather 并发"]:::okStyle
        R3["⚠️ top_k 不宜过大<br/>粗排 20~30，精排 3~5"]:::warnStyle
    end
    class retrieveOpt subgraphStyle

    subgraph llmOpt["生成层优化"]
        L1["✅ 上下文压缩<br/>减少无关 Token"]:::okStyle
        L2["✅ Streaming 输出<br/>降低感知延迟"]:::okStyle
        L3["⚠️ temperature=0.1<br/>RAG 场景避免高随机性"]:::warnStyle
    end
    class llmOpt subgraphStyle

    indexOpt --> retrieveOpt --> llmOpt
```

---

## 十、面试常见问题（FAQ）

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

### Q8：HyDE 的原理是什么？它为什么能提升检索效果？

**A：** HyDE（Hypothetical Document Embeddings）的核心洞察是：

- **问题与答案向量分布不同**：用户提问（短句、疑问句）和知识库文档（陈述句、专业描述）在 Embedding 空间中的分布存在 Domain Gap
- **解法**：让 LLM 先生成一个"假想答案文档"，用假想文档的向量去检索，假想文档与真实文档在向量空间中分布更接近

**适用场景**：
- 问题非常短或模糊（如 "attention 怎么算？"）
- 知识库语言风格与用户提问差异大（学术文档 vs 口语化提问）

**代价**：每次查询多一次 LLM 调用（延迟 + 成本增加），通常与查询向量平均融合使用，降低假想文档错误的影响。

---

### Q9：父子块检索和 Sentence Window 检索的区别是什么？分别适合什么场景？

**A：**

| 对比维度 | 父子块检索 | Sentence Window 检索 |
|---------|-----------|---------------------|
| **索引粒度** | 两层：小块（128 token）+ 大块（512 token） | 单句（1句） |
| **检索单元** | 小块（精准向量匹配） | 单句（精准向量匹配） |
| **生成上下文** | 对应的父块（固定大小） | 前后 N 句的滑动窗口 |
| **优势** | 语义边界清晰，父块完整性好 | 颗粒度更细，上下文扩展灵活 |
| **适合场景** | 结构化文档（技术手册、FAQ 文档）| 连续叙述文本（论文、新闻、书籍）|
| **实现复杂度** | 中（需维护父子 ID 映射） | 低（句子索引 + 窗口扩展）|

**工程建议**：先用 Sentence Window 快速实现，效果不满意再升级父子块。

---

### Q10：RAG 系统出现"幻觉"答案的根本原因是什么？如何系统性预防？

**A：** RAG 幻觉主要来源于三个环节：

**① 检索阶段：** 召回的文档与问题不相关，LLM 被迫用无关上下文"硬生成"
- 解决：提升检索质量（混合搜索 + 重排序）；设置相关性阈值，低于阈值直接回复"未找到相关信息"

**② Prompt 设计阶段：** Prompt 没有明确要求"只根据上下文回答"
- 解决：
```python
ANTI_HALLUCINATION_PROMPT = """严格根据以下参考资料回答问题。
如果参考资料中没有足够信息，直接回答"根据现有资料无法回答该问题"，不要编造。
不要在参考资料之外添加任何信息。

参考资料：
{context}

问题：{query}
答案："""
```

**③ 生成阶段：** temperature 过高导致随机性过大
- 解决：RAG 场景 temperature 设为 0.0~0.1

**系统性防御清单：**
- `Faithfulness` 评估分数 < 0.8 时触发告警
- 答案中关键事实需引用来源（Citation）
- 定期用 RAGAS 跑回归评估，监控指标趋势

---

### Q11：如何处理 RAG 中的多跳推理问题（Multi-Hop Question）？

**A：** 多跳问题如"A 公司的 CEO 毕业于哪所大学？"需要先找 CEO 是谁，再找其教育背景。

**方法一：问题分解（Query Decomposition）**
```python
DECOMPOSE_PROMPT = """将复杂问题分解为可以独立回答的子问题序列，按依赖关系排序。
每行一个子问题。

复杂问题：{query}
子问题列表："""

def multi_hop_rag(question: str) -> str:
    # ① 分解问题
    sub_questions_text = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=question)}],
    ).choices[0].message.content

    sub_questions = [q.strip() for q in sub_questions_text.split("\n") if q.strip()]

    # ② 顺序回答子问题，将中间结果作为下一跳的上下文
    intermediate_answers = []
    for sq in sub_questions:
        # 加入之前的中间结果作为额外上下文
        enriched_query = sq
        if intermediate_answers:
            enriched_query += f" （已知：{'；'.join(intermediate_answers[-2:])}）"

        docs = vector_search(enriched_query, top_k=3)
        context = "\n".join([d["text"] for d in docs])

        answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"参考资料：{context}\n\n问题：{sq}\n答案："}],
        ).choices[0].message.content
        intermediate_answers.append(answer)

    # ③ 综合子问题答案给出最终回答
    final_context = "\n".join(intermediate_answers)
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"基于以下中间分析给出最终答案：\n{final_context}\n\n原始问题：{question}"}],
    ).choices[0].message.content
```

**方法二：迭代检索（Iterative Retrieval / Self-RAG）**
- 每次生成后判断是否需要继续检索，直到有足够信息才生成最终答案

---

### Q12：向量数据库的数据量达到亿级时，应如何扩展？

**A：** 亿级数据向量数据库扩展方案：

| 挑战 | 解决方案 |
|------|---------|
| **内存不足（HNSW 内存高）** | 切换 IVF-PQ：向量量化压缩，内存减少 8~32 倍，精度损失 < 5% |
| **单机 QPS 瓶颈** | 水平分片（Sharding）：Milvus/Qdrant 均支持多节点分片 |
| **写入吞吐不足** | 异步批量写入 + WAL 预写日志，Milvus 支持 Kafka 作为消息队列 |
| **冷热数据分层** | 热数据（近 30 天）放内存索引，冷数据放磁盘（DiskANN）|
| **多租户隔离** | 按租户分 Collection（强隔离）或 Namespace + 元数据过滤（弱隔离）|

**选型建议：**
- < 1000 万：Qdrant 单机，HNSW 索引
- 1000 万 ~ 1 亿：Weaviate 或 Qdrant 集群
- > 1 亿：Milvus 分布式（Pulsar + MinIO + etcd）

---

### Q13：上下文窗口有限时，如何处理长文档 RAG？

**A：** 长文档处理策略：

**策略一：Map-Reduce 摘要**
```python
def long_doc_rag(query: str, long_doc: str, chunk_size: int = 2000) -> str:
    """超长文档分块处理，再综合"""
    chunks = [long_doc[i:i+chunk_size] for i in range(0, len(long_doc), chunk_size)]

    # Map：对每个块分别提取相关信息
    partial_answers = []
    for chunk in chunks:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"从以下文本中提取与「{query}」相关的关键信息，无相关信息则输出空：\n{chunk}"
            }],
        ).choices[0].message.content
        if resp.strip():
            partial_answers.append(resp)

    # Reduce：综合所有部分答案给出最终回答
    combined = "\n\n".join(partial_answers)
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"基于以下信息片段，综合回答问题「{query}」：\n\n{combined}"
        }],
    ).choices[0].message.content
```

**策略二：层次化检索（RAPTOR）**
- 构建文档摘要树，细节问题检索叶节点，全局问题检索根节点

**策略三：上下文压缩**
- 用 LLM 或 LLMLingua 压缩检索到的文档，在相同 Token 预算内放入更多文档

---

*文档版本：v2.0 | 最后更新：2026-03*
