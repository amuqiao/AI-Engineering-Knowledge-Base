# OpenTelemetry + Sentry 协作流程

> 展示从应用代码到 Sentry 的完整可观测性数据流转过程

```mermaid
flowchart LR
    %% ── 配色定义：按职责分色 ──────────────────────────────
    classDef appStyle       fill:#1e40af,stroke:#1e3a8a,stroke-width:2.5px,color:#fff
    classDef otelStyle      fill:#4f46e5,stroke:#3730a3,stroke-width:2px,color:#fff
    classDef collectorStyle fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff
    classDef sentryStyle    fill:#dc2626,stroke:#991b1b,stroke-width:2.5px,color:#fff
    classDef storageStyle   fill:#059669,stroke:#064e3b,stroke-width:2px,color:#fff
    classDef noteStyle      fill:#fffbeb,stroke:#f59e0b,stroke-width:1.5px,color:#78350f
    classDef layerStyle     fill:#f8fafc,stroke:#cbd5e0,stroke-width:1.5px

    %% ── 应用层 ─────────────────────────────────────────────
    subgraph Application["应用层"]
        direction LR
        APP1["业务应用<br>Business App"]:::appStyle
        APP2["微服务<br>Micro Service"]:::appStyle
    end
    class Application layerStyle

    %% ── OpenTelemetry 层 ───────────────────────────────────
    subgraph OpenTelemetry["OpenTelemetry 层"]
        direction LR
        OTEL_SDK["OpenTelemetry SDK<br>数据采集"]:::otelStyle
        OTEL_COLLECTOR["OpenTelemetry Collector<br>数据处理/聚合"]:::collectorStyle
    end
    class OpenTelemetry layerStyle

    %% ── Sentry 层 ──────────────────────────────────────────
    subgraph Sentry["Sentry 层"]
        direction LR
        SENTRY_SDK["Sentry SDK<br>错误/性能监控"]:::sentryStyle
        SENTRY_SERVER["Sentry Server<br>数据存储/分析"]:::sentryStyle
    end
    class Sentry layerStyle

    %% ── 存储层 ─────────────────────────────────────────────
    subgraph Storage["存储层"]
        direction LR
        DB[("Sentry 数据库<br>PostgreSQL")]:::storageStyle
        KAFKA[("消息队列<br>Kafka")]:::storageStyle
    end
    class Storage layerStyle

    %% ── 主流程数据流 ───────────────────────────────────────
    APP1 -->|"业务操作"| OTEL_SDK
    APP1 -->|"错误/异常"| SENTRY_SDK
    APP2 -->|"业务操作"| OTEL_SDK
    APP2 -->|"错误/异常"| SENTRY_SDK
    
    OTEL_SDK -->|"Traces/Metrics/Logs"| OTEL_COLLECTOR
    SENTRY_SDK -->|"Events/Transactions"| SENTRY_SERVER
    OTEL_COLLECTOR -->|"OpenTelemetry 协议"| SENTRY_SERVER
    
    SENTRY_SERVER -->|"存储事件"| DB
    SENTRY_SERVER -->|"异步处理"| KAFKA
    KAFKA -->|"消费处理"| SENTRY_SERVER

    %% ── 设计注记 ───────────────────────────────────────────
    NOTE["协作关键点<br>① OpenTelemetry 提供标准化数据采集<br>② Sentry 专注于错误监控和性能分析<br>③ 通过 OpenTelemetry Collector 实现数据转发<br>④ 双向集成：Sentry SDK 可直接采集，也可接收 OTel 数据"]:::noteStyle
    NOTE -.- Sentry

    %% 边索引：0-9，共 10 条
    linkStyle 0 stroke:#1e40af,stroke-width:2px
    linkStyle 1 stroke:#dc2626,stroke-width:2px
    linkStyle 2 stroke:#1e40af,stroke-width:2px
    linkStyle 3 stroke:#dc2626,stroke-width:2px
    linkStyle 4 stroke:#4f46e5,stroke-width:2px
    linkStyle 5 stroke:#dc2626,stroke-width:2px
    linkStyle 6 stroke:#0891b2,stroke-width:2px
    linkStyle 7 stroke:#059669,stroke-width:2px
    linkStyle 8 stroke:#d97706,stroke-width:2px,stroke-dasharray:4 3
    linkStyle 9 stroke:#d97706,stroke-width:2px
```

## 流程说明

1. **应用层**：业务应用和微服务产生各种操作和事件
2. **OpenTelemetry 层**：
   - OpenTelemetry SDK 采集应用的 Traces、Metrics 和 Logs
   - OpenTelemetry Collector 对数据进行处理和聚合
3. **Sentry 层**：
   - Sentry SDK 直接捕获应用的错误和异常
   - Sentry Server 接收来自 Sentry SDK 和 OpenTelemetry Collector 的数据
4. **存储层**：
   - PostgreSQL 存储 Sentry 的事件数据
   - Kafka 用于异步处理和消息队列

## 协作优势

- **标准化数据**：OpenTelemetry 提供统一的数据采集标准
- **专业分析**：Sentry 专注于错误监控和性能分析
- **灵活集成**：支持多种数据流转路径
- **全面可观测性**：结合 OpenTelemetry 的广泛采集和 Sentry 的深度分析