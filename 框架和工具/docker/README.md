# Docker 完整指南

本指南系统整理了 Docker 的核心功能、最佳实践和常见问题解决方案，结合命令示例、说明和 Mermaid 流程图，帮助开发者全面掌握 Docker 技术栈。

## 目录结构

1. [镜像管理](./1.镜像管理.md)
2. [容器管理](./2.容器管理.md)
3. [网络管理](./3.网络管理.md)
4. [数据卷管理](./4.数据卷管理.md)
5. [Docker Compose](./5.Docker%20Compose.md)
6. [镜像仓库管理](./6.镜像仓库管理.md)
7. [系统管理](./7.系统管理.md)
8. [监控与调试](./8.监控与调试.md)
   - [容器监控](./8.1.容器监控.md)
   - [容器调试](./8.2.容器调试.md)
9. [最佳实践](./9.最佳实践.md)
   - [镜像构建最佳实践](./9.1.镜像构建最佳实践.md)
   - [容器运行最佳实践](./9.2.容器运行最佳实践.md)
   - [安全最佳实践](./9.3.安全最佳实践.md)
10. [常见问题解决方案](./10.常见问题解决方案.md)
    - [权限问题](./10.1.权限问题.md)
    - [网络问题](./10.2.网络问题.md)
    - [存储问题](./10.3.存储问题.md)
    - [镜像拉取问题](./10.4.镜像拉取问题.md)
11. [命令速查表](./11.命令速查表.md)
    - [常用命令](./11.1.常用命令.md)
    - [Docker Compose 常用命令](./11.2.Docker%20Compose常用命令.md)
    - [资源清理命令](./11.3.资源清理命令.md)
12. [附录](./12.附录.md)
    - [Dockerfile 示例](./12.1.Dockerfile示例.md)
    - [docker-compose.yml 示例](./12.2.docker-compose.yml示例.md)
    - [常用 Docker 镜像](./12.3.常用Docker镜像.md)

## Docker 核心组件关系

```mermaid
flowchart TD
    subgraph Docker生态系统
        A[Docker客户端] -->|发送命令| B[Docker守护进程]
        B -->|管理| C[镜像]
        B -->|管理| D[容器]
        B -->|管理| E[网络]
        B -->|管理| F[数据卷]
        C -->|创建| D
        G[Docker Compose] -->|编排| D
        H[镜像仓库] <-->|拉取/推送| C
    end
    
    style A fill:#FF6B6B,stroke:#2D3436,stroke-width:3px,color:white,rx:8,ry:8
    style B fill:#4ECDC4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8
    style C fill:#45B7D1,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8
    style D fill:#96CEB4,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8
    style E fill:#FF9FF3,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8
    style F fill:#54A0FF,stroke:#2D3436,stroke-width:2px,color:white,rx:8,ry:8
    style G fill:#FECA57,stroke:#2D3436,stroke-width:2px,color:#2D3436,rx:8,ry:8
    style H fill:#E9ECEF,stroke:#2D3436,stroke-width:3px,color:#2D3436,rx:8,ry:8
```

## 学习路径

1. 基础概念：镜像、容器、仓库
2. 核心操作：镜像管理、容器管理
3. 高级功能：网络管理、数据卷管理
4. 编排工具：Docker Compose
5. 最佳实践：镜像构建、容器运行、安全
6. 监控与调试：容器监控、调试技巧
7. 常见问题：解决方案与排查思路

通过本指南的学习，您将能够：
- 熟练使用 Docker 命令行工具
- 构建高效、安全的 Docker 镜像
- 编排和管理多容器应用
- 解决 Docker 使用过程中的常见问题
- 遵循 Docker 最佳实践

让我们开始 Docker 之旅！