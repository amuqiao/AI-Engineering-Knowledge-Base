# Codex 安装和使用教程

## 目录
- [1. Codex 简介](#1-codex-简介)
- [2. 为什么选择 GPT-5.4](#2-为什么选择-gpt-54)
- [3. 国内使用 Codex 的核心思路](#3-国内使用-codex-的核心思路)
- [4. Codex 安装教程（CLI 方式）](#4-codex-安装教程cli-方式)
- [5. 国内中转方案（推荐：cc-switch）](#5-国内中转方案推荐cc-switch)
- [6. Codex 核心使用方式](#6-codex-核心使用方式)
- [7. IDE 中使用 Codex](#7-ide-中使用-codex)
- [8. Codex 自动化能力（核心价值）](#8-codex-自动化能力核心价值)
- [9. 实战开发流程](#9-实战开发流程)
- [10. 常见问题](#10-常见问题)
- [11. 总结](#11-总结)

## 1. Codex 简介

在 AI 编程进入"智能体时代"的今天，Codex 已经不再只是一个代码补全工具，而是一个可以理解项目、修改代码、自动执行任务的 AI 编程助手。随着 GPT-5.4 的发布，Codex 的能力也完成了一次关键跃迁：

**从"写代码" → "做工程"**

Codex 是由 OpenAI 提供的 AI 编程模型与工具体系，基于 GPT-5 系列模型构建，能够理解自然语言并生成、修改和执行代码。

**核心能力：**
- 项目级代码理解
- 自动修改与重构
- Shell 命令执行
- Bug 自动修复
- 自动化任务编排

**可以理解为：**
```
Codex = AI 编程模型（GPT-5.x） + 本地执行代理 + 开发工具链
```

**常见使用形态：**
- CLI（命令行，最强）
- IDE 插件（Cursor / VS Code）
- 桌面应用
- 云端模式

## 2. 为什么选择 GPT-5.4

GPT-5.4 是目前 Codex 体系中的最强通用模型，适合复杂工程任务。

**核心优势：**
- 更强代码理解能力
- 支持复杂项目推理
- 上下文更大（适合大型代码库）
- 自动化能力更强

**推荐使用策略：**
| 场景 | 模型 |
|------|------|
| 日常开发 | GPT-5.3 |
| 复杂任务 | GPT-5.4（推荐） |

## 3. 国内使用 Codex 的核心思路

在国内使用 Codex，本质就是解决三个问题：

1. **安装 Codex**
2. **配置 API Key**
3. **配置中转（关键）**

> 其中第三点最重要：是否稳定，取决于中转配置

## 4. Codex 安装教程（CLI 方式）

### 1. 环境准备

### 2. 安装 Codex
```bash
npm install -g @openai/codex
# 国内建议：
npm install -g @openai/codex --registry=https://registry.npmmirror.com
```

**验证：**
```bash
codex --version
```

### 3. 登录方式

**✅ ChatGPT 登录（推荐）**
```bash
codex
# 浏览器授权即可
```

**✅ API Key 登录**
```bash
export OPENAI_API_KEY="sk-xxx"
codex
```

### 4. 指定 GPT-5.4
```bash
codex --model gpt-5.4
# 或写入配置：
# model = "gpt-5.4"
```

## 5. 国内中转方案（推荐：cc-switch）

在国内使用 Codex（尤其 GPT-5.4），强烈建议配置中转，否则体验极差甚至不可用。

### 1. cc-switch 是什么

cc-switch 是一个 AI 接口管理工具，可用于：
- API Key 管理
- Base URL 转发（中转）
- 多模型切换
- 本地统一入口

> 本质：AI 网关 + 配置管理器

### 2. 安装 cc-switch
```bash
wget https://github.com/farion1231/cc-switch/releases/download/v3.12.2/CC-Switch-v3.12.2-Linux-x86_64.AppImage

chmod +x CC-Switch-v3.12.2-Linux-x86_64.AppImage

./CC-Switch-v3.12.2-Linux-x86_64.AppImage
```

### 3. 配置中转

| 参数 | 示例 |
|------|------|
| Base URL | https://你的中转地址/v1 |
| API Key | sk-xxx |
| Model | gpt-5.4 |

### 4. 导出给 Codex
```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://你的中转地址/v1
# 或：
# base_url = "https://你的中转地址/v1"
# model = "gpt-5.4"
```

### 5. 验证
```bash
codex
# 输入：写一个 hello world
# 成功即配置完成 ✅
```

### 6. 进阶用法

**多模型策略**
- GPT-5.3 → 日常
- GPT-5.4 → 复杂任务

**搭配 Cursor**
```json
{
  "openai_base_url": "http://localhost:xxxx",
  "openai_api_key": "sk-xxx"
}
```

> 实现： Codex + Cursor + cc-switch 联动开发

## 6. Codex 核心使用方式

### 1. 启动
```bash
cd your-project
codex
```

### 2. 常见指令
- **分析项目**：分析这个项目结构
- **修复 Bug**：修复当前项目的报错
- **重构优化**：优化这个模块，降低耦合
- **生成功能**：实现一个用户登录系统（Go）

### 3. 自动模式
```bash
codex --auto-edit
```

**模式说明：**
| 模式 | 说明 |
|------|------|
| Suggest | 建议 |
| Auto Edit | 自动改 |
| Full Auto | 全自动 |

## 7. IDE 中使用 Codex

**支持：**
- VS Code
- Cursor
- Windsurf

**步骤：**
1. 安装插件
2. 登录
3. 选择 GPT-5.4

**优势：**
- 实时代码生成
- 项目上下文理解
- 边写边改

## 8. Codex 自动化能力（核心价值）

Codex 最大的价值不是"写代码"，而是：**自动完成开发任务**

**示例：创建一个 Golang 支付系统**

Codex 会自动：
1. 创建项目结构
2. 编写代码
3. 安装依赖
4. 执行测试

## 9. 实战开发流程

一个标准流程：

**Step 1：初始化**
- 创建一个 Go 微服务项目（Redis + MySQL）

**Step 2：模块开发**
- 实现用户系统 + 登录接口

**Step 3：调试**
- 修复所有接口错误

**Step 4：优化**
- 提升性能

> 最终效果：一个人 + Codex = 一个开发团队

## 10. 常见问题

**❓ 连接失败**
- 原因：网络问题、未配置中转
- 解决：使用 cc-switch

**❓ 额度问题**
- 建议：混用 5.3 / 5.4、控制上下文

**❓ Linux 无界面**
- 直接：codex

## 11. 总结

Codex + GPT-5.4 正在改变软件开发方式：

- 从"写代码" → "描述需求"
- 从"工具" → "执行代理"
- 从"开发" → "自动化"

**未来的开发模式：**
人类负责思考，AI 负责实现