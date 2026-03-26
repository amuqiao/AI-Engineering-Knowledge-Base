# AgentScope uv 依赖管理指南

本文档详细介绍 uv 包管理器如何管理 AgentScope 项目的依赖，帮助开发者理解依赖管理的原理和最佳实践。

## 1. uv 包管理器简介

uv 是一个高性能的 Python 包管理器，由 Astral 公司开发，旨在替代传统的 pip 和 virtualenv，提供更快的依赖解析和安装速度。

### 1.1 uv 的主要特点

- **速度快**：使用 Rust 编写，依赖解析和安装速度比 pip 快 10-100 倍
- **内存占用低**：相比 pip，内存使用更高效
- **支持虚拟环境**：内置虚拟环境管理功能
- **向后兼容**：支持标准的 pyproject.toml 配置
- **缓存机制**：智能缓存依赖包，避免重复下载

## 2. uv 与其他包管理器的比较

| 特性 | uv | pip | pipenv | poetry |
|------|----|-----|--------|--------|
| 速度 | 极快 | 中等 | 较慢 | 中等 |
| 内存使用 | 低 | 高 | 高 | 中等 |
| 虚拟环境管理 | 内置 | 需单独安装 | 内置 | 内置 |
| 依赖解析 | 快速准确 | 较慢 | 较慢 | 准确 |
| 配置文件 | pyproject.toml | requirements.txt | Pipfile | pyproject.toml |

## 3. AgentScope 的依赖配置

AgentScope 使用 `pyproject.toml` 文件管理项目依赖，配置结构清晰，便于维护。

### 3.1 基本依赖配置
依赖管理相关术语

| 术语 | 定义 | 示例 |
|------|------|------|
| **基本依赖** | 项目运行所必需的核心依赖，定义在 `[project.dependencies]` 部分 | `aioitertools`, `numpy` 等 |
| **生产依赖** | 与基本依赖同义，指项目运行时必需的依赖 | 同基本依赖 |
| **可选依赖** | 提供特定功能的依赖，定义在 `[project.optional-dependencies]` 部分 | `websockets` (realtime 功能) |
| **依赖组** | `[project.optional-dependencies]` 下定义的功能模块依赖集合 | `a2a`, `realtime`, `gemini` 等 |
| **开发依赖** | 仅用于开发环境的依赖，定义在 `[project.optional-dependencies.dev]` 部分 | `pytest`, `pre-commit` 等 |
| **可编辑模式** | 以符号链接方式安装项目，修改源码后立即生效 | `uv pip install -e .` |
| **依赖解析** | 分析依赖关系并确定兼容版本的过程 | uv 的依赖解析机制 |
| **依赖树** | 依赖项及其子依赖的层级结构 | 可通过 `uv pip list --tree` 查看 |

基本依赖定义在 `[project]` 部分的 `dependencies` 列表中，是项目运行的核心依赖：

```toml
[project]
dependencies = [
    "aioitertools",
    "anthropic",
    "dashscope",
    "docstring_parser",
    "json5",
    "json_repair",
    "mcp>=1.13",
    "numpy",
    "openai",
    "python-datauri",
    "opentelemetry-api>=1.39.0",
    "opentelemetry-sdk>=1.39.0",
    "opentelemetry-exporter-otlp>=1.39.0",
    "opentelemetry-semantic-conventions>=0.60b0",
    "python-socketio",
    "shortuuid",
    "tiktoken",
    "sounddevice",
    "sqlalchemy",
    "python-frontmatter",
]
```

### 3.2 可选依赖组配置

可选依赖组定义在 `[project.optional-dependencies]` 部分，用于不同功能的场景：

```toml
[project.optional-dependencies]
# ------------ A2A protocol ------------
a2a = [
    "a2a-sdk",
    "httpx",
    "nacos-sdk-python>=3.0.0",
]

# ------------ Realtime -------------
realtime = ["websockets>=14.0", "scipy"]

# ------------ Model APIs ------------
gemini = ["google-genai"]
ollama = ["ollama>=0.5.4"]
models = [
    "agentscope[ollama]",
    "agentscope[gemini]",
]

# 其他依赖组...
```

## 4. uv 依赖管理机制

### 4.1 依赖安装命令

#### 4.1.1 安装基本依赖

```bash
uv pip install -e . --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

- `-e`：以可编辑模式安装，创建源码的符号链接
- `.`：当前目录，指向包含 pyproject.toml 的目录
- `--index-url`：指定 PyPI 镜像源，提高下载速度
- `--trusted-host`：信任指定的镜像源主机

#### 4.1.2 安装可选依赖组

```bash
uv pip install -e ".[realtime]" --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

- `.[realtime]`：安装基本依赖的同时，额外安装 `realtime` 依赖组

### 4.2 依赖解析机制

1. **依赖收集**：uv 首先收集项目中所有直接依赖
2. **版本解析**：根据依赖约束（如 `>=1.0.0`）解析兼容的版本
3. **依赖树构建**：构建完整的依赖树，处理依赖冲突
4. **依赖下载**：从指定的镜像源下载依赖包
5. **依赖安装**：将依赖安装到指定的环境中

### 4.3 虚拟环境管理

uv 内置了虚拟环境管理功能：

```bash
# 创建虚拟环境
uv venv --python 3.10

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
source .venv/Scripts/activate # Windows
```

## 5. 依赖类型与区别

### 5.1 依赖类型概述

| 依赖类型 | 定义位置 | 用途 | 安装方式 |
|---------|---------|------|----------|
| **基本依赖** | `[project.dependencies]` | 项目运行的核心依赖，必须安装 | `uv pip install -e .` |
| **可选依赖** | `[project.optional-dependencies]` | 特定功能的依赖，按需安装 | `uv pip install -e ".[依赖组]"` |
| **开发依赖** | `[project.optional-dependencies.dev]` | 开发和测试工具，仅开发时需要 | `uv pip install -e ".[dev]"` |

### 5.2 详细说明

#### 5.2.1 基本依赖
- **定义**：在 `[project]` 部分的 `dependencies` 列表中定义
- **用途**：提供项目运行的核心功能，是项目正常运行所必需的
- **特点**：所有环境都必须安装，包含最基础的功能依赖，安装时自动包含在所有依赖组中

#### 5.2.2 可选依赖
- **定义**：在 `[project.optional-dependencies]` 部分定义
- **用途**：提供特定功能的支持，如 A2A 协议、实时语音、模型集成等
- **特点**：按需安装，根据需要选择特定功能的依赖组，可以相互嵌套组合，形成依赖组的层级结构

#### 5.2.3 开发依赖
- **定义**：在 `[project.optional-dependencies]` 部分的 `dev` 依赖组中定义
- **用途**：提供开发、测试和文档生成等工具
- **特点**：仅在开发环境中需要，包含测试框架、代码检查工具、文档生成工具等

### 5.3 依赖组的使用

#### 5.3.1 主要依赖组

| 依赖组 | 包含内容 | 用途 |
|-------|---------|------|
| `a2a` | a2a-sdk, httpx, nacos-sdk-python | A2A 协议支持 |
| `realtime` | websockets, scipy | 实时语音功能 |
| `gemini` | google-genai | Google Gemini 模型 |
| `ollama` | ollama | 本地 Ollama 模型 |
| `tokens` | Pillow, transformers, jinja2 | 令牌处理 |
| `memory` | redis, mem0ai, reme-ai | 内存管理 |
| `rag` | 各种阅读器和向量数据库 | 检索增强生成 |
| `full` | 所有可选依赖 | 完整功能 |
| `dev` | 开发工具和测试依赖 | 开发环境 |

#### 5.3.2 依赖组的嵌套组合

**什么是依赖组的嵌套组合**：
- 依赖组可以相互引用，形成层级结构
- 一个依赖组可以包含其他依赖组，安装时会自动递归安装所有引用的依赖

**具体示例**：

```toml
# models 依赖组引用了 ollama 和 gemini 依赖组
models = [
    "agentscope[ollama]",  # 引用 ollama 依赖组
    "agentscope[gemini]",  # 引用 gemini 依赖组
]

# full 依赖组引用了多个依赖组，包括 models
full = [
    "agentscope[a2a]",      # 引用 a2a 依赖组
    "agentscope[models]",    # 引用 models 依赖组（会自动包含 ollama 和 gemini）
    "agentscope[tokens]",    # 引用 tokens 依赖组
    "agentscope[memory]",    # 引用 memory 依赖组
    "agentscope[rag]",       # 引用 rag 依赖组
    "agentscope[evaluate]",  # 引用 evaluate 依赖组
    "agentscope[realtime]",  # 引用 realtime 依赖组
]
```

**嵌套组合的优势**：
1. **简化配置**：避免重复定义相同的依赖组合
2. **模块化管理**：将相关功能的依赖组织成独立的组
3. **灵活安装**：用户可以根据需要安装不同级别的依赖
4. **一致性保证**：确保相关功能的依赖版本一致

**使用方式**：
- 安装嵌套依赖组时，只需指定最顶层的依赖组
- 例如：`uv pip install -e ".[full]"` 会自动安装所有嵌套的依赖组
- 系统会自动解析依赖树，避免重复安装和版本冲突

## 6. 命令参考

### 6.1 常用 uv 命令

```bash
# 安装 uv
pip install uv

# 创建虚拟环境
uv venv --python 3.10

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e .

# 安装可选依赖组
uv pip install -e ".[realtime]"

# 安装完整依赖
uv pip install -e ".[full]"

# 添加依赖（自动更新 pyproject.toml）
uv add new-dependency
uv add new-dependency --group realtime

# 卸载依赖
uv pip uninstall dependency-name

# 更新依赖
uv pip upgrade dependency-name
uv pip upgrade

# 同步依赖
# 默认只同步生产依赖（[project.dependencies]）
uv sync

# 同步生产依赖 + 特定可选依赖组
uv sync --group realtime

# 同步生产依赖 + 多个依赖组
uv sync --group realtime --group a2a

# 同步生产依赖 + 开发依赖
uv sync --group dev

# 同步生产依赖 + 所有可选依赖（使用 full 依赖组）
uv sync --group full

# 查看依赖
uv pip list
uv pip list --tree

# 清理缓存
uv cache clean
```

### 6.2 镜像源配置

```bash
# 使用阿里云镜像
uv pip install -e . --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 使用清华大学镜像
uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

## 7. 依赖管理操作指南

### 7.1 管理依赖的两种方式

#### 7.1.1 方法一：使用 uv add 命令（推荐）

**优势**：自动更新 `pyproject.toml` 文件并安装依赖，操作简洁且不易出错

**操作步骤**：
1. **添加依赖**：使用 `uv add` 命令添加依赖到指定依赖组
2. **验证依赖**：使用 `uv pip list` 命令验证依赖状态

**示例工作流**：
```bash
# 添加依赖到基本依赖组
uv add new-dependency

# 添加依赖到指定可选依赖组
uv add new-realtime-dependency --group realtime

# 添加带版本约束的依赖
uv add new-dependency>=1.0.0 --group realtime

# 验证依赖
uv pip list
```

#### 7.1.2 方法二：手动编辑配置文件

**适用场景**：需要批量修改依赖或进行复杂配置时

**操作步骤**：
1. **编辑 pyproject.toml**：在配置文件中定义依赖及其版本
2. **同步依赖**：使用 `uv sync` 命令确保环境与配置文件一致
3. **验证依赖**：使用 `uv pip list` 命令验证依赖状态

**示例工作流**：
```bash
# 1. 编辑 pyproject.toml 文件添加或修改依赖
# 2. 同步依赖到环境
uv sync

# 3. 验证依赖
uv pip list
```

### 7.2 常见操作示例

#### 7.2.1 添加依赖到指定组

**通过配置文件**：
```toml
# 添加基本依赖
[project]
dependencies = [
    # 现有依赖...
    "new-dependency>=1.0.0"  # 添加新的基本依赖
]

# 添加到可选依赖组
[project.optional-dependencies]
realtime = [
    "websockets>=14.0",
    "scipy",
    "new-realtime-dependency"  # 添加到 realtime 依赖组
]
```

**通过命令**：
```bash
# 添加到基本依赖组
uv add new-dependency

# 添加到可选依赖组
uv add new-realtime-dependency --group realtime
```

#### 7.2.2 删除依赖

**通过配置文件**：
```toml
# 删除基本依赖
[project]
dependencies = [
    # 保留需要的依赖，删除不需要的
]

# 从可选依赖组中删除
[project.optional-dependencies]
realtime = [
    "websockets>=14.0"  # 删除了 scipy
]
```

**通过命令**：
```bash
# 卸载依赖（需手动从配置文件中删除）
uv pip uninstall dependency-name
```

#### 7.2.3 修改依赖版本

**通过配置文件**：
```toml
[project.optional-dependencies]
realtime = [
    "websockets>=15.0"  # 修改版本约束
]
```

**通过命令**：
```bash
# 更新依赖版本（需手动更新配置文件）
uv pip upgrade dependency-name
```

### 7.3 更新环境依赖

当修改了 `pyproject.toml` 文件后，使用 `uv sync` 命令更新环境依赖。`uv sync` 命令会确保环境依赖与配置文件完全一致，根据指定的依赖组同步相应的依赖。

**基本用法**：

```bash
# 默认只同步生产依赖（[project.dependencies]）
uv sync

# 同步生产依赖 + 特定可选依赖组
uv sync --group realtime

# 同步生产依赖 + 多个依赖组
uv sync --group realtime --group a2a

# 同步生产依赖 + 开发依赖
uv sync --group dev

# 同步生产依赖 + 所有可选依赖（使用 full 依赖组）
uv sync --group full

# 验证依赖
uv pip list
```

**注意**：
- `uv sync` 会移除环境中不在配置文件中的依赖，确保环境与配置完全一致
- 默认只同步生产依赖，可选依赖和开发依赖需要明确指定
- 对于多环境部署，可根据不同环境的需求选择同步不同的依赖组

## 8. 依赖管理最佳实践

### 8.1 开发环境设置

1. **创建专用虚拟环境**：为每个项目创建独立的虚拟环境，避免依赖冲突
2. **使用镜像源**：指定国内镜像源，提高下载速度
3. **可编辑模式安装**：使用 `-e` 模式安装项目，方便开发调试
4. **安装必要的依赖组**：根据需要安装特定功能的依赖组

### 8.2 依赖版本管理

1. **固定关键依赖版本**：对关键依赖指定明确的版本范围
2. **定期更新依赖**：保持依赖的安全性和性能
3. **测试依赖变更**：更新依赖后进行充分测试

### 8.3 常见问题处理

- **依赖冲突**：使用 `uv pip list --tree` 查看依赖树，分析冲突原因
- **安装失败**：检查网络连接，尝试更换镜像源
- **版本不兼容**：查看错误信息，调整依赖版本约束
- **配置不一致**：使用 `uv sync` 确保环境与配置文件一致

## 9. 总结

uv 作为现代 Python 包管理器，为 AgentScope 项目提供了高效的依赖管理能力。通过合理配置 `pyproject.toml` 文件和使用可选依赖组，开发者可以根据需要灵活安装依赖，提高开发效率。

使用 uv 管理依赖的优势：
- 安装速度快，节省开发时间
- 依赖解析准确，减少版本冲突
- 配置简单清晰，易于维护
- 与标准工具兼容，学习成本低

通过本文档的介绍，希望开发者能更好地理解和使用 uv 进行依赖管理，为 AgentScope 项目的开发和部署提供有力支持。
附录
 `pyproject.toml` 配置文件
```toml
[project]
name = "agentscope"
dynamic = ["version"]
description = "AgentScope: A Flexible yet Robust Multi-Agent Platform."
readme = "README.md"
authors = [
    { name = "SysML team of Alibaba Tongyi Lab", email = "gaodawei.gdw@alibaba-inc.com" }
]
license = "Apache-2.0"
keywords = ["deep-learning", "multi agents", "agents"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.10"
dependencies = [
    "aioitertools",
    "anthropic",
    "dashscope",
    "docstring_parser",
    "json5",
    "json_repair",
    "mcp>=1.13",
    "numpy",
    "openai",
    "python-datauri",
    "opentelemetry-api>=1.39.0",
    "opentelemetry-sdk>=1.39.0",
    "opentelemetry-exporter-otlp>=1.39.0",
    "opentelemetry-semantic-conventions>=0.60b0",
    "python-socketio",
    "shortuuid",
    "tiktoken",
    "sounddevice",
    "sqlalchemy",
    "python-frontmatter",
]

[project.optional-dependencies]
# ------------ A2A protocol ------------
a2a = [
    "a2a-sdk",
    "httpx",
    # TODO: split the card resolvers from the a2a dependency
    "nacos-sdk-python>=3.0.0",
]

# ------------ Realtime -------------
realtime = ["websockets>=14.0", "scipy"]

# ------------ Model APIs ------------
gemini = ["google-genai"]
ollama = ["ollama>=0.5.4"]
models = [
    "agentscope[ollama]",
    "agentscope[gemini]",
]

# ------------ Tokenizers ------------
tokens = [
    "Pillow",
    "transformers",
    "jinja2",
]

# ------------ Memory ------------
redis_memory = ["redis"]

mem0ai = [
    "mem0ai<=1.0.3",
    "packaging"
]
reme = ["reme-ai>=0.2.0.3"]
memory = [
    "agentscope[redis_memory]",
    "agentscope[mem0ai]",
    "agentscope[reme]",
]

# ------------ RAG ------------
# readers
text-reader = ["nltk"]
pdf-reader = [
    "agentscope[text-reader]",
    # TODO: the latest pypdf has some issues with parsing PDFs
    #  (2026-01-13), so we fix the version here temporarily.
    "pypdf<=6.5.0",
]
docx-reader = [
    "agentscope[text-reader]",
    "python-docx"
]
excel-reader = [
    "agentscope[text-reader]",
    "pandas",
    "openpyxl",
]
ppt-reader = [
    "agentscope[text-reader]",
    "python-pptx"
]
readers = [
    "agentscope[text-reader]",
    "agentscope[pdf-reader]",
    "agentscope[docx-reader]",
    "agentscope[excel-reader]",
    "agentscope[ppt-reader]",
]

# vdb
# The qdrant-client >= 1.16.0 has conflicts with pymilvus, so we fix
# the version to 1.15.1 here.
qdrant = ["qdrant-client==1.15.1"]
milvus = ["pymilvus[milvus_lite]"]
ali_mysql = ["mysql-connector-python"]
mongodb = ["pymongo"]
oceanbase = ["pyobvector>=0.2.0,<0.3.0"]
vdbs = [
    "agentscope[ali_mysql]",
    "agentscope[qdrant]",
    "agentscope[milvus]",
    "agentscope[mongodb]",
    "agentscope[oceanbase]",
]

rag = [
    "agentscope[readers]",
    "agentscope[vdbs]",
]

# ------------ Evaluation ------------
evaluate = ["ray"]

# ------------ Full ------------
full = [
    "agentscope[a2a]",
    "agentscope[models]",
    "agentscope[tokens]",
    "agentscope[memory]",
    "agentscope[rag]",
    "agentscope[evaluate]",
    "agentscope[realtime]",
]

# ------------ Development ------------
dev = [
    # Include full dependencies from local package
    "agentscope[full]",
    # Development tools
    "pre-commit",
    "pytest",
    "pytest-forked",
    "sphinx-gallery",
    "furo",
    "myst_parser",
    "matplotlib",
    # For unittests
    # For mocking redis in unittests
    "fakeredis",
    "aiosqlite",
    "greenlet",
    # For openjudge
    "py-openjudge",
]

[project.urls]
Homepage = "https://github.com/agentscope-ai/agentscope"
Documentation = "https://doc.agentscope.io/"
Repository = "https://github.com/agentscope-ai/agentscope"

[tool.setuptools]
packages = { find = { where = ["src"] } }
include-package-data = true

[tool.setuptools.package-data]
"*" = ["py.typed"]

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.dynamic]
version = {attr = "agentscope._version.__version__"}

```