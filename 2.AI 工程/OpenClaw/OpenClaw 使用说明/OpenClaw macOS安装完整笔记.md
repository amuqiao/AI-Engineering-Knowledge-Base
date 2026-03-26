# OpenClaw macOS源码安装完整笔记（2026.3.7版本）

## 一、安装前置条件

### 1. 环境要求

| 依赖     | 版本要求                           | 验证命令（Terminal）                               |
| -------- | ---------------------------------- | -------------------------------------------------- |
| Node.js  | ≥22.12.0（推荐24.14.0）            | `node -v`                                          |
| npm      | ≥9.0.0（适配Node24：10.8.2稳定版） | `npm -v`                                           |
| Git      | 任意稳定版                         | `git --version`                                    |
| 终端权限 | 普通用户权限（无需sudo）           | -                                                  |
| 路径要求 | 无中文/空格/特殊字符               | 示例：`~/github_project/openclaw-2026.3.7`（推荐） |

### 2. 工具关系说明

**nvm、npm、Node.js 和 pnpm 之间的关系：**

1. **Node.js**：
   - JavaScript 运行时环境，用于执行 JavaScript 代码
   - 是基础运行环境，其他工具都围绕它工作
   - 每个版本的 Node.js 都会附带一个特定版本的 npm

2. **npm (Node Package Manager)**：
   - Node.js 的默认包管理器，用于安装、管理依赖包
   - 随 Node.js 一起安装，但可以独立升级
   - 负责处理项目的依赖关系和脚本执行

3. **nvm (Node Version Manager)**：
   - Node.js 版本管理工具，允许在同一台机器上安装和切换多个 Node.js 版本
   - 独立于 Node.js 本身，用于管理不同版本的 Node.js
   - 当切换 Node.js 版本时，npm 版本也会随之切换到对应版本

4. **pnpm (Performant npm)**：
   - 高性能的 Node.js 包管理器，是 npm 的替代品
   - 提供更快的安装速度和更小的磁盘占用
   - 支持与 npm 相同的命令和配置，但内部实现更高效

**使用建议：**

- 使用 nvm 管理多个 Node.js 版本，适应不同项目的需求
- 对于 Node.js 24.x 版本，推荐使用 npm 10.8.2 以避免兼容性问题
- 优先使用 pnpm 进行依赖安装和项目构建，获得更好的性能和兼容性
- 当切换 Node.js 版本后，记得检查并调整 npm 版本以确保兼容性

### 3. 环境配置（版本不达标时）

#### （1）Node.js安装/升级（二选一）

- **方式1：直接安装（单版本）**
  1. 下载：[Node.js官网](https://nodejs.org/zh-cn/download/) → 选择`node-v24.14.0.pkg`；
  2. 安装：按照向导完成安装，默认路径`/usr/local/bin/node`；
  3. 验证：`node -v`显示`v24.14.0`。

- **方式2：nvm（多版本管理）**
  1. 安装nvm：
     ```bash
     curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
     ```
  2. 重新打开终端或执行：
     ```bash
     source ~/.zshrc
     ```
  3. 安装Node：
     ```bash
     nvm install 24.14.0
     nvm use 24.14.0
     ```
  4. 设置为默认版本（确保新建终端也使用v24.14.0）：
     ```bash
     nvm alias default 24.14.0
     ```
  5. 验证：
     ```bash
     node -v # 显示v24.14.0
     nvm list # 确认default -> 24.14.0
     ```

## 二、OpenClaw安装方式

### 源码安装（开发者/自定义改造）

#### 步骤1：克隆源码

```bash
# 切换到目标目录（无中文/空格）
cd ~/github_project

# 克隆仓库
git clone https://github.com/openclaw/openclaw.git openclaw-2026.3.7

# 进入项目目录
cd openclaw-2026.3.7
```

#### 步骤2：安装依赖（解决兼容/缓存问题）

```bash
# 降级npm到稳定版（避免Node24+npm11兼容问题）
npm install -g npm@10.8.2 --registry=https://registry.npmmirror.com

# 验证npm版本
npm -v # 显示10.8.2

# 清空缓存（解决npm报错）
npm cache clean --force

# 安装项目依赖
npm install --registry=https://registry.npmmirror.com

# 安装pnpm（推荐）
npm install -g pnpm

# 验证pnpm版本
pnpm -v
```

#### 步骤3：构建源码（生成可执行文件）

```bash
# 执行构建脚本（生成dist/entry.mjs）
pnpm build

# 专门构建UI部分（可选）
pnpm ui:build

# 验证构建结果（检查dist目录）
ls -l dist/ # 需包含entry.mjs/entry.js
```

#### 步骤4：验证版本

```bash
npx openclaw --version # 输出2026.3.7即为成功
```

## 三、启动OpenClaw服务

### 版本说明

- **源码版**：从GitHub克隆源码后在本地构建运行的版本，启动命令使用 `pnpm openclaw` 或 `npx openclaw`

### 1. 推荐启动方式（使用onboarding向导）

**推荐启动方式与手动启动的区别：**

- **推荐启动方式**：使用`onboard`命令，提供交互式向导，自动完成所有配置步骤，包括网关设置、渠道连接、技能安装等，适合首次安装和完整配置
- **手动启动方式**：需要分步执行`setup`和`gateway`命令，适合有经验的用户进行自定义配置

  2026.3.7版本推荐使用onboarding向导进行完整设置：

```bash
# 源码版
pnpm openclaw onboard --install-daemon
```

**执行说明：**

- `--install-daemon` 参数会自动安装并启动网关服务，无需再手动执行 `pnpm openclaw gateway` 命令
- 向导完成后，网关服务会自动运行，您可以直接访问控制台
- 如果需要手动打开控制台，可以执行：
  ```bash
  pnpm openclaw dashboard --no-open
  ```
  然后复制输出的 URL 到浏览器中打开

### 2. 手动启动步骤

#### 步骤1：初始化配置

```bash
# 源码版
pnpm openclaw setup
```

#### 步骤2：启动网关服务

```bash
# 启动网关和控制台
pnpm openclaw gateway
pnpm openclaw dashboard --no-open
# 示例输出:Dashboard URL: http://127.0.0.1:18789/#token=be422afe6ee7553b74dadaaf04e33d27fed7e614f507b9e4
# 复制url到浏览器中打开
```

#### 强制启动（杀死占用端口的进程）

```bash
# 源码版
pnpm openclaw gateway --mode local --force
```

### 3. 验证服务启动成功

- 终端输出：`✅ Gateway started successfully`；
- 浏览器访问：`http://localhost:18789`（能看到Control UI）；
- 接口验证（可选）：
  ```bash
  curl -X POST http://localhost:18789/api/health
  ```
  响应正常即为服务运行状态良好。

## 四、开发模式（源码版）

```bash
# 开发循环（TS文件自动重载）
pnpm gateway:watch

# 开发配置（隔离状态）
pnpm openclaw --dev gateway
```

## 五、常见问题及解决方案

### 1. 重新安装OpenClaw（源码版）

```bash
# 清理现有依赖和缓存
rm -rf node_modules && rm package-lock.json && npm cache clean --force

# 重新安装依赖（使用淘宝镜像）
npm install --registry=https://registry.npmmirror.com

# 重新构建
pnpm build
```

### 2. 升级Node.js（使用nvm）

```bash
# 查看可用版本
nvm list available

# 安装新版本
nvm install 24.14.0

# 使用新版本
nvm use 24.14.0

# 设置为默认版本
nvm alias default 24.14.0
```

### 3. 升级npm

```bash
# 升级到指定版本
npm install -g npm@10.8.2

# 验证版本
npm -v
```

### 4. 中断安装时的处理

```bash
# 清理node_modules
rm -rf node_modules

# 清理package-lock.json
rm package-lock.json

# 清理npm缓存
npm cache clean --force

# 重新安装依赖
npm install --registry=https://registry.npmmirror.com
```

### 5. 环境变量问题

```bash
# 检查Node.js路径
echo $PATH

# 在zsh中添加Node.js路径到~/.zshrc
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc

# 刷新环境变量
source ~/.zshrc

# 确保nvm配置正确加载（新建终端显示旧版本时）
source ~/.zshrc && nvm use 24.14.0 && node -v
```

### 6. 构建错误处理

```bash
# 清理构建缓存
rm -rf dist

# 重新构建
pnpm build

# 专门重新构建UI部分（可选）
pnpm ui:build

# 验证构建结果
ls -l dist/
```

### 7. 端口占用问题

```bash
# 查找占用18789端口的进程
lsof -i :18789

# 杀死占用端口的进程
kill -9 <进程ID>

# 或使用--force参数启动
openclaw gateway --mode local --force
```

### 8. 命令未找到错误

```bash
# 错误：zsh: command not found: openclaw

# 解决方案：如果是源码版，使用pnpm或npx
pnpm openclaw onboard --install-daemon
# 或
npx openclaw onboard --install-daemon
```

### 9. 一键解决依赖问题（构建失败时）

```bash
# 清理现有依赖和缓存
rm -rf node_modules && rm package-lock.json && npm cache clean --force

# 重新安装依赖（使用淘宝镜像）
npm install --registry=https://registry.npmmirror.com

# 重新构建
pnpm build

# 专门构建UI部分（可选）
pnpm ui:build
```

## 六、关键注意事项

1. **路径规范**：所有目录（Node安装/源码/项目）避免中文/空格/特殊字符；
2. **权限要求**：macOS系统无需管理员权限，普通用户权限即可；
3. **版本兼容**：Node.js 22.x+需搭配npm 10.8.2（避免11.x版本的兼容Bug）；
4. **推荐工具**：使用pnpm进行构建和运行，性能更好且兼容性更佳；
5. **沙箱模式**：生产环境务必开启沙箱（默认开启），禁止`--no-sandbox`；
6. **日志排查**：npm报错可查看`~/.npm/_logs`下的日志文件；
7. **macOS特殊**：确保系统已安装Xcode Command Line Tools，可通过`xcode-select --install`安装。

## 七、重要命令参考

| 命令                      | 功能描述                            | 适用场景     |
| ------------------------- | ----------------------------------- | ------------ |
| `openclaw onboard`        | 完整的向导式安装配置                | 首次安装     |
| `openclaw setup`          | 初始化本地配置和工作区              | 手动配置     |
| `openclaw gateway`        | 启动WebSocket网关                   | 核心服务启动 |
| `openclaw channels login` | 连接聊天渠道（WhatsApp/Telegram等） | 渠道配置     |
| `openclaw models`         | 模型配置和管理                      | AI模型设置   |
| `openclaw status`         | 显示渠道健康状态和最近会话          | 服务监控     |
| `openclaw doctor`         | 健康检查和快速修复                  | 故障排查     |
| `openclaw update`         | 更新OpenClaw版本                    | 版本升级     |

## 八、文档资源

- [官方文档](https://docs.openclaw.ai)
- [入门指南](https://docs.openclaw.ai/start/getting-started)
- [配置参考](https://docs.openclaw.ai/gateway/configuration)
- [渠道文档](https://docs.openclaw.ai/channels)
- [macOS平台指南](https://docs.openclaw.ai/platforms/macos)

## 九、版本更新说明

### 2026.3.7版本重要变更

- **Node.js要求**：最低版本从18.0.0提升至22.12.0
- **推荐包管理器**：优先使用pnpm进行构建和运行
- **启动命令**：推荐使用`openclaw onboard`进行完整设置
- **网关模式**：启动时需指定`--mode local`参数
- **开发体验**：新增`gateway:watch`命令支持TS文件自动重载

> 注：本笔记基于2026.3.7版本，后续版本可能会有命令和参数变更，请以官方文档为准。
