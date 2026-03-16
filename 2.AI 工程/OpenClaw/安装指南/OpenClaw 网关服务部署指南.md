# OpenClaw 网关服务部署指南

## 1. 网关服务安装

### 1.1 安装命令

```bash
# 常规安装
pnpm openclaw gateway install

# 强制安装（如需覆盖现有配置）
pnpm openclaw gateway install --force
```

### 1.2 安装执行结果

```bash
> node scripts/run-node.mjs gateway install

🦞 OpenClaw 2026.3.7 (1e94d88) — I run on caffeine, JSON5, and the audacity of "it worked on my machine."

Installed LaunchAgent: /Users/wangqiao/Library/LaunchAgents/ai.openclaw.gateway.plist
Logs: /Users/wangqiao/.openclaw/logs/gateway.log
```

## 2. 配置文件说明

### 2.1 配置文件路径

`~/.openclaw/openclaw.json`

### 2.2 核心配置项（已配置飞书 + Minimax 模型）

```json
{
  "meta": {
    "lastTouchedVersion": "2026.3.7",
    "lastTouchedAt": "2026-03-09T07:18:07.233Z"
  },
  "wizard": {
    "lastRunAt": "2026-03-09T07:03:11.898Z",
    "lastRunVersion": "2026.3.7",
    "lastRunCommand": "onboard",
    "lastRunMode": "local"
  },
  "auth": {
    "profiles": {
      "minimax-cn:default": {
        "provider": "minimax-cn",
        "mode": "api_key"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "minimax-cn": {
        "baseUrl": "https://api.minimaxi.com/anthropic",
        "api": "anthropic-messages",
        "authHeader": true,
        "models": [
          {
            "id": "MiniMax-M2.5",
            "name": "MiniMax M2.5",
            "reasoning": true,
            "input": ["text"],
            "cost": {
              "input": 0.3,
              "output": 1.2,
              "cacheRead": 0.03,
              "cacheWrite": 0.12
            },
            "contextWindow": 200000,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "minimax-cn/MiniMax-M2.5"
      },
      "models": {
        "minimax-cn/MiniMax-M2.5": {
          "alias": "Minimax"
        }
      },
      "workspace": "/Users/wangqiao/.openclaw/workspace"
    }
  },
  "tools": {
    "profile": "coding"
  },
  "bindings": [
    {
      "agentId": "main",
      "match": {
        "channel": "feishu",
        "accountId": "default"
      }
    }
  ],
  "commands": {
    "native": "auto",
    "nativeSkills": "auto",
    "restart": true,
    "ownerDisplay": "raw"
  },
  "session": {
    "dmScope": "per-channel-peer"
  },
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_a927137f6638dbd1",
      "appSecret": "KRUZNw9gv0dEi1ObRvwyogPjFEfGJUfe",
      "connectionMode": "websocket",
      "domain": "feishu",
      "groupPolicy": "open"
    }
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "token",
      "token": "fb80da53c9d07bd68a6a2d24db7d425c0aff1b72e5d49eaf"
    },
    "tailscale": {
      "mode": "off",
      "resetOnExit": false
    },
    "nodes": {
      "denyCommands": [
        "camera.snap",
        "camera.clip",
        "screen.record",
        "contacts.add",
        "calendar.add",
        "reminders.add",
        "sms.send"
      ]
    }
  },
  "plugins": {
    "entries": {
      "feishu": {
        "enabled": true
      }
    },
    "installs": {
      "feishu": {
        "source": "npm",
        "spec": "@openclaw/feishu",
        "installPath": "/Users/wangqiao/.openclaw/extensions/feishu",
        "version": "2026.3.7",
        "resolvedName": "@openclaw/feishu",
        "resolvedVersion": "2026.3.7",
        "resolvedSpec": "@openclaw/feishu@2026.3.7",
        "integrity": "sha512-CHPcL+WHYKYR2HJKRYsRtlXx/wbQRy5axltjjH9qXkR8ghxygDmOHZREjxyFEbjFJ3wnIuvgjLE7JYTg3nPpDA==",
        "shasum": "c4b31dbe2ff0bc7034334873482ad18ac60a0767",
        "resolvedAt": "2026-03-09T07:14:47.397Z",
        "installedAt": "2026-03-09T07:14:50.652Z"
      }
    }
  }
}
```

### 2.3 qwen-plus 配置

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "bailian/qwen-plus"
      },
      "models": {
        "bailian/qwen-plus": {
          "alias": "通义千问 Plus"
        }
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "bailian": {
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "env:DASHSCOPE_API_KEY",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen-plus",
            "name": "通义千问 Plus",
            "reasoning": false,
            "input": ["text"],
            "cost": {
              "input": 0.008,
              "output": 0.008,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 262144,
            "maxTokens": 32000
          }
        ]
      }
    }
  }
}
```

### 2.4 检查配置文件语法合法性

在修改配置文件后，建议检查其语法是否合法，以避免因语法错误导致服务启动失败。可以使用 `jq` 工具来验证 JSON 文件的语法。

#### 安装 jq 工具

```bash
# 在 macOS 上使用 Homebrew 安装
brew install jq

# 在 Linux 上使用包管理器安装
# Ubuntu/Debian
sudo apt-get install jq

# CentOS/RHEL
sudo yum install jq
```

#### 检查配置文件语法

```bash
# 检查配置文件语法是否合法
jq . ~/.openclaw/openclaw.json

# 输出美化后的 JSON（如果语法正确）
# 如果语法错误，会显示错误信息
```

#### 检查结果示例

- **语法正确时**：会输出美化后的 JSON 内容
- **语法错误时**：会显示具体的错误信息，例如：
  ```
  parse error: Expected property name or '}' at line 123, column 5
  ```

## 3. 配置修改后重启网关

### 3.1 操作命令

```bash
# 停止网关服务
pnpm openclaw gateway stop

# 启动网关服务
pnpm openclaw gateway start

# 重启网关服务
pnpm openclaw gateway restart
```

### 3.2 重启执行结果

```bash
➜  openclaw-2026.3.7 git:(main) ✗ pnpm openclaw gateway restart

> openclaw@2026.3.7 openclaw /Users/wangqiao/Downloads/github_project/openclaw-2026.3.7
> node scripts/run-node.mjs gateway restart


🦞 OpenClaw 2026.3.7 (1e94d88) — I'm the middleware between your ambition and your attention span.

Restarted LaunchAgent: gui/501/ai.openclaw.gateway
```

## 4. 网关状态检查

### 4.1 检查命令

```bash
pnpm openclaw gateway status
```

### 4.2 检查结果与关键信息解读

```bash
> node scripts/run-node.mjs gateway status

🦞 OpenClaw 2026.3.7 (1e94d88) — Claws out, commit in—let's ship something mildly responsible.

│
◇
Service: LaunchAgent (loaded)
File logs: /tmp/openclaw/openclaw-2026-03-09.log
Command: /Users/wangqiao/.nvm/versions/node/v24.14.0/bin/node /Users/wangqiao/Downloads/github_project/openclaw-2026.3.7/dist/index.js gateway --port 18789
Service file: ~/Library/LaunchAgents/ai.openclaw.gateway.plist
Service env: OPENCLAW_GATEWAY_PORT=18789

Service config looks out of date or non-standard.
Service config issue: Gateway service uses Node from a version manager; it can break after upgrades. (/Users/wangqiao/.nvm/versions/node/v24.14.0/bin/node)
Recommendation: run "openclaw doctor" (or "openclaw doctor --repair").
Config (cli): ~/.openclaw/openclaw.json
Config (service): ~/.openclaw/openclaw.json

Gateway: bind=loopback (127.0.0.1), port=18789 (service args)
Probe target: ws://127.0.0.1:18789
Dashboard: http://127.0.0.1:18789/
Probe note: Loopback-only gateway; only local clients can connect.

Runtime: running (pid 75209)
RPC probe: ok

Listening: 127.0.0.1:18789
Troubles: run openclaw status
Troubleshooting: https://docs.openclaw.ai/troubleshooting
```

### 4.3 关键信息

- **网关进程状态**：`running (pid 75209)`（正常运行）
- **监听地址/端口**：`127.0.0.1:18789`（仅本地可访问）
- **潜在问题**：Node 路径依赖版本管理器（NVM），升级后可能失效，建议执行 `openclaw doctor --repair` 修复

## 5. 日志查看

### 5.1 查看 OpenClaw 网关的实时日志（macOS LaunchAgent 日志）

```bash
log show --predicate 'process == "node" AND subsystem == "ai.openclaw.gateway"' --info --debug --latest 10m
```

### 5.2 查看进程 stderr/stdout 日志（如果有文件输出）

```bash
# 核心日志文件
cat ~/.openclaw/logs/gateway.log

# 错误日志
cat ~/.openclaw/logs/error.log
```

### 5.3 实时查看日志（使用 OpenClaw 命令）

```bash
# 实时查看 OpenClaw 网关日志
pnpm openclaw logs --follow
```

## 6. Dashboard 访问

### 6.1 访问命令（不自动打开浏览器）

```bash
pnpm clawdbot dashboard
pnpm openclaw dashboard --no-open
```

### 6.2 执行结果与访问地址

```bash
> node scripts/run-node.mjs dashboard --no-open

🦞 OpenClaw 2026.3.7 (1e94d88) — Gateway online—please keep hands, feet, and appendages inside the shell at all times.

Dashboard URL: http://127.0.0.1:18789/#token=fb80da53c9d07bd68a6a2d24db7d425c0aff1b72e5d49eaf
Copied to clipboard.
Browser launch disabled (--no-open). Use the URL above.
```

## 7. 总结

1. **网关服务状态**：已成功安装并运行在本地 `127.0.0.1:18789` 端口，仅支持本地访问
2. **已配置组件**：飞书（websocket 模式）和 MiniMax-M2.5 模型，核心功能可正常使用
3. **潜在问题**：Node 路径依赖版本管理器（NVM），升级后可能失效，建议执行 `openclaw doctor --repair` 优化
4. **Dashboard 访问**：需使用带 token 的专属链接
5. **配置扩展**：已添加 qwen-plus 模型配置，可根据需要切换默认模型
6. **配置验证**：修改配置后建议使用 `jq` 工具检查 JSON 语法合法性，确保服务启动正常
7. **日志查看**：可通过 macOS LaunchAgent 日志或文件日志查看网关运行状态和错误信息
