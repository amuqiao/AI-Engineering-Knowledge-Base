本文整理 macOS 系统下查看 Shell 信息、切换默认 Shell（以 zsh 为例）的完整操作步骤，结构清晰且可直接复用。

## 1. 查看当前 Shell 及版本信息
| 命令 | 作用 | 示例输出（参考） |
|------|------|------------------|
| `echo $SHELL` | 显示当前默认 Shell 的路径 | `/bin/zsh` |
| `$SHELL --version` | 查看当前 Shell 的具体版本 | `zsh 5.9 (x86_64-apple-darwin22.0)` |
| `bash --version` | 单独查看 bash 版本（若安装） | `GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin22)` |
| `which bash` | 查看 bash 可执行文件的路径 | `/bin/bash` |

### 快速执行示例
```bash
# 一键查看核心信息
echo "当前Shell路径：$SHELL"
echo "当前Shell版本：" && $SHELL --version
```

## 2. 查看系统安装的所有 Shell
系统支持的 Shell 列表存储在 `/etc/shells` 文件中，可直接查看：
```bash
cat /etc/shells
```

### 预期输出（macOS 通用）
```bash
# List of acceptable shells for chpass(1).
# Ftpd will not allow users to connect who are not using
# one of these shells.

/bin/bash
/bin/csh
/bin/dash
/bin/ksh
/bin/sh
/bin/tcsh
/bin/zsh
```

## 3. 切换默认 Shell 为 zsh
### 3.1 前提：确认 zsh 是否预装
- macOS 10.15（Catalina）及以上版本：默认预装 zsh，可直接跳过安装步骤。
- macOS 10.14 及以下版本：需先通过 Homebrew 安装 zsh。

### 3.2 旧版本 macOS 安装 zsh（可选）
```bash
# 第一步：安装 Homebrew（若未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 第二步：通过 Homebrew 安装 zsh
brew install zsh
```

### 3.3 执行切换命令
无论是否手动安装，切换默认 Shell 均使用 `chsh` 命令：
```bash
# 切换默认 Shell 为 zsh（路径需与 /etc/shells 中一致）
chsh -s /bin/zsh
```

### 3.4 验证切换是否生效
1. 关闭当前终端窗口，重新打开（**关键步骤**：配置需重启终端生效）；
2. 执行以下命令验证：
   ```bash
   echo $SHELL  # 输出应为 /bin/zsh
   ```

## 总结
1. 查看当前 Shell 核心命令：`echo $SHELL`（路径）、`$SHELL --version`（版本）；
2. 切换默认 Shell 使用 `chsh -s 【Shell路径】`，路径需匹配 `/etc/shells` 中的记录；
3. 切换后需**重启终端**才能生效，旧版 macOS 需先通过 Homebrew 安装 zsh。