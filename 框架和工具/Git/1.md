可以通过以下几种方式实现一键提交推送：

## 方式一：使用 && 连接命令（最简单）

```bash
git add . && git commit -m "update" && git push
```

## 方式二：创建 Git 别名

```bash
git config --global alias.pushall '!git add -A && git commit -m "update" && git push'
```

之后只需执行：

```bash
git pushall
```

## 方式三：创建 Shell 脚本（推荐）

创建一个可执行脚本文件 `git-push.sh`：

```bash
#!/bin/bash
git add .
git commit -m "${1:-update}"
git push
```

> 说明：`${1:-update}` 是 Shell 参数扩展语法，表示如果传入了第一个参数则使用该参数，否则使用默认值 "update"。

设置可执行权限：

```bash
chmod +x git-push.sh
```

### 使用方法

| 场景 | 命令 |
|------|------|
| 使用默认消息 | `./git-push.sh` |
| 自定义 commit 消息 | `./git-push.sh "新增深度学习笔记"` |
