# Fork 仓库开发分支配置指南

> **场景**：从 GitHub 上 fork 了别人的仓库，想在自己的 fork 中新建分支做开发，
> 即使在 GitHub 页面点击 "Sync fork" 也不影响自己的提交。

---

## 前置命令流（首次配置）

### 第一步：克隆自己的 fork 到本地

```bash
git clone git@github.com:amuqiao/CMeKG_tools.git
cd CMeKG_tools
```

### 第二步：添加上游远程仓库

```bash
# 添加上游（原作者的仓库）
git remote add upstream git@github.com:king-yyf/CMeKG_tools.git

# 验证远程配置
git remote -v
# 预期输出：
# origin    git@github.com:amuqiao/CMeKG_tools.git (fetch)
# origin    git@github.com:amuqiao/CMeKG_tools.git (push)
# upstream  git@github.com:king-yyf/CMeKG_tools.git (fetch)
# upstream  git@github.com:king-yyf/CMeKG_tools.git (push)
```

### 第三步：拉取上游分支信息，确保本地 main 是最新的

```bash
git fetch upstream
git checkout main
git merge upstream/main      # 让本地 main 与上游保持一致
git push origin main         # 同步到自己的 fork
```

### 第四步：基于 main 创建 dev 开发分支

```bash
git checkout -b dev          # 从当前 main 创建并切换到 dev 分支
```

### 第五步：推送 dev 分支到自己的远程仓库

```bash
git push -u origin dev       # -u 建立跟踪关系，以后直接 git push 即可
```

> 至此配置完成，后续所有开发工作在 `dev` 分支上进行。

---

## 当前配置总结

**远程仓库：**
- `origin` → `git@github.com:amuqiao/CMeKG_tools.git`（你自己的 fork）
- `upstream` → `git@github.com:king-yyf/CMeKG_tools.git`（上游原仓库）

**分支结构：**
- `main` — 跟踪 `origin/main`，与上游保持同步，**不在此开发**
- `dev` — 你的开发分支，所有个人修改都在这里提交

---

## 为什么 "Sync fork" 不会影响你的提交

GitHub 的 **"Sync fork"** 按钮只会同步**默认分支**（`main`）。你的 `dev` 分支完全独立，无论怎么点 Sync fork 都不会被修改或删除。

---

## 日常工作流

**1. 在 `dev` 分支上开发和提交：**
```bash
git checkout dev
# ... 修改代码 ...
git add .
git commit -m "feat: 你的修改描述"
git push origin dev
```

**2. 当上游有更新时，手动同步到本地 `main` 再 rebase `dev`：**
```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main          # 更新自己 fork 的 main

git checkout dev
git rebase main               # 把 dev 的提交移到最新 main 之上
git push --force-with-lease origin dev
```

这样即使在 GitHub 页面点了 "Sync fork"，`dev` 分支上你的提交永远是安全的。