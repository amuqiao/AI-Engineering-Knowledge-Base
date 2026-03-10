## Git 分支分歧问题解决方案

### 问题分析
从终端输出可以看到，当执行 `git pull` 时出现了分支分歧（divergent branches）的问题。这是因为本地分支和远程分支都有各自的提交，Git 不知道如何自动合并这些更改。
```bash
➜  FastAPI-MVC git:(main) git pull
remote: Enumerating objects: 43, done.
remote: Counting objects: 100% (43/43), done.
remote: Compressing objects: 100% (19/19), done.
remote: Total 34 (delta 16), reused 30 (delta 12), pack-reused 0 (from 0)
Unpacking objects: 100% (34/34), 28.86 KiB | 40.00 KiB/s, done.
From https://github.com/amuqiao/FastAPI-MVC
   d06c5f4..7ed99d6  main       -> origin/main
hint: You have divergent branches and need to specify how to reconcile them.
hint: You can do so by running one of the following commands sometime before
hint: your next pull:
hint: 
hint:   git config pull.rebase false  # merge
hint:   git config pull.rebase true   # rebase
hint:   git config pull.ff only       # fast-forward only
hint: 
hint: You can replace "git config" with "git config --global" to set a default
hint: preference for all repositories. You can also pass --rebase, --no-rebase,
hint: or --ff-only on the command line to override the configured default per
hint: invocation.
fatal: Need to specify how to reconcile divergent branches.
```


### 解决方案
根据 Git 的提示，有三种方式解决这个问题：

1. **使用 merge 方式**（默认行为）：
   ```bash
   git config pull.rebase false  # merge
   ```
   这种方式会创建一个新的合并提交，保留双方的历史记录。

2. **使用 rebase 方式**：
   ```bash
   git config pull.rebase true   # rebase
   ```
   这种方式会将本地提交重新应用到远程分支的最新提交之后，使历史记录更加线性。

3. **只使用 fast-forward 方式**：
   ```bash
   git config pull.ff only       # fast-forward only
   ```
   这种方式只在远程分支是本地分支的直接祖先时才执行 pull，否则会拒绝合并。

### 具体操作步骤
1. **选择一种合并策略**，例如使用 rebase 方式：
   ```bash
   git config pull.rebase true
   ```

2. **再次执行 git pull**：
   ```bash
   git pull
   ```

3. **如果遇到冲突**，解决冲突后执行：
   ```bash
   git rebase --continue
   ```

### 全局配置
如果希望在所有仓库中使用相同的策略，可以使用 `--global` 选项：
```bash
git config --global pull.rebase true
```

这样就可以解决 Git 分支分歧的问题，成功执行 `git pull` 操作。