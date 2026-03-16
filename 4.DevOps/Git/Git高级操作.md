
# 本地项目关联 Git 仓库并强制推送

```
# ========== 1. 基础准备 ==========
# 进入项目根目录
cd /path/to/your/project          # 替换为实际路径

# 初始化本地仓库（如果尚未初始化）
git init

# ========== 2. 远程仓库管理 ==========
# 添加主远程（origin）
git remote add origin <仓库URL.git>
# 示例：git remote add origin git@github.com:amuqiao/TruFor_1111.git
# 示例：git remote add origin https://github.com/amuqiao/TruFor_Note2.git

# 添加其他远程（如华为云）
git remote add huawei <仓库URL2.git>
# 示例：git remote add huawei git@codehub-cn-south-1.devcloud.huaweicloud.com:xxx/servo_ai.git

# 查看已配置的远程
git remote -v

# 删除远程（需更换时使用）
git remote remove origin

# ========== 3. 提交与推送 ==========
# 添加所有文件到暂存区
git add .

# 提交本地更改
git commit -m "初始提交"

# 将本地分支重命名为 main（按需执行，推送 main 前完成）
git branch -m master main          # 指定从 master 重命名为 main
git branch -M main                 # 将当前分支强制重命名为 main
# 说明：-M 为 -m 的强制版，若目标分支 main 已存在会直接覆盖，常用于统一使用 main 作为主分支

# 强制推送覆盖远端（根据实际分支选择）
# 使用-force参数会完全覆盖远程仓库内容，确保远程仓库没有需要保留的内容
git push --force origin master    # master 分支
git push --force origin main      # main 分支
```