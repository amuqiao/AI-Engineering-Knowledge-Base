
Tavily 是一个专门为 AI 应用设计的搜索 API 服务，提供高质量、结构化的搜索结果。它专为大语言模型（LLM）和智能体设计，能够快速获取准确的信息。

### 如何配置 TAVILY_API_KEY

1. **注册 Tavily 账号**：
    - 访问 [Tavily 官方网站](https://www.tavily.com/)
    - 点击 "Sign Up" 或 "注册" 按钮
    - 完成注册流程，创建您的账号
2. **获取 API 密钥**：
    - 登录您的 Tavily 账号
    - 进入 "API Keys" 或 "API 密钥" 页面
    - 复制您的 API 密钥
3. **设置环境变量**：
    - **在 Linux/Mac 系统中**：
    - `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`
        
        ```bash
        export TAVILY_API_KEY="your_tavily_api_key_here"
        export TAVILY_API_KEY="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"
        ```
        
        您可以将此命令添加到 `~/.bashrc` 或 `~/.zshrc` 文件中，使其在每次启动终端时自动设置。
        
    - **在 Windows 系统中**：

### **步骤 1：通过系统属性设置永久环境变量**
1. **打开系统属性**：
   - 右键点击桌面或文件资源管理器中的“此电脑”，选择“属性”。
   - 或者按下 `Win + Pause` 快捷键直接打开“系统”窗口。

2. **进入高级系统设置**：
   - 在“系统”窗口中，点击左侧的“高级系统设置”。

3. **配置环境变量**：
   - 在弹出的“系统属性”窗口中，点击“环境变量”按钮。
   - 在“环境变量”窗口中，找到“系统变量”区域，点击“新建”按钮。
   - 在“新建系统变量”对话框中：
     - **变量名**：输入 `TAVILY_API_KEY`
     - **变量值**：输入 `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`
     - 点击“确定”保存。

4. **保存设置**：
   - 依次点击“环境变量”和“系统属性”窗口的“确定”按钮，完成设置。


### **步骤 2：让环境变量立即生效**
设置完永久环境变量后，需要让当前打开的终端（如 CMD 或 PowerShell）立即识别新的环境变量，无需重启系统。


#### **方法 1：在 CMD 中刷新环境变量**
1. 打开 CMD 终端（按下 `Win + R`，输入 `cmd` 并回车）。
2. 执行以下命令刷新环境变量：
   ```cmd
   refreshenv
   ```
   > 注意：`refreshenv` 命令需要安装 Chocolatey 或使用 Windows Terminal 中的工具，若不可用，可尝试方法 2。


#### **方法 2：在 PowerShell 中刷新环境变量**
1. 打开 PowerShell 终端（按下 `Win + R`，输入 `powershell` 并回车）。
2. 执行以下命令刷新环境变量：
   ```powershell
   $env:TAVILY_API_KEY = [System.Environment]::GetEnvironmentVariable("TAVILY_API_KEY", "Machine")
   ```
### **步骤 3：验证环境变量是否设置成功**
1. **在 CMD 中验证**：
   ```cmd
   echo %TAVILY_API_KEY%
   ```
   若显示 `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` 则设置成功。

2. **在 PowerShell 中验证**：
   ```powershell
   echo $env:TAVILY_API_KEY
   ```
   若显示 `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` 则设置成功。


### **注意事项**
- 设置永久环境变量后，新打开的终端会自动识别该变量，无需重复刷新。
- 若使用的是 IDE（如 VS Code）或其他应用，可能需要重启该应用以识别新的环境变量。

设置完成后，您就可以运行深度研究智能体，它会使用 Tavily API 进行网络搜索，获取最新的信息。


