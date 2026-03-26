要修改OpenClaw中已配置的飞书机器人Channel的App ID和密钥，您可以按照以下步骤操作：

## 方法一：使用交互式命令更新（推荐）

1. **运行更新命令**：

   ```bash
   pnpm openclaw channels add
   ```

2. **选择飞书通道**：
   - 在Channel列表中选择 **Feishu/Lark (飞书)**

3. **更新凭证**：
   - 当提示输入App Secret时，输入新的App Secret
   - 当提示输入App ID时，输入新的App ID

4. **完成配置**：
   - 按照后续提示完成其他配置选项
   - 系统会自动更新配置文件

## 方法二：直接编辑配置文件

1. **查看配置文件路径**：

   ```bash
   pnpm openclaw config file
   ```

   这会显示配置文件的路径，通常是 `~/.openclaw/openclaw.json`

2. **编辑配置文件**：
   使用文本编辑器打开配置文件，找到飞书通道的配置部分：

   ```json
   "channels": {
     "feishu": {
       "enabled": true,
       "appId": "旧的App ID",
       "appSecret": "旧的App Secret",
       "connectionMode": "websocket",
       "domain": "feishu",
       "groupPolicy": "open"
     }
   }
   ```

3. **更新凭证**：
   将 `appId` 和 `appSecret` 的值替换为新的凭证。

4. **保存文件**：
   保存配置文件后，重启OpenClaw网关使更改生效。

## 方法三：使用环境变量

如果您希望通过环境变量管理凭证，可以：

1. **设置环境变量**：

   ```bash
   export FEISHU_APP_ID="新的App ID"
   export FEISHU_APP_SECRET="新的App Secret"
   ```

2. **更新配置**：
   运行交互式命令并选择使用环境变量：
   ```bash
   pnpm openclaw channels add
   ```
   当提示时，选择 "Use env vars" 选项。

## 验证更改

更新后，您可以使用以下命令验证连接状态：

```bash
pnpm openclaw channels status --probe
```

如果显示连接成功，说明App ID和密钥已成功更新。

## 注意事项

1. **重启网关**：更新凭证后，建议重启OpenClaw网关以使更改生效：

   ```bash
   pnpm openclaw gateway restart
   ```

2. **飞书应用配置**：确保在飞书开放平台上，新的App ID和App Secret对应的应用已经正确配置了所需的权限和事件订阅。

3. **插件冲突**：如果您之前遇到了飞书插件重复的警告，建议先解决插件冲突问题，再进行凭证更新。
