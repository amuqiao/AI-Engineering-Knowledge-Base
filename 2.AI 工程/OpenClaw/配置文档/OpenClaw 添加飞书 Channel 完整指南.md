根据终端输出日志，我为您整理了添加飞书 Channel 的详细步骤和推荐选择：

## 一、添加飞书 Channel 步骤

### 1. 启动添加流程

```bash
pnpm openclaw channels add
```

### 2. 选择 Channel 类型

- 在 Channel 列表中选择 **Feishu/Lark (飞书)**

### 3. 准备飞书凭证

根据提示，需要：

1. 访问 https://open.feishu.cn/app
2. 创建自建应用
3. 获取 App ID 和 App Secret
4. 为应用开启机器人能力
5. 为应用开通相关权限
6. 发布应用或添加到测试组

#### 3.1 开启机器人能力

- 在飞书开放平台进入应用详情页
- 点击「能力」选项卡
- 找到「机器人」能力并开启

#### 3.2 配置权限

批量导入以下权限：

```json
{
  "scopes": {
    "tenant": [
      "cardkit:card:write",
      "contact:contact.base:readonly",
      "contact:user.base:readonly",
      "im:chat:readonly",
      "im:message",
      "im:message.group_at_msg:readonly",
      "im:message.group_msg",
      "im:message.p2p_msg:readonly",
      "im:message.reactions:read",
      "im:message:readonly",
      "im:message:recall",
      "im:message:send_as_bot",
      "im:message:update",
      "im:resource"
    ],
    "user": ["contact:contact.base:readonly"]
  }
}
```

#### 3.3 订阅事件

⚠️ 注意：事件订阅操作一定要在第2步接入飞书 channel 之后进行，否则选择长连接会失败

- 在飞书开放平台进入应用详情页
- 点击「事件与回调」选项卡
- 订阅方式选择默认推荐的 **长连接**
- 添加上以下事件：
  - `im.message.receive_v1` 接收消息（必需）
  - `im.message.message_read_v1` 消息已读回执
  - `im.chat.member.bot.added_v1` 机器人进群
  - `im.chat.member.bot.deleted_v1` 机器人被移除群聊

#### 3.4 创建版本并发布

- 在飞书开放平台进入应用详情页
- 点击「版本管理与发布」选项卡
- 创建新版本并提交审核
- 审核通过后发布应用，或添加到测试组

### 4. 输入凭证

- **App Secret**: 输入从飞书开放平台获取的 App Secret
- **App ID**: 输入从飞书开放平台获取的 App ID

### 5. 连接测试

系统会自动测试连接，成功后显示：

```
[info]: [ 'client ready' ]
Connected as ou_7a2ba91ffb458522ec4cefd786ec039f
```

### 6. 配置选项

#### 6.1 飞书连接模式

- **终端显示**: `◇  Feishu connection mode`
- **选项**: WebSocket (default)
- **推荐选择**: 保持默认，使用 WebSocket 模式
- **说明**: WebSocket 模式使用长连接接收事件，不需要暴露公网 URL

#### 6.2 飞书域名

- **终端显示**: `◇  Which Feishu domain?`
- **选项**: Feishu (feishu.cn) - China
- **推荐选择**: 根据实际使用的飞书域名选择
- **说明**: 选择适合您所在地区的飞书域名

#### 6.3 群聊策略

- **终端显示**: `◇  Group chat policy`
- **选项**: Open - respond in all groups (requires mention)
- **推荐选择**: 保持默认的 Open 模式
- **说明**: Open 模式下，机器人会在所有群聊中响应，但需要 @ 机器人

#### 6.4 选择完成

- **终端显示**: `◇  Select a channel`
- **选项**: Finished

#### 6.5 确认选择的通道

- **终端显示**: `◇  Selected channels`
- **显示内容**: Feishu — 飞书/Lark enterprise messaging

#### 6.6 DM 访问策略

- **终端显示**: `◇  Configure DM access policies now? (default: pairing)`
- **选项**: No (使用默认的 pairing 策略)
- **推荐选择**: 选择 No，使用默认的 pairing 策略
- **说明**:
  - `pairing` 策略：未知用户发送消息时会收到配对码，需要通过 `openclaw pairing approve feishu <code>` 批准
  - `open` 策略：允许所有人发送消息，不需要配对
  - `allowlist` 策略：只允许白名单用户发送消息

#### 6.7 添加显示名称

- **终端显示**: `◇  Add display names for these accounts? (optional)`
- **选项**: No
- **推荐选择**: 选择 No，使用默认账户 ID
- **说明**: 显示名称用于在 OpenClaw 界面中识别账户，单账户场景下不需要额外设置

#### 6.8 绑定到代理

- **终端显示**: `◇  Bind configured channel accounts to agents now?`
- **选项**: Yes
- **推荐选择**: 选择 Yes，将通道绑定到默认代理
- **说明**: 绑定后，来自飞书的消息会正确路由到指定的 AI 助手

#### 6.9 选择代理

- **终端显示**: `◇  Route feishu account "default" to agent`
- **选项**: main (default)
- **推荐选择**: 保持默认，绑定到 main 代理
- **说明**: main 是默认代理，适合大多数使用场景

#### 6.10 确认绑定

- **终端显示**: `◇  Routing bindings`
- **显示内容**: Added: feishu accountId=default

### 7. 完成配置

系统会自动更新配置文件：

```
Config overwrite: C:\Users\97821\.openclaw\openclaw.json (sha256 d9a9b0ea3d4120f8c555ce259b30f9c43e450784bc14eb4210d0cf96e1af0061 -> 833bc48aa4306fef7fa9586bc8294cb3cc3f321dc606f7dccbd47e13aba279b9, backup=C:\Users\97821\.openclaw\openclaw.json.bak)
Channels updated.
```

## 二、关键注意事项

1. **权限配置**: 确保在飞书开放平台启用了必要的权限，否则连接会失败
2. **应用发布**: 飞书应用需要发布或添加到测试组才能正常使用
3. **绑定验证**: 首次使用时，需要通过配对码批准用户
4. **连接模式**: 推荐使用 WebSocket 模式，配置更简单且不需要公网访问
5. **群聊设置**: Open 模式下，机器人需要被 @ 才会响应，这是正常行为

## 三、后续操作

1. **启动网关**:

   ```bash
   openclaw gateway
   ```

2. **接入飞书群聊**:
   - 用手机或者电脑打开飞书客户端
   - 创建一个测试群
   - 将机器人拉入群聊
   - @OpenClaw 打个招呼
   - OpenClaw 可以正常回复就说明飞书接入成功了

3. **测试消息**:
   - 在飞书中向机器人发送消息
   - 使用配对码批准：`openclaw pairing approve feishu <配对码>`

4. **查看状态**:

   ```bash
   openclaw gateway status
   openclaw logs --follow
   ```

## 四、故障排除

如果遇到连接问题：

1. 检查飞书应用是否已发布
2. 确认已启用必要的权限
3. 验证 App ID 和 App Secret 是否正确
4. 确保网关正在运行
5. 查看日志获取详细错误信息

希望这份指南能帮助您成功添加和配置飞书 Channel！

TODO
飞书 Channel 断开连接的解决方案
