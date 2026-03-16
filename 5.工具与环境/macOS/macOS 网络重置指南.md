#### 1. （可选但建议）备份原有网络配置文件
在删除配置前先备份，防止意外问题，执行：
```bash
# 创建备份目录并复制配置文件
mkdir -p ~/NetworkConfigBackup
sudo cp -R /Library/Preferences/SystemConfiguration/* ~/NetworkConfigBackup/
```

#### 2. 刷新 DNS 缓存
```bash
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
```

#### 3. 重置 Wi-Fi 网络接口
```bash
# 1. 查看当前活跃的 Wi-Fi 接口名（确认不是 en1 等）
networksetup -listallhardwareports | grep -A 1 Wi-Fi

# 2. 重置接口（默认 en0，若上一步显示其他名称则替换）
sudo ifconfig en0 down
sleep 2  # 等待2秒确保接口完全关闭
sudo ifconfig en0 up
```

#### 4. 重置网络配置文件
这一步替代「删除配置后重启电脑」，删除配置后直接重启网络管理服务：
```bash
# 删除非关键的网络配置文件（保留原有过滤逻辑）
sudo find /Library/Preferences/SystemConfiguration/ -type f -not -name "com.apple.Boot.plist" -exec sudo rm -f {} \;
```

#### 5.重启电脑

#### 6. 重新连接 Wi-Fi（手动操作）
在 MacBook 右上角的 Wi-Fi 图标中，先断开当前连接，再重新连接你的 Wi-Fi 网络即可。

### 验证网络是否恢复
执行以下命令检查网络连通性：
```bash
# 检查 DNS 解析
nslookup baidu.com

# 检查网络连通性
ping -c 3 8.8.8.8  # 测试 ping 谷歌 DNS
ping -c 3 baidu.com  # 测试域名解析和访问
```


如果优化后仍无法上网，大概率是 VPN 客户端残留了路由规则，可执行 `sudo route -n flush` 清空路由表后再重试。