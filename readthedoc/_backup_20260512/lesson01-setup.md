# 第一课：实验环境搭建 — SSHFS 挂载远程 Linux 目录到 Windows

## 课程目标

本课的目标是搭建一个高效、稳定的开发环境，让同学们能够在 **Windows 本地 VS Code** 中直接编辑远程 **Linux 服务器（192.168.3.10）** 上的文件，就像操作本地文件一样。

整个方案包含：

1. **一次性环境搭建**：安装驱动 → 配置 SSH → 挂载为 Z: 盘 → 创建符号链接
2. **日常一键使用**：每次开机按快捷键即可恢复挂载

```{note}
为什么不用 VS Code 的 Remote-SSH？
Remote-SSH 需要在远程安装 VS Code Server，对老旧系统（如 Red Hat Linux 9/Ubuntu 旧版）兼容性较差。SSHFS-Win 只依赖标准 SSH 协议，无需在远程安装任何额外软件。
```

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 远程服务器 IP | `192.168.3.10` |
| 远程操作系统 | Ubuntu（基于 Red Hat Linux 9） |
| 远程用户 | `zzl` |
| 远程密码 | `123456` |
| 远程家目录 | `/home/zzl` |
| SSH 端口 | 22（默认） |
| 本地操作系统 | Windows |
| 本地工作区 | `C:\zjl\teach\Linux` |
| 本地挂载目标 | `C:\zjl\teach\Linux\192.168.3.10` |
| 挂载盘符 | `Z:` |
| 挂载工具 | WinFsp + SSHFS-Win |

---

## 第〇步：安装 WinFsp 和 SSHFS-Win（一次性）

打开 VS Code，按 `Ctrl + `` 打开终端（确保终端类型为 **PowerShell**），依次执行：

```powershell
# 安装 WinFsp（SSHFS 底层驱动）
winget install WinFsp.WinFsp

# 安装 SSHFS-Win
winget install SSHFS-Win.SSHFS-Win
```

安装完成后，**建议注销当前用户后重新登录**，以确保驱动正确加载。

### 验证安装

```powershell
winget list --name WinFsp
winget list --name SSHFS
```

预期输出中应能看到 `WinFsp.WinFsp` 和 `SSHFS-Win.SSHFS-Win` 的版本号。

---

## 第一步：配置 SSH 客户端（一次性）

### 1.1 创建/编辑 SSH 配置文件

在 VS Code 中按 `Ctrl + O`，打开（或创建）文件：

```
C:\Users\zjl\.ssh\config
```

写入以下内容：

```
Host redhat-server
    User zzl
    HostName 192.168.3.10
    KexAlgorithms +diffie-hellman-group-exchange-sha1,diffie-hellman-group1-sha1
    HostKeyAlgorithms +ssh-rsa,ssh-dss
    Ciphers +aes128-cbc,3des-cbc,aes256-cbc
    MACs hmac-sha1,hmac-sha1-96
```

```{note}
为什么需要算法配置？
本例中的远程服务器使用较老的 SSH 版本。OpenSSH 8.8+ 已默认禁用这些老旧算法，不配置会导致连接时报错 `no matching key exchange method`。如果是新版 Linux 服务器，可省略 `KexAlgorithms` 等四行。
```

### 1.2 测试 SSH 连接

```powershell
ssh redhat-server "hostname; whoami; pwd; ls | head -10"
```

输入密码 `123456`，成功后应看到类似输出：

```
ubuntu
zzl
/home/zzl
Desktop
Documents
Downloads
...
```

如果连接失败，请检查：

- 服务器 IP 是否可达：`ping 192.168.3.10`
- SSH 端口是否开放：`Test-NetConnection -ComputerName 192.168.3.10 -Port 22`

---

## 第二步：挂载远程目录到 Z: 盘（一次性）

### 2.1 执行挂载命令

在 VS Code 的 **PowerShell** 终端中执行：

```powershell
# 先删除旧的 Z 盘映射（如果有）
net use Z: /delete

# 挂载远程 /home/zzl 到本地 Z: 盘
net use Z: "\\sshfs\zzl@192.168.3.10" 123456
```

命令格式说明：

- `Z:` — 本地盘符，可更换为其他未占用的字母
- `\\sshfs\` — SSHFS 协议前缀，固定写法
- `zzl@192.168.3.10` — 用户名@服务器IP
- `123456` — 远程用户密码

提示 `命令成功完成` 即表示挂载成功。

### 2.2 查看挂载状态

```powershell
net use
```

预期输出中应看到 `Z:` 盘映射到 `\\sshfs\zzl@192.168.3.10`。此时在 Windows 资源管理器地址栏输入 `Z:\` 即可直接访问。

### 2.3 验证 Z: 盘内容

```powershell
Get-ChildItem Z:\ | Select-Object Name, LastWriteTime
```

---

## 第三步：创建符号链接到工作区（一次性）

Z: 盘虽可用，但在 VS Code 侧边栏中访问不便。我们需要在工作区目录下创建符号链接。

### 3.1 删除旧目录（如有）

```powershell
Remove-Item -Recurse -Force "C:\zjl\teach\Linux\192.168.3.10"
```

### 3.2 以管理员身份创建符号链接

Z: 盘是网络驱动器，必须使用 `mklink /D` 且需要管理员权限。选择以下任一方式：

**方式一：管理员 PowerShell 窗口**

右键开始菜单 → **Windows PowerShell (管理员)**，执行：

```powershell
cmd /c mklink /D "C:\zjl\teach\Linux\192.168.3.10" "Z:\"
```

**方式二：管理员 VS Code**

右键 VS Code 图标 → **以管理员身份运行**，打开工作区后在终端中执行上述命令。

**方式三：VS Code 终端中提权（不切换窗口）**

```powershell
Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -Command "cmd /c mklink /D \"C:\zjl\teach\Linux\192.168.3.10\" \"Z:\\\"; Write-Host Done; pause"'
```

系统弹出 UAC 窗口，点击"是"即可。成功后显示：

```
为 C:\zjl\teach\Linux\192.168.3.10 <<===>> Z:\ 创建的符号链接
```

### 3.3 验证符号链接

```powershell
Get-Item "C:\zjl\teach\Linux\192.168.3.10" | Select-Object Name, LinkType, Target
```

回到 VS Code 文件资源管理器（`Ctrl + Shift + E`），展开工作区即可看到 `192.168.3.10` 目录。

```{tip}
符号链接只需创建一次。即使 Z: 盘断开重新挂载，链接自动恢复生效，无需重新创建。
```

---

## 日常一键挂载

系统重启后 Z: 盘映射会消失，但符号链接仍在。只需重新挂载 Z: 盘。

### 方式一：双击 .bat 文件（最简单）

在 VS Code 侧边栏右键 `挂载192.168.3.10.bat` → **在文件资源管理器中显示** → 双击运行。

挂载脚本内容：

```batch
@echo off
echo Mounting 192.168.3.10...
net use Z: /delete >nul 2>&1
net use Z: "\\sshfs\zzl@192.168.3.10" 123456
if %errorlevel% equ 0 (
    echo Mounted! 192.168.3.10 ready.
) else (
    echo Mount failed. Check network.
)
pause
```

卸载脚本内容：

```batch
@echo off
echo Unmounting 192.168.3.10...
net use Z: /delete
pause
```

### 方式二：VS Code 快捷键（最便捷）

在 `.vscode/tasks.json` 中配置任务：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "挂载 192.168.3.10",
            "type": "shell",
            "command": "挂载192.168.3.10.bat",
            "options": { "cwd": "C:\\zjl\\teach\\Linux" },
            "problemMatcher": []
        },
        {
            "label": "卸载 192.168.3.10",
            "type": "shell",
            "command": "卸载192.168.3.10.bat",
            "options": { "cwd": "C:\\zjl\\teach\\Linux" },
            "problemMatcher": []
        }
    ]
}
```

在 `.vscode/keybindings.json` 中绑定快捷键：

```json
[
    {
        "key": "ctrl+shift+m",
        "command": "workbench.action.tasks.runTask",
        "args": "挂载 192.168.3.10"
    },
    {
        "key": "ctrl+shift+u",
        "command": "workbench.action.tasks.runTask",
        "args": "卸载 192.168.3.10"
    }
]
```

配置后：

- `Ctrl + Shift + M` — 一键挂载
- `Ctrl + Shift + U` — 一键卸载

### 方式三：PowerShell 脚本

```powershell
# mount_192.168.3.10.ps1
net use Z: /delete 2>$null | Out-Null
net use Z: "\\sshfs\zzl@192.168.3.10" 123456
if ($LASTEXITCODE -eq 0) {
    Write-Host "Mounted! 192.168.3.10 ready." -ForegroundColor Green
} else {
    Write-Host "Mount failed." -ForegroundColor Red
}
```

---

## 完整操作时序

### 初次环境搭建（仅一次）

1. **安装驱动**：`winget install WinFsp.WinFsp` 和 `winget install SSHFS-Win.SSHFS-Win`
2. **重启登录**：注销后重新登录
3. **配置 SSH**：创建 `~/.ssh/config`
4. **测试连接**：`ssh redhat-server`
5. **挂载 Z: 盘**：`net use Z: "\\sshfs\zzl@192.168.3.10" 123456`
6. **创建符号链接**：以管理员身份执行 `mklink /D`
7. **准备一键脚本**：配置 `.vscode/tasks.json` 和 `keybindings.json`

### 日常开机（每次）

1. 打开 VS Code，打开 `C:\zjl\teach\Linux` 文件夹
2. 按 `Ctrl + Shift + M`
3. 等待终端显示 `Mounted!`
4. 在侧边栏展开 `192.168.3.10`，开始工作

---

## 常见问题排查

### Q: `net use` 提示"找不到网络路径"

先测试 SSH 连接：`ssh zzl@192.168.3.10 "echo ok"`，如果不通则检查网络和 SSH 服务。

### Q: 提示"多重连接"或"设备已被使用"

```powershell
net use Z: /delete /y
net use Z: "\\sshfs\zzl@192.168.3.10" 123456
```

### Q: `mklink` 提示"需要管理员权限"

必须以管理员身份运行 PowerShell/CMD 才能创建符号链接。

### Q: `mklink /J` 提示"完成该操作需要本地卷"

`/J`（目录连接点）不支持网络驱动器，必须使用 `/D`（符号链接）。

### Q: 保存文件时很慢

SSHFS 通过网络传输，大文件会有延迟。代码和文档等小文件延迟可接受。

### Q: 密码泄露风险

密码以明文出现在命令中，可通过 SSH 密钥认证代替：

```bash
ssh-keygen
ssh-copy-id zzl@192.168.3.10
# 使用密钥后挂载命令可省略密码
net use Z: "\\sshfs\zzl@192.168.3.10"
```

---

## 卸载与清理

```powershell
# 1. 卸载 Z: 盘
net use Z: /delete

# 2. 删除符号链接（仅删除链接，不影响远程数据）
cmd /c rmdir "C:\zjl\teach\Linux\192.168.3.10"

# 3. 如需彻底卸载工具
winget uninstall SSHFS-Win.SSHFS-Win
winget uninstall WinFsp.WinFsp
```

---

## 小结

通过 WinFsp + SSHFS-Win + VS Code 的组合，我们实现了：

- 无需在远程服务器安装任何额外软件
- 兼容老旧 SSH 服务器
- 在 VS Code 工作区中直接编辑远程文件，修改实时同步
- 一次搭建，永久受益；每次开机一键挂载

---

## 课后练习

```{admonition} 练习 1：验证环境
:class: tip

完成本课的全部搭建步骤后，在 VS Code 侧边栏的 `192.168.3.10` 目录中创建一个名为 `test.txt` 的文件，写入你的学号和姓名。然后通过 SSH 登录服务器，确认文件已出现在远程 `/home/zzl/` 目录下。
```

```{admonition} 练习 2：配置快捷键
:class: tip

在 VS Code 中配置 `Ctrl + Shift + M`（挂载）和 `Ctrl + Shift + U`（卸载）快捷键，并实际操作验证。
```

```{admonition} 练习 3：SSH 密钥认证
:class: tip

（加分项）配置 SSH 密钥认证以替代密码，使挂载命令中不再需要明文密码。
```
