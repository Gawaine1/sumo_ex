# 第二课：虚拟 LED 字符设备驱动实验

## 实验目的

- 深入理解 Linux **"一切皆文件"** 的设计思想，掌握设备文件与普通文件的异同
- 学习字符设备驱动的基本框架，熟悉 `file_operations` 结构体的使用
- 掌握内核模块的编写、编译、加载与卸载方法
- 通过虚拟 LED 设备，体会操作系统如何将"硬件"抽象为文件接口（**无需真实硬件**）

```{note}
本实验对应操作系统原理大纲中的资源抽象（★1 操作系统的目标和作用）、文件系统（1. 文件与文件系统），以及 Linux 体系与编程大纲中的文件处理（★2）和系统管理（★3 设备驱动管理）。
```

---

## 实验器材

### 软件

- Linux 操作系统（推荐 Ubuntu 20.04/22.04，可使用虚拟机或远程服务器 192.168.3.10）
- 内核头文件（与当前运行内核版本一致）
- 编译工具链：`build-essential`、`make`、`gcc`
- 文本编辑器（vim / VS Code 等）

### 硬件

- **无需任何外部硬件**。本实验使用全局变量 `vled_state` 模拟 LED 状态。

---

## 实验原理

### 1. Linux 设备文件

在 Linux 中，硬件设备被抽象为文件，位于 `/dev` 目录下。用户程序可以使用标准的文件操作函数（`open`、`read`、`write`、`close`）与设备交互，而无需关心底层硬件细节。

### 2. 字符设备驱动框架

字符设备是 Linux 驱动中最基本的设备类型，以字节流形式进行数据传输。驱动需要实现 `struct file_operations` 中的相关函数，并通过 `register_chrdev` 向内核注册。当用户程序对设备文件进行操作时，VFS（虚拟文件系统）会根据设备号找到对应的驱动程序，并调用注册的函数。

### 3. 虚拟 LED 设计

本实验使用一个全局变量 `vled_state` 模拟 LED 状态：

- `0` 表示灭（OFF）
- `1` 表示亮（ON）

驱动提供了以下接口：

| 操作 | 说明 |
|------|------|
| `write` | 接收用户写入的字符 `'1'` 或 `'0'`，修改 `vled_state` |
| `read` | 读取 `vled_state` 并返回给用户 |
| `/proc/vled_status` | proc 文件，用于查看当前 LED 状态，方便调试 |

### 4. 用户空间与内核空间的数据传递

由于内核空间和用户空间的内存区域相互隔离，驱动不能直接使用用户传递的指针，必须借助专用函数：

- `copy_from_user(to, from, n)` — 将数据从用户空间拷贝到内核空间
- `copy_to_user(to, from, n)` — 将数据从内核空间拷贝到用户空间

```{important}
直接在内核中解引用用户空间指针会导致内核 panic 或安全漏洞。`copy_from_user` 和 `copy_to_user` 会校验地址合法性，确保安全。
```

---

## 代码文件总览

本实验的全部代码位于远程服务器 `~/teach/code/` 目录（在 VS Code 中展开 `192.168.3.10/teach/code/` 即可查看）。

| 文件名 | 说明 |
|--------|------|
| `vled_driver.c` | 基础版虚拟LED字符设备驱动（约 120 行） |
| `vled_driver_mutex.c` | 加分版：添加互斥锁，防止并发写入冲突 |
| `Makefile` | 编译脚本，同时构建两个驱动版本 |
| `test_vled.c` | C 语言测试程序 |
| `test_vled.sh` | Shell 一键测试脚本 |

---

## 任务 1：搭建开发环境

在 Linux 服务器（`192.168.3.10`）上执行：

```bash
sudo apt update
sudo apt install build-essential linux-headers-$(uname -r)
```

验证安装：

```bash
gcc --version
make --version
ls /lib/modules/$(uname -r)/build
```

```{tip}
如果你在 VS Code 中通过 SSHFS 挂载操作远程服务器，可以直接在 VS Code 终端中使用 `ssh redhat-server` 登录到远程服务器执行上述命令。也可以直接在 VS Code 终端中通过 SSH 执行（课程环境中 VS Code 终端默认即可 SSH 到远程）。
```

---

## 任务 2：编写驱动代码

### 2.1 源代码：`vled_driver.c`

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

#define DEVICE_NAME "vled_device"
#define PROC_NAME   "vled_status"

static int major_num;
static int vled_state = 0;  // 0: off, 1: on

/* ── proc file read function ── */
static int vled_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%d\n", vled_state);
    return 0;
}

static int vled_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, vled_proc_show, NULL);
}

static const struct file_operations vled_proc_fops = {
    .owner   = THIS_MODULE,
    .open    = vled_proc_open,
    .read    = seq_read,
    .llseek  = seq_lseek,
    .release = single_release,
};

/* ── file operation function declarations ── */
static int     vled_open(struct inode *inodep, struct file *filep);
static ssize_t vled_read(struct file *filep, char __user *buffer,
              size_t len, loff_t *offset);
static ssize_t vled_write(struct file *filep, const char __user *buffer,
              size_t len, loff_t *offset);
static int     vled_release(struct inode *inodep, struct file *filep);

/* ── file_operations structure ── */
static struct file_operations fops = {
    .owner   = THIS_MODULE,
    .open    = vled_open,
    .read    = vled_read,
    .write   = vled_write,
    .release = vled_release,
};

/* ── Module initialization ── */
static int __init vled_init(void)
{
    /* Register character device */
    major_num = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_num < 0) {
        printk(KERN_ALERT "vled_device: Failed to register device\n");
        return major_num;
    }
    printk(KERN_INFO "vled_device: registered with major number %d\n",
           major_num);

    /* Create proc entry */
    proc_create(PROC_NAME, 0444, NULL, &vled_proc_fops);
    printk(KERN_INFO "vled_device: proc entry created\n");

    vled_state = 0;
    printk(KERN_INFO "vled_device: initialized\n");
    return 0;
}

/* ── Module exit ── */
static void __exit vled_exit(void)
{
    remove_proc_entry(PROC_NAME, NULL);
    unregister_chrdev(major_num, DEVICE_NAME);
    printk(KERN_INFO "vled_device: unloaded\n");
}

/* ================================================================
 * Functions to be completed by students
 * ================================================================ */

static int vled_open(struct inode *inodep, struct file *filep)
{
    /* TODO: Add necessary open logic, can simply return 0 */
    return 0;
}

static int vled_release(struct inode *inodep, struct file *filep)
{
    /* TODO: Cleanup when releasing, can simply return 0 */
    return 0;
}

static ssize_t vled_read(struct file *filep, char __user *buffer,
              size_t len, loff_t *offset)
{
    char buf[2];
    int bytes_to_copy;

    if (*offset > 0)
        return 0;  /* EOF */

    buf[0] = vled_state ? '1' : '0';
    buf[1] = '\n';

    bytes_to_copy = (len < 2) ? len : 2;
    if (copy_to_user(buffer, buf, bytes_to_copy))
        return -EFAULT;

    *offset += bytes_to_copy;
    return bytes_to_copy;
}

static ssize_t vled_write(struct file *filep, const char __user *buffer,
              size_t len, loff_t *offset)
{
    char cmd;

    if (copy_from_user(&cmd, buffer, 1))
        return -EFAULT;

    if (cmd == '1')
        vled_state = 1;
    else if (cmd == '0')
        vled_state = 0;

    return len;
}

module_init(vled_init);
module_exit(vled_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A virtual LED driver for educational purpose");
```

### 2.2 代码逐段讲解

#### 2.2.1 `file_operations` 结构体 — 将 LED 抽象为文件

```c
static struct file_operations fops = {
    .owner   = THIS_MODULE,
    .open    = vled_open,
    .read    = vled_read,
    .write   = vled_write,
    .release = vled_release,
};
```

这是驱动的核心。`file_operations` 是一个函数指针表，它将 VFS（虚拟文件系统）的用户文件操作（`read`/`write`/`open`）映射到具体的驱动函数。当用户程序对设备文件调用 `read()` 时，内核通过 **VFS → 设备号 → file_operations → vled_read** 这条路径找到并执行驱动中的读函数。

#### 2.2.2 全局变量模拟 LED 状态

```c
static int vled_state = 0;  // 0: off, 1: on
```

真实的 LED 驱动会操作 GPIO 引脚，而我们的虚拟版本用内存中的一个整数来"模拟"LED，原理完全相同。

#### 2.2.3 proc 文件系统

```c
/* 在 vled_init 中 */
proc_create(PROC_NAME, 0444, NULL, &vled_proc_fops);

/* 在 vled_exit 中 */
remove_proc_entry(PROC_NAME, NULL);
```

`/proc` 是 Linux 中的一个虚拟文件系统，用于内核与用户空间的通信。我们创建 `/proc/vled_status`，用户可以通过 `cat /proc/vled_status` 查看 LED 状态。

`proc_ops` 定义了 proc 文件的行为：

```c
static int vled_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%d\n", vled_state);
    return 0;
}

static const struct file_operations vled_proc_fops = {
    .owner   = THIS_MODULE,
    .open    = vled_proc_open,
    .read    = seq_read,
    .llseek  = seq_lseek,
    .release = single_release,
};
```

#### 2.2.4 `vled_write` — 用户空间到内核空间的数据传递

```c
static ssize_t vled_write(struct file *filep, const char __user *buffer,
              size_t len, loff_t *offset)
{
    char cmd;

    if (copy_from_user(&cmd, buffer, 1))
        return -EFAULT;

    if (cmd == '1')
        vled_state = 1;
    else if (cmd == '0')
        vled_state = 0;

    return len;
}
```

关键函数 `copy_from_user` 将用户空间的 1 个字节安全地拷贝到内核空间的 `cmd` 变量中。注意 `buffer` 参数带有 `__user` 标记——这是内核文档化的提醒，告诉开发者这个指针来自用户空间，不能直接解引用。

#### 2.2.5 `vled_read` — 内核空间到用户空间的数据传递

```c
static ssize_t vled_read(struct file *filep, char __user *buffer,
              size_t len, loff_t *offset)
{
    char buf[2];
    int bytes_to_copy;

    if (*offset > 0)
        return 0;  /* EOF */

    buf[0] = vled_state ? '1' : '0';
    buf[1] = '\n';

    bytes_to_copy = (len < 2) ? len : 2;
    if (copy_to_user(buffer, buf, bytes_to_copy))
        return -EFAULT;

    *offset += bytes_to_copy;
    return bytes_to_copy;
}
```

`copy_to_user` 将内核空间的 `buf` 数据安全地拷贝到用户空间 `buffer`。`*offset` 管理文件读写位置，确保重复读取时能正确返回 EOF（End Of File）。

#### 2.2.6 模块初始化与退出

```c
module_init(vled_init);
module_exit(vled_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A virtual LED driver for educational purpose");
```

- `module_init()` 指定模块加载时执行的函数
- `module_exit()` 指定模块卸载时执行的函数
- `MODULE_LICENSE("GPL")` 声明许可证（GPL 是内核模块的常见选择）

---

## 任务 3：编写 Makefile

```makefile
# Build both basic and mutex versions
obj-m += vled_driver.o vled_driver_mutex.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean
	rm -f test_vled

# Build test program
test: test_vled.c
	gcc test_vled.c -o test_vled

.PHONY: all clean test
```

```{note}
**关键解释：**
- `obj-m += vled_driver.o` — 告诉内核编译系统将 `vled_driver.c` 编译为内核模块（`.ko`）
- `$(KDIR)` — 内核源码路径，通过 `uname -r` 自动获取当前运行内核的版本
- `-C $(KDIR) M=$(PWD)` — 跳转到内核源码目录，但模块源码在当前目录（`M=`）编译
```

---

## 任务 4：编译与测试

### 4.1 编译驱动模块

```bash
cd ~/teach/code
make
```

编译成功后会生成：
- `vled_driver.ko` — 基础版内核模块
- `vled_driver_mutex.ko` — 加分版内核模块（带互斥锁）

### 4.2 加载模块

```bash
sudo insmod vled_driver.ko
```

### 4.3 获取主设备号

```bash
dmesg | tail
```

应能看到类似输出：

```
vled_device: registered with major number 240
vled_device: proc entry created
vled_device: initialized
```

请记下输出的主设备号（上例中为 240）。

### 4.4 创建设备文件

```bash
sudo mknod /dev/vled_device c 240 0   # 将 240 替换为实际的主设备号
sudo chmod 666 /dev/vled_device
```

- `c` 表示字符设备（character device）
- `240` 是主设备号
- `0` 是次设备号

### 4.5 测试虚拟 LED 控制

**Shell 命令测试：**

```bash
# 点亮虚拟 LED
echo 1 > /dev/vled_device
cat /proc/vled_status       # 应输出 1

# 熄灭虚拟 LED
echo 0 > /dev/vled_device
cat /proc/vled_status       # 应输出 0

# 读取设备文件
cat /dev/vled_device         # 应输出当前状态（如 0 或 1）
```

### 4.6 测试程序：`test_vled.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main(void)
{
    int  fd;
    char buf[4];

    /* 打开设备文件 — 与打开普通文件语法完全一致 */
    fd = open("/dev/vled_device", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    /* 点亮 LED */
    printf("Turning virtual LED ON ...\n");
    if (write(fd, "1", 1) < 0) {
        perror("write");
        close(fd);
        return 1;
    }
    sleep(1);

    /* 回读验证 */
    lseek(fd, 0, SEEK_SET);
    memset(buf, 0, sizeof(buf));
    if (read(fd, buf, sizeof(buf) - 1) > 0)
        printf("Current LED state: %s", buf);

    /* 熄灭 LED */
    printf("Turning virtual LED OFF ...\n");
    if (write(fd, "0", 1) < 0) {
        perror("write");
        close(fd);
        return 1;
    }
    sleep(1);

    /* 回读验证 */
    lseek(fd, 0, SEEK_SET);
    memset(buf, 0, sizeof(buf));
    if (read(fd, buf, sizeof(buf) - 1) > 0)
        printf("Current LED state: %s", buf);

    close(fd);
    return 0;
}
```

编译运行：

```bash
gcc test_vled.c -o test_vled
./test_vled
```

### 4.7 一键测试脚本

我们提供了一键测试脚本 `test_vled.sh`，可自动完成编译→加载→创建节点→测试→C程序测试的全流程：

```bash
sudo bash test_vled.sh
```

### 4.8 卸载模块

```bash
sudo rmmod vled_driver
```

---

## 工程实践：VS Code + SSH 远程开发流程

本课程推荐的工作方式：

```text
Windows VS Code ──SSHFS挂载──▶ Z: 盘 ──符号链接──▶ 192.168.3.10 目录
       │                                              │
       │ VS Code 编辑                                 │ 远程文件
       ▼                                              ▼
   SSH 终端 ──────────────── make / insmod ──────▶ 编译测试
```

1. 在 VS Code 侧边栏编辑 `192.168.3.10/teach/code/` 下的代码
2. 在 VS Code 终端中通过 SSH 登录远程服务器
3. 在远程终端中执行 `make`、`insmod`、测试命令

这样代码编辑体验与本地完全一致，而编译测试在真实的 Linux 环境中运行。

---

## 加分项：互斥锁（`vled_driver_mutex.c`）

针对思考题中的并发访问问题，`vled_driver_mutex.c` 添加了互斥锁保护：

```c
#include <linux/mutex.h>

static DEFINE_MUTEX(vled_mutex);  // 定义互斥锁

/* 在 vled_write 中使用互斥锁 */
static ssize_t vled_write(...)
{
    // ... copy_from_user ...

    mutex_lock(&vled_mutex);       // 加锁
    if (cmd == '1')
        vled_state = 1;
    else if (cmd == '0')
        vled_state = 0;
    mutex_unlock(&vled_mutex);     // 解锁

    return len;
}

/* 在 vled_read 中也使用互斥锁，确保读到一致的状态 */
static ssize_t vled_read(...)
{
    mutex_lock(&vled_mutex);
    buf[0] = vled_state ? '1' : '0';
    mutex_unlock(&vled_mutex);
    // ...
}
```

```{note}
**原理说明：** 当两个进程同时写入时，mutex 确保同一时刻只有一个进程能进入临界区修改 `vled_state`，避免了经典的 read-modify-write 竞态条件。读操作也加锁是为了确保读到的状态是一致的（不会出现"读到一半被另一个进程修改"的情况）。
```

测试互斥锁版本：

```bash
sudo insmod vled_driver_mutex.ko
# 设备文件和 proc 文件名与基础版相同，操作方式一致
# 可以打开两个终端同时写入，测试并发场景
sudo rmmod vled_driver_mutex
```

---

## 思考题

```{admonition} 思考题 1
:class: note

什么是设备文件？`/dev/vled_device` 与普通文件（如 `/home/user/test.txt`）在操作上有什么相同点和不同点？

**提示：** 相同点 — 都可以使用 `open/read/write/close`；不同点 — 设备文件没有实际数据存储，读写由内核驱动完成。
```

```{admonition} 思考题 2
:class: note

在驱动函数 `vled_write` 中，为什么不能直接使用用户空间传递的 `buffer` 指针？`copy_from_user` 的作用是什么？如果不使用它直接访问会有什么后果？

**提示：** 内核空间和用户空间使用不同的地址空间（虚拟地址隔离），直接解引用用户指针会导致内核 panic 或安全漏洞。
```

```{admonition} 思考题 3
:class: note

如果两个程序同时打开设备文件并向虚拟 LED 写入数据，驱动如何响应？可能会产生什么问题？如何解决？

**提示：** 并发访问 → 竞态条件 → 互斥锁（参见加分项代码）。
```

```{admonition} 思考题 4
:class: note

简述 Linux 是如何通过 `file_operations` 结构体将文件操作与（虚拟）硬件控制联系起来的。

**提示：** VFS → 设备号 → file_operations 函数指针表 → 具体驱动函数。
```

```{admonition} 思考题 5
:class: note

本次实验中，你从哪些方面体会到了"一切皆文件"的设计思想？

**提示：** `/dev/vled_device` 抽象为文件 → `echo`/`cat` 控制"硬件" → `/proc/vled_status` 查看状态 → 操作方式与普通文件一致。
```

---

## 评分标准

| 项目 | 分值 | 要求 |
|------|:----:|------|
| 驱动代码编写 | 30 | 正确实现 `open`、`release` 函数，注释清晰，代码风格良好 |
| Makefile 与编译 | 10 | 能正确编译生成 `.ko` 文件，无错误 |
| 功能测试 | 30 | 能通过 `echo` 命令控制虚拟 LED，`/proc` 状态正确，提供终端截图 |
| 测试程序 | 10 | 能正确打开设备并读写数据，体现对文件操作的理解 |
| 思考题 | 15 | 回答问题准确，逻辑清晰，体现对文件系统抽象的深入理解 |
| 实验报告 | 5 | 格式规范，包含所有要求的内容 |
| **加分项**（互斥锁） | **+5** | 在驱动中添加互斥锁，防止并发写入冲突，并说明原理 |

---

## 提交要求

1. 提交完整的实验报告（PDF 格式），包含：
   - 实验目的、器材、原理简述
   - 驱动源代码（关键部分截图或粘贴，并加以解释）
   - Makefile 内容
   - 测试程序代码
   - 终端测试截图（显示 `echo` 命令和 `cat /proc/vled_status` 的结果）
   - 思考题的回答

2. 将源代码文件打包为 `vled_driver_code.zip` 一并提交：
   - `vled_driver.c`
   - `Makefile`
   - `test_vled.c`
   - `vled_driver_mutex.c`（加分项，可选）

---

## 课后练习

```{admonition} 练习 1：基础驱动验证
:class: tip

在远程服务器上完成驱动编译、加载、测试、卸载的全流程。将终端截图贴在实验报告中。
```

```{admonition} 练习 2：代码理解
:class: tip

修改 `vled_write` 函数，使写入 `'2'` 时实现 LED 状态翻转（toggle）——如果当前是亮则变为灭，灭则变为亮。编译并测试。
```

```{admonition} 练习 3：proc 扩展
:class: tip

在 `/proc/vled_status` 的 show 函数中，输出更友好的信息，例如 `LED Status: ON`（当 `vled_state=1`）或 `LED Status: OFF`（当 `vled_state=0`）。
```

```{admonition} 练习 4：并发测试（加分）
:class: tip

加载 `vled_driver_mutex.ko`，打开两个终端同时向设备文件写入不同值，观察行为。然后加载基础版 `vled_driver.ko`（无互斥锁），重复测试，对比差异并分析原因。
```
