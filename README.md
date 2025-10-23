# 邮件汇总系统

一个基于 Python Flask 的邮件汇总与转发服务，用于定时从企业邮箱拉取未读邮件，调用 DeepSeek API 完成翻译与摘要，并通过 HTML 邮件的形式将目录化内容转发给指定收件人。系统自带 Web 管理界面，可视化查看运行日志、手动触发任务以及维护 IMAP/SMTP/DeepSeek 等核心配置。

## 目录
- [功能概览](#功能概览)
- [项目结构](#项目结构)
- [技术结构与工作原理](#技术结构与工作原理)
- [环境准备](#环境准备)
- [开发环境启动](#开发环境启动)
- [生产部署建议](#生产部署建议)
- [配置与数据文件](#配置与数据文件)
- [核心技术点说明](#核心技术点说明)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 功能概览
- **未读邮件采集**：周期性连接 IMAP 邮箱，筛选符合发件人规则的未读邮件。
- **自动翻译与摘要**：调用 DeepSeek API 将正文翻译为中文并生成概览。
- **HTML 汇总邮件**：构建带目录锚点的 HTML 内容，并使用 SMTP 发送到目标邮箱。
- **Web 管理台**：通过 Flask 模板提供首页、配置页、日志页三大界面，支持手动执行任务和查看运行记录。
- **日志归档**：所有执行信息写入本地 `logs.json`，方便排查问题。

## 项目结构
```
mail-trans/
├─ app.py              # 核心应用，包含 Flask 路由、任务调度和邮件处理逻辑
├─ requirements.txt    # 依赖列表
├─ static/
│  └─ style.css        # 前端样式
├─ templates/          # Jinja2 模板（首页/配置/日志）
├─ config.json         # 运行时生成的配置（首次启动可为空）
└─ logs.json           # 运行日志（运行后生成）
```

## 技术结构与工作原理
系统由四个主要子模块组成：

1. **Flask Web 层**（`app.py` 中的路由 `index`, `config_page`, `logs_page`）：
   - 提供配置维护、日志查看和手动触发能力。
   - 使用 `templates/` 下的 Jinja2 模板渲染页面，`static/style.css` 提供统一样式。

2. **任务调度层**（`start_scheduler` 函数）：
   - 基于 APScheduler 的 `BackgroundScheduler`，在应用启动时加载。
   - 使用 `interval` 触发器每小时执行一次 `process_emails`。

3. **邮件处理流水线**（`process_emails` 函数）：
   - 读取配置后校验必填项。
   - 连接 IMAP（使用 `imaplib`）获取未读邮件，解析主题、发件人和正文。
   - 通过 `deepseek_translate`、`deepseek_summarize` 调用 DeepSeek API 处理文本。
   - 将结果汇总成带目录锚点的 HTML 内容，使用 `smtplib` 通过 SMTP 发送。
   - 记录执行结果到 `logs.json`，供前端展示。

4. **外部服务集成**：
   - IMAP/SMTP：用于收取与发送邮件。
   - DeepSeek API：提供翻译与摘要服务，可按需求替换为自研/其它第三方接口。

### 数据流示意
1. 调度器触发 `process_emails` → 连接 IMAP → 拉取未读邮件。
2. 对符合筛选条件的邮件执行翻译/摘要 → 生成目录化 HTML。
3. 通过 SMTP 将汇总邮件转发 → 写入执行日志。
4. Web 层从 `logs.json` 读取最近记录，用于页面展示；配置更新写回 `config.json`。

## 环境准备
- Python >= 3.9（建议 3.11）。
- 可访问互联网（需要访问邮箱服务器和 DeepSeek API）。
- 拥有目标邮箱的 IMAP/SMTP 权限及授权码。

### 1. 克隆项目
```bash
git clone <repo-url>
cd mail-trans
```

### 2. （可选）创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 开发环境启动
```bash
python app.py
```
默认监听 `http://0.0.0.0:5000/`，浏览器访问后即可使用：
1. 首次进入首页点击导航中的“配置”，填写 IMAP/SMTP/DeepSeek 参数并保存。
2. 返回首页可手动点击“立即执行任务”验证流程。
3. “日志”页面可查看历史执行信息。

> **提示**：开发模式下 Flask 会直接在主线程启动 APScheduler 后台任务，因此无需单独运行调度器。

## 生产部署建议
Python 应用无需编译，但建议使用 WSGI 服务器托管，并将调度器与 Web 服务作为同一进程启动。

### 方式一：Gunicorn + Systemd（Linux）
1. 安装 Gunicorn：
   ```bash
   pip install gunicorn
   ```
2. 创建启动命令（保持 APScheduler 在 `app.py` 内启动）：
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 app:app
   ```
3. 可将上述命令写入 systemd service，确保崩溃后自动重启。

### 方式二：Docker（示例）
1. 创建 `Dockerfile`（需自建）：
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . /app
   RUN pip install --no-cache-dir -r requirements.txt
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "app:app"]
   ```
2. 构建并运行：
   ```bash
   docker build -t mail-trans .
   docker run -d --name mail-trans -p 5000:5000 \
     -v $(pwd)/config.json:/app/config.json \
     -v $(pwd)/logs.json:/app/logs.json mail-trans
   ```
   > 若使用 Docker，请确保挂载配置和日志文件，避免容器重建导致数据丢失。

## 配置与数据文件
- `config.json`：由配置界面写入，包含 IMAP/SMTP 凭证、发件人过滤、DeepSeek token、转发邮箱等信息。
- `logs.json`：追加存储每次运行的时间、成功状态与描述。
- 两个文件均位于项目根目录，建议设置合理的文件权限，避免敏感信息泄漏。

## 核心技术点说明
1. **Flask + Jinja2**：快速构建轻量级管理界面，满足配置与日志查看需求。路由处理详见 `index`、`config_page`、`logs_page`。  
2. **APScheduler `BackgroundScheduler`**：在 Web 应用内部维护稳定的定时任务，确保每小时触发邮件处理逻辑。  
3. **IMAP/SMTP 协议栈**：通过 `imaplib` 和 `smtplib` 与邮箱服务器交互，支持多发件人过滤、HTML 邮件发送。  
4. **DeepSeek API 对接**：`requests` 调用外部翻译/摘要服务，失败时自动降级使用原文/截取内容，增强健壮性。  
5. **JSON 配置与日志**：配置与日志均采用 JSON 文件存储，简单直观，便于自动化备份与查看。

## 常见问题
- **DeepSeek 调用失败**：检查网络连通性与 token 是否有效；代码中已做降级处理，仍建议在日志中排查异常信息。
- **未收到汇总邮件**：确认 SMTP 凭证与端口设置正确，必要时开启邮箱的“允许第三方应用”或“授权码”功能。
- **定时任务未执行**：确保进程未被守护程序杀死，可查看 `logs.json` 中是否有定时记录；生产环境建议结合 systemd 监控。

## 许可证
此项目仅供学习和内部使用，未经授权请勿用于商业用途。
