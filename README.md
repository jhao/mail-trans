# 邮件汇总系统

一个基于 Python Flask 的邮件汇总与转发服务，用于定时从企业邮箱（支持 IMAP 与 POP3 协议）拉取未读邮件，调用 DeepSeek API 完成翻译与摘要，并通过 HTML 邮件的形式将目录化内容转发给指定收件人。系统自带 Web 管理界面，可视化查看运行日志、手动触发任务、维护收发协议/SMTP/DeepSeek 等核心配置，并内置身份认证与异步队列机制，保障运行安全与稳定性。

## 目录
- [功能概览](#功能概览)
- [项目结构](#项目结构)
- [技术结构与工作原理](#技术结构与工作原理)
- [环境准备](#环境准备)
- [启动与访问](#启动与访问)
- [配置与数据文件](#配置与数据文件)
- [任务队列与调度说明](#任务队列与调度说明)
- [身份认证](#身份认证)
- [核心技术点说明](#核心技术点说明)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 功能概览
- **未读邮件采集**：周期性连接 IMAP 或 POP3 邮箱，筛选符合发件人规则的未读邮件，可自定义目标文件夹并支持自动获取邮箱列表。
- **自动翻译与摘要**：调用 DeepSeek API 将正文翻译为中文并生成概览，可配置请求超时时间。
- **HTML 汇总邮件**：构建带目录锚点的 HTML 内容，并使用 SMTP 发送到目标邮箱。
- **异步处理队列**：拉取的邮件批次先入队列，再由后台轮询任务异步发送，失败自动重试并记录。
- **Web 管理台**：提供首页、配置、日志三大界面，展示下一次执行时间、队列状态和近期日志，支持手动执行任务与配置管理。
- **登录认证**：通过内置账号 `email_owner` 登录后方可访问管理台，可自定义密码。
- **日志归档**：所有执行信息写入本地 JSON 日志，方便排查问题。

## 项目结构
```
mail-trans/
├─ app.py              # 核心应用，包含 Flask 路由、任务调度、队列与邮件处理逻辑
├─ requirements.txt    # 依赖列表
├─ static/
│  └─ style.css        # 前端样式
├─ templates/          # Jinja2 模板（首页/配置/日志/登录）
├─ config.json         # 运行时生成的配置（首次启动可为空）
├─ logs.json           # 运行日志（运行后生成）
└─ queue.db            # 任务队列 SQLite 数据库（首次运行自动创建）
```

## 技术结构与工作原理
系统由以下子模块组成：

1. **Flask Web 层**（`index` / `config_page` / `logs_page` / `login` 等路由）：
   - 提供配置维护、日志查看、邮箱验证、密码修改与手动触发能力。
   - 使用 `templates/` 下的 Jinja2 模板渲染页面，`static/style.css` 提供统一样式。

2. **任务调度层**（`start_scheduler`、`refresh_scheduler_job`、`ensure_queue_worker_job`）：
   - 基于 APScheduler 的 `BackgroundScheduler`，应用启动后常驻后台。
   - 支持分钟间隔、按日固定时间、按整点列表三种调度模式，可在配置页切换。

3. **异步任务队列**（`enqueue_email_batch`、`process_task_queue`、`run_queue_worker`）：
   - 拉取到的待处理邮件统一写入 `queue.db`，交由队列轮询任务异步发送。
   - 队列任务失败会记录错误并在限次内自动重试。

4. **邮件处理流水线**（`process_emails`）：
   - 读取配置后校验必填项。
   - 按协议连接 IMAP/POP3 获取未读邮件，解析主题、发件人和正文。POP3 模式使用 UIDL/LIST 去重，IMAP 模式支持自定义 ID 指令参数与 UTF-7 文件夹编码。
   - 通过 DeepSeek API 生成翻译与摘要，再入队等待异步发送。

5. **外部服务集成**：
   - IMAP/POP3/SMTP：用于收取与发送邮件。
   - DeepSeek API：提供翻译与摘要服务，可按需求替换为其它接口。

### 数据流示意
1. 调度器触发 `process_emails` → 按配置连接 IMAP/POP3 → 拉取未读邮件。
2. 对符合筛选条件的邮件执行翻译/摘要 → 将批次入队列。
3. 队列轮询任务异步构建 HTML → 通过 SMTP 转发汇总邮件。
4. 成功/失败结果写入 `logs.json`，队列状态写入 `queue.db`。
5. Web 层展示最近日志、队列统计与下一次运行时间；配置更新写回 `config.json`。

## 环境准备
- Python >= 3.9（建议 3.11）。
- 可访问互联网（需要访问邮箱服务器和 DeepSeek API）。
- 拥有目标邮箱的 IMAP 或 POP3 权限，以及 SMTP 授权码。

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

## 启动与访问
```bash
python app.py --pwd <初始密码>
```

- 应用默认监听 `http://0.0.0.0:6006/`。若需修改端口，可通过环境变量 `APP_PORT`（或 `PORT`）指定；当端口位于浏览器的非安全列表时会自动回退到 6006。
- 首次启动可通过命令行参数 `--pwd` 或环境变量 `APP_DEFAULT_PASSWORD`（未设置时回退读取 `DEFAULT_PASSWORD`）初始化 `email_owner` 账号的登录密码，后续可在配置页修改；若未提供且配置中没有密码，系统会拒绝所有登录请求。
- 首次访问浏览器会跳转至登录页，使用 `email_owner` + 设置的密码登录。
- 登录后：
  1. 进入“配置”页面，填写 IMAP/POP3/SMTP/DeepSeek 参数，必要时点击“验证并获取邮箱列表”以拉取可选文件夹。
  2. 返回首页可手动点击“立即执行任务”验证流程。
  3. “日志”页面可查看历史执行信息，首页亦展示队列统计与下一次运行时间。

> **提示**：开发模式下 Flask 会直接在主线程启动 APScheduler 和队列轮询任务，无需额外进程。

### 生产部署建议
Python 应用无需编译，但建议使用 WSGI 服务器托管，并与调度器/队列在同一进程运行。

#### 方式一：Gunicorn + Systemd（Linux）
1. 安装 Gunicorn：
   ```bash
   pip install gunicorn
   ```
2. 创建启动命令（保持 APScheduler 在 `app.py` 内启动）：
   ```bash
   gunicorn --bind 0.0.0.0:6006 --workers 2 --threads 4 app:app
   ```
3. 将上述命令写入 systemd service，确保崩溃后自动重启。

#### 方式二：Docker
1. 使用内置 `Dockerfile`（节选如下），镜像会通过清华大学 PyPI 镜像加速安装依赖，并使用 Gunicorn 作为生产服务：
   ```dockerfile
   FROM python:3.11-slim

   ENV PYTHONUNBUFFERED=1 \
       PIP_NO_CACHE_DIR=1

   WORKDIR /app

   COPY requirements.txt ./
   RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
       && python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

   COPY . ./

   ENV APP_PORT=6006

   CMD ["gunicorn", "--bind", "0.0.0.0:${APP_PORT}", "--workers", "2", "--threads", "4", "app:app"]
   ```
2. 构建并运行：
   ```bash
   docker build -t mail-trans .
   docker run -d --name mail-trans -p 6006:6006 \
     -e APP_DEFAULT_PASSWORD=<初始密码> \
     -v $(pwd)/config.json:/app/config.json \
     -v $(pwd)/logs.json:/app/logs.json \
     -v $(pwd)/queue.db:/app/queue.db \
     mail-trans
   ```
   > 若使用 Docker，请确保挂载配置、日志与队列数据库文件，避免容器重建导致数据丢失。

## 配置与数据文件
- `config.json`：由配置界面写入，包含 IMAP/POP3/SMTP 凭证、发件人过滤、DeepSeek token、转发邮箱等信息；POP3 模式下会记录已处理的 UIDL 以避免重复；还保存调度设置、DeepSeek 超时时间与可用邮箱列表。
- `logs.json`：追加存储每次运行的时间、成功状态与描述。
- `queue.db`：SQLite 数据库，保存待发送、处理中、已完成或失败的邮件批次。
- 建议为上述文件设置合理的文件权限，避免敏感信息泄漏。

## 任务队列与调度说明
- 队列轮询任务每 10 秒检查一次新任务，默认每批最多处理一组邮件，失败将记录错误并在 3 次内自动重试。
- 配置页提供三种调度模式：
  - **按分钟间隔**：指定执行间隔分钟数，默认 60 分钟。
  - **按天定时**：指定间隔天数及每日执行时间（HH:MM）。
  - **固定整点**：输入多个小时（以逗号分隔），系统会在对应整点执行。
- 首页展示下一次执行时间及队列统计，方便监控任务运行状态。

## 身份认证
- 系统内置账号：`email_owner`。
- 首次启动可通过 `python app.py --pwd <密码>` 或设置环境变量 `APP_DEFAULT_PASSWORD` 初始化密码；也可在配置页的“更新登录密码”区域修改。
- 密码以加盐 MD5 形式存储于 `config.json` 的 `auth.email_owner.password_hash` 字段。
- 所有受保护页面均需登录，退出可点击右上角“退出登录”。

## 核心技术点说明
1. **Flask + Jinja2**：构建轻量级管理界面，满足配置、日志、密码修改及邮箱验证需求。
2. **APScheduler `BackgroundScheduler`**：在 Web 应用内部维护稳定的定时任务与队列轮询，确保按计划执行。
3. **IMAP/POP3/SMTP 协议栈**：通过 `imaplib`、`poplib` 和 `smtplib` 与邮箱服务器交互，支持多发件人过滤、UTF-7 文件夹处理及 POP3 UIDL 去重。
4. **DeepSeek API 对接**：`requests` 调用外部翻译/摘要服务，失败时自动降级使用原文/截取内容，增强健壮性。
5. **SQLite 队列存储 + JSON 日志**：采用 SQLite 保障队列可靠性，JSON 便于快速查看与备份。
6. **安全端口选择**：在浏览器限制的环境下自动回退到允许的端口，提升体验。

## 常见问题
- **DeepSeek 调用失败**：检查网络连通性与 token 是否有效；代码中已做降级处理，仍建议在日志中排查异常信息。
- **未收到汇总邮件**：确认 SMTP 凭证与端口设置正确，必要时开启邮箱的“允许第三方应用”或“授权码”功能；查看队列状态是否存在失败任务。
- **定时任务未执行**：确保进程未被守护程序杀死，可查看首页的“下一次执行时间”与 `logs.json`；生产环境建议结合 systemd 监控。
- **无法登录**：确认启动时是否传入 `--pwd`、配置了 `APP_DEFAULT_PASSWORD`（或 `DEFAULT_PASSWORD`）或在配置中已设置密码；若忘记密码，可删除 `config.json` 中的 `auth` 字段并重新启动传入新密码。

## 许可证
此项目仅供学习和内部使用，未经授权请勿用于商业用途。
