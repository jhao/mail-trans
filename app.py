import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from apscheduler.schedulers.background import BackgroundScheduler
import imaplib
import smtplib
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import email
import requests

"""
邮件汇总系统后台

该模块实现了以下功能：
1. 使用 IMAP 协议从 163 邮箱读取未读邮件，并根据配置的发件人筛选条件过滤。
2. 调用 DeepSeek API 将邮件正文翻译成中文并生成简要概览（当前为示例实现）。
3. 按照目录卡片的形式构建 HTML 内容并通过 SMTP 邮件发送到指定邮箱。
4. 通过 Flask 提供 Web 界面，允许用户查看最近运行记录、手动触发任务、修改配置并查看日志。
5. 使用 APScheduler 以后台调度的方式每小时执行一次邮件检查任务。

配置文件为 `config.json`，日志文件为 `logs.json`，均保存在当前工作目录下。
"""

app = Flask(__name__)
app.secret_key = 'change-me-secret-key'

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs.json')

UNSAFE_PORTS = {
    1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 77, 79,
    87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135,
    139, 143, 179, 389, 427, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540,
    548, 556, 563, 587, 601, 636, 993, 995, 2049, 3659, 4045, 6000, 6665, 6666,
    6667, 6668, 6669, 6697, 10080,
}


def load_config():
    """读取配置文件，不存在则返回空字典"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def save_config(config: dict) -> None:
    """保存配置到文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_logs():
    """读取日志文件，返回列表"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def save_logs(logs: list) -> None:
    """保存日志到文件"""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def deepseek_translate(text: str, token: str) -> str:
    """调用 DeepSeek API 将文本翻译成中文。

    这里给出示例实现，实际使用时请根据 DeepSeek 官方文档调整接口地址和参数。
    若调用失败则返回原文本。
    """
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    payload = {
        "model": "deepseek-translate",
        "prompt": f"翻译成中文: {text}",
        "temperature": 0.3,
    }
    try:
        resp = requests.post('https://api.deepseek.com/v1/text/completions', headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # 假设返回格式中包含 choices 字段
            return data.get('choices', [{}])[0].get('text', text)
    except Exception:
        pass
    return text


def deepseek_summarize(text: str, token: str) -> str:
    """调用 DeepSeek API 对中文文本进行摘要。

    示例实现，若调用失败则返回原文本的前 200 个字符。
    """
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    payload = {
        "model": "deepseek-summary",
        "prompt": f"请用简洁的方式概括下面的内容，用中文回答:{text}",
        "temperature": 0.3,
    }
    try:
        resp = requests.post('https://api.deepseek.com/v1/text/completions', headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('choices', [{}])[0].get('text', text[:200])
    except Exception:
        pass
    return text[:200]


def fetch_imap_mailboxes(host: str, port: int, username: str, password: str) -> list[str]:
    """验证 IMAP 连接并返回邮箱文件夹名称列表。"""
    mail = imaplib.IMAP4_SSL(host, int(port))
    try:
        mail.login(username, password)
        status, raw_mailboxes = mail.list()
        if status != 'OK':
            raise RuntimeError('获取邮箱列表失败')
        mailboxes = []
        for raw in raw_mailboxes or []:
            if not raw:
                continue
            decoded = raw.decode('utf-8', errors='ignore')
            # 典型格式：(\HasNoChildren) "/" "INBOX"
            if '"' in decoded:
                mailbox = decoded.split('"')[-2]
            else:
                parts = decoded.split(') ', 1)
                mailbox = parts[-1] if len(parts) > 1 else decoded
            mailbox = mailbox.strip()
            if mailbox:
                mailboxes.append(mailbox)
        return mailboxes
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def process_emails():
    """核心任务：读取邮件、翻译、汇总并转发"""
    config = load_config()
    logs = load_logs()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {"time": timestamp, "success": False, "message": ""}

    # 必要配置检查
    required_fields = [
        'imap_host', 'imap_port', 'imap_user', 'imap_pass',
        'smtp_host', 'smtp_port', 'smtp_user', 'smtp_pass',
        'forward_email', 'senders', 'deepseek_token'
    ]
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        log_entry['message'] = f"配置项缺失：{', '.join(missing_fields)}。请在配置页面完善。"
        logs.append(log_entry)
        save_logs(logs)
        return

    # 解析配置
    imap_host = config.get('imap_host')
    imap_port = int(config.get('imap_port', 993))
    imap_user = config.get('imap_user')
    imap_pass = config.get('imap_pass')
    smtp_host = config.get('smtp_host')
    smtp_port = int(config.get('smtp_port', 465))
    smtp_user = config.get('smtp_user')
    smtp_pass = config.get('smtp_pass')
    forward_email = config.get('forward_email')
    # 发件人过滤规则，允许多个发件人逗号分隔
    sender_list = [s.strip() for s in config.get('senders', '').split(',') if s.strip()]
    deepseek_token = config.get('deepseek_token')

    processed_emails = []
    try:
        # 连接 IMAP
        mail = imaplib.IMAP4_SSL(imap_host, imap_port)
        mail.login(imap_user, imap_pass)
        target_mailbox = config.get('imap_mailbox', 'INBOX') or 'INBOX'
        status, _ = mail.select(target_mailbox)
        if status != 'OK':
            raise Exception(f'IMAP 选择邮箱失败：{target_mailbox}')
        # 搜索未读邮件
        status, messages = mail.search(None, 'UNSEEN')
        if status != 'OK':
            raise Exception('IMAP 搜索失败')
        email_ids = messages[0].split()
        for num in email_ids:
            res, data = mail.fetch(num, '(RFC822)')
            if res != 'OK':
                continue
            msg = email.message_from_bytes(data[0][1])
            # 解析主题
            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                try:
                    subject = subject.decode(encoding or 'utf-8', errors='ignore')
                except Exception:
                    subject = subject.decode('utf-8', errors='ignore')
            else:
                subject = subject or ''
            from_ = msg.get("From", "")
            # 发件人筛选
            matched = False
            if sender_list:
                for s in sender_list:
                    if s.lower() in from_.lower():
                        matched = True
                        break
            else:
                matched = True
            if not matched:
                continue
            # 提取正文
            body = ""
            if msg.is_multipart():
                # 优先取纯文本部分
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and 'attachment' not in content_disposition:
                        try:
                            charset = part.get_content_charset() or 'utf-8'
                            body = part.get_payload(decode=True).decode(charset, errors='ignore')
                        except Exception:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                # 如果没有纯文本，再尝试读取 html
                if not body:
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if content_type == "text/html" and 'attachment' not in content_disposition:
                            try:
                                charset = part.get_content_charset() or 'utf-8'
                                body = part.get_payload(decode=True).decode(charset, errors='ignore')
                            except Exception:
                                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
            else:
                content_type = msg.get_content_type()
                if content_type in ["text/plain", "text/html"]:
                    try:
                        charset = msg.get_content_charset() or 'utf-8'
                        body = msg.get_payload(decode=True).decode(charset, errors='ignore')
                    except Exception:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            if not body:
                continue
            # 调用 DeepSeek 翻译
            chinese = deepseek_translate(body, deepseek_token)
            # 摘要
            summary = deepseek_summarize(chinese, deepseek_token)
            processed_emails.append({
                "subject": subject,
                "from": from_,
                "original": body,
                "chinese": chinese,
                "summary": summary
            })
        mail.logout()
    except Exception as e:
        # 处理过程报错
        log_entry['message'] = f"处理邮件出现异常: {str(e)}"
        logs.append(log_entry)
        save_logs(logs)
        return

    # 如果有处理到邮件则发送汇总邮件
    if processed_emails:
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = "自动汇总邮件"
            message["From"] = smtp_user
            message["To"] = forward_email
            # 构建 HTML
            html = "<html><body>"
            html += "<h2>邮件目录</h2>"
            html += "<ul>"
            for idx, itm in enumerate(processed_emails, start=1):
                anchor = f"email{idx}"
                summary = itm["summary"]
                # 截取概述前 100 个字符
                short_summary = summary[:100]
                html += f'<li><a href="#{anchor}">{itm["subject"]}</a> - {short_summary}</li>'
            html += "</ul><hr/>"
            for idx, itm in enumerate(processed_emails, start=1):
                anchor = f"email{idx}"
                html += f'<h3 id="{anchor}">{itm["subject"]}</h3>'
                html += f'<p><strong>发件人:</strong> {itm["from"]}</p>'
                # 将换行替换成 <br/>
                body_html = itm["chinese"].replace('\n', '<br/>')
                html += f'<p>{body_html}</p>'
                html += "<hr/>"
            html += "</body></html>"
            part = MIMEText(html, "html", "utf-8")
            message.attach(part)
            # 发送邮件
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [forward_email], message.as_string())
            server.quit()
            log_entry['success'] = True
            log_entry['message'] = f"成功处理 {len(processed_emails)} 封邮件并转发。"
        except Exception as e:
            log_entry['success'] = False
            log_entry['message'] = f"转发邮件失败: {str(e)}"
    else:
        # 没有符合条件的邮件
        log_entry['success'] = True
        log_entry['message'] = "没有符合条件的未读邮件。"

    logs.append(log_entry)
    save_logs(logs)


@app.route('/')
def index():
    """首页：显示最近一次运行结果和最近日志"""
    logs = load_logs()
    last_run = logs[-1] if logs else None
    # 仅展示最近 10 条
    recent_logs = logs[-10:]
    return render_template('index.html', last_run=last_run, logs=recent_logs)


@app.route('/run', methods=['POST'])
def run_now():
    """手动触发任务"""
    process_emails()
    flash("任务已执行。")
    return redirect(url_for('index'))


@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """配置页面：显示或保存配置"""
    config = load_config()
    mailboxes = config.get('available_mailboxes', [])
    if request.method == 'POST':
        action = request.form.get('action', 'save')
        # 保存用户提交的配置
        config['imap_host'] = request.form.get('imap_host', '').strip()
        config['imap_port'] = request.form.get('imap_port', '').strip()
        config['imap_user'] = request.form.get('imap_user', '').strip()
        config['imap_pass'] = request.form.get('imap_pass', '').strip()
        config['imap_mailbox'] = request.form.get('imap_mailbox', '').strip() or 'INBOX'
        config['smtp_host'] = request.form.get('smtp_host', '').strip()
        config['smtp_port'] = request.form.get('smtp_port', '').strip()
        config['smtp_user'] = request.form.get('smtp_user', '').strip()
        config['smtp_pass'] = request.form.get('smtp_pass', '').strip()
        config['forward_email'] = request.form.get('forward_email', '').strip()
        config['senders'] = request.form.get('senders', '').strip()
        config['deepseek_token'] = request.form.get('deepseek_token', '').strip()

        if action == 'fetch_mailboxes':
            try:
                mailboxes = fetch_imap_mailboxes(
                    config['imap_host'],
                    int(config['imap_port'] or 993),
                    config['imap_user'],
                    config['imap_pass'],
                )
                config['available_mailboxes'] = mailboxes
                if mailboxes and config['imap_mailbox'] not in mailboxes:
                    config['imap_mailbox'] = mailboxes[0]
                save_config(config)
                flash("邮箱验证成功，已获取可用列表。")
            except Exception as exc:
                flash(f"获取邮箱列表失败：{exc}")
            return render_template('config.html', config=config, mailboxes=mailboxes)

        config['available_mailboxes'] = mailboxes
        save_config(config)
        flash("配置已保存。")
        return redirect(url_for('config_page'))
    return render_template('config.html', config=config, mailboxes=mailboxes)


@app.route('/logs')
def logs_page():
    """日志页面：查看全部日志"""
    logs = load_logs()
    return render_template('logs.html', logs=logs)


def start_scheduler():
    """启动后台调度器，每小时运行一次"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_emails, 'interval', hours=1)
    scheduler.start()


def get_runtime_port(default: int = 6006) -> int:
    """返回浏览器可访问的安全端口。"""
    env_port = os.environ.get('APP_PORT') or os.environ.get('PORT')
    port = default
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            logging.warning("无效的端口号 %s，使用默认端口 %s。", env_port, default)
            port = default
    if port in UNSAFE_PORTS:
        logging.warning("端口 %s 在浏览器中被标记为不安全，自动切换到 %s。", port, default)
        port = default
    return port


if __name__ == '__main__':
    # 开启定时任务
    start_scheduler()
    # 启动 Flask
    host = os.environ.get('APP_HOST', '0.0.0.0')
    port = get_runtime_port()
    app.run(host=host, port=port)
