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
        mail.select("INBOX")
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
    if request.method == 'POST':
        # 保存用户提交的配置
        config['imap_host'] = request.form.get('imap_host', '').strip()
        config['imap_port'] = request.form.get('imap_port', '').strip()
        config['imap_user'] = request.form.get('imap_user', '').strip()
        config['imap_pass'] = request.form.get('imap_pass', '').strip()
        config['smtp_host'] = request.form.get('smtp_host', '').strip()
        config['smtp_port'] = request.form.get('smtp_port', '').strip()
        config['smtp_user'] = request.form.get('smtp_user', '').strip()
        config['smtp_pass'] = request.form.get('smtp_pass', '').strip()
        config['forward_email'] = request.form.get('forward_email', '').strip()
        config['senders'] = request.form.get('senders', '').strip()
        config['deepseek_token'] = request.form.get('deepseek_token', '').strip()
        save_config(config)
        flash("配置已保存。")
        return redirect(url_for('config_page'))
    return render_template('config.html', config=config)


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


if __name__ == '__main__':
    # 开启定时任务
    start_scheduler()
    # 启动 Flask
    app.run(host='0.0.0.0', port=5000)