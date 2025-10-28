import os
import json
import logging
import re
import base64
import argparse
import hashlib
import sqlite3
import threading
from functools import wraps
from datetime import datetime, timedelta, time
from urllib.parse import urlparse
from flask import Flask, render_template, request, redirect, url_for, flash, session
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import imaplib
import poplib
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
2. 调用 DeepSeek API 将邮件正文翻译成中文并生成简要概览。
3. 按照目录卡片的形式构建 HTML 内容并通过 SMTP 邮件发送到指定邮箱。
4. 通过 Flask 提供 Web 界面，允许用户查看最近运行记录、手动触发任务、修改配置并查看日志。
5. 使用 APScheduler 以后台调度的方式每小时执行一次邮件检查任务。

配置文件为 `config.json`，日志文件为 `logs.json`，均保存在当前工作目录下。
"""

app = Flask(__name__)
app.secret_key = 'change-me-secret-key'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs.json')
QUEUE_DB_FILE = os.path.join(os.path.dirname(__file__), 'queue.db')

SCHEDULER_JOB_ID = "process_emails_job"
QUEUE_WORKER_JOB_ID = "queue_worker_job"
QUEUE_BATCH_LIMIT = 1
QUEUE_MAX_ATTEMPTS = 3
QUEUE_CONFIG_WARNING_INTERVAL = timedelta(minutes=5)

CONFIG_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()

SCHEDULER: BackgroundScheduler | None = None
LAST_QUEUE_CONFIG_WARNING: datetime | None = None

DEFAULT_IMAP_ID = {
    'name': 'Apple Mail',
    'version': '16.0',
    'vendor': 'Apple Inc.',
    'support-url': 'https://support.apple.com/mail',
}

UNSAFE_PORTS = {
    1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 77, 79,
    87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135,
    139, 143, 179, 389, 427, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540,
    548, 556, 563, 587, 601, 636, 993, 995, 2049, 3659, 4045, 6000, 6665, 6666,
    6667, 6668, 6669, 6697, 10080,
}

MAIL_COMMON_REQUIRED = [
    'smtp_host',
    'smtp_port',
    'smtp_user',
    'smtp_pass',
    'forward_email',
    'senders',
    'deepseek_token',
]

MAIL_IMAP_REQUIRED = ['imap_host', 'imap_port', 'imap_user', 'imap_pass']
MAIL_POP3_REQUIRED = ['pop3_host', 'pop3_port', 'pop3_user', 'pop3_pass']

QUEUE_REQUIRED_FIELDS = [
    'smtp_host',
    'smtp_port',
    'smtp_user',
    'smtp_pass',
    'forward_email',
    'deepseek_token',
]

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_DEEPSEEK_TIMEOUT = 120


PASSWORD_SECRET = "f8a3c1b4de5"
AUTH_USER = "email_owner"
SESSION_KEY = "auth_user"
AUTH_CONFIG_KEY = "auth"
AUTH_PASSWORD_FIELD = "password_hash"

AUTH_PASSWORD_HASH: str | None = None


def hash_password(password: str) -> str:
    """Generate MD5 hash for the given password with secret salt."""

    payload = f"{password}{PASSWORD_SECRET}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str | None) -> bool:
    """Verify plain password against stored MD5 hash."""

    if not stored_hash:
        return False
    return hash_password(password) == stored_hash


def _extract_password_hash(config: dict | None = None) -> str | None:
    """Fetch stored password hash for email_owner from config."""

    if config is None:
        config = load_config()
    auth_section = config.get(AUTH_CONFIG_KEY)
    if not isinstance(auth_section, dict):
        return None
    entry = auth_section.get(AUTH_USER)
    if isinstance(entry, dict):
        password_hash = entry.get(AUTH_PASSWORD_FIELD)
    elif isinstance(entry, str):
        password_hash = entry
    else:
        password_hash = None
    if isinstance(password_hash, str) and password_hash.strip():
        return password_hash.strip()
    return None


def set_auth_password_hash(password_hash: str, *, config: dict | None = None) -> None:
    """Persist hashed password for email_owner."""

    global AUTH_PASSWORD_HASH

    if config is None:
        config = load_config()
    auth_section = config.get(AUTH_CONFIG_KEY)
    if not isinstance(auth_section, dict):
        auth_section = {}
    auth_section[AUTH_USER] = {AUTH_PASSWORD_FIELD: password_hash}
    config[AUTH_CONFIG_KEY] = auth_section
    save_config(config)
    AUTH_PASSWORD_HASH = password_hash


def set_auth_password(password: str, *, config: dict | None = None) -> None:
    """Set plain password for email_owner by hashing with secret."""

    password_hash = hash_password(password)
    set_auth_password_hash(password_hash, config=config)


def initialize_auth(default_password: str | None = None) -> None:
    """Ensure authentication state is initialized with stored or default password."""

    global AUTH_PASSWORD_HASH

    if not default_password:
        env_password = os.environ.get('APP_DEFAULT_PASSWORD') or os.environ.get('DEFAULT_PASSWORD')
        if env_password:
            logging.info("未检测到启动参数中的密码，使用环境变量初始化登录密码。")
            default_password = env_password

    config = load_config()
    stored_hash = _extract_password_hash(config)
    if stored_hash:
        AUTH_PASSWORD_HASH = stored_hash
        return
    if default_password:
        logging.info("未检测到已设置的登录密码，使用启动参数中的默认密码初始化。")
        set_auth_password(default_password, config=config)
    else:
        logging.warning("未提供默认登录密码 (--pwd) 且配置中不存在密码，系统将拒绝所有登录。")
        AUTH_PASSWORD_HASH = None


def is_authenticated() -> bool:
    """Check whether current session is authenticated as email_owner."""

    return session.get(SESSION_KEY) == AUTH_USER


def login_required(view):
    """Decorator to enforce authentication for protected routes."""

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if is_authenticated():
            return view(*args, **kwargs)
        next_url = request.path
        if request.query_string:
            next_url = f"{request.path}?{request.query_string.decode()}"
        flash("请先登录。")
        return redirect(url_for('login', next=next_url))

    return wrapped_view


def _safe_next_url(candidate: str | None) -> str:
    """Ensure next URL is safe and relative."""

    if not candidate:
        return url_for('index')
    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        return url_for('index')
    return candidate


@app.context_processor
def inject_auth_user() -> dict[str, str | None]:
    """Expose current authenticated user to templates."""

    return {"current_user": session.get(SESSION_KEY)}


def _trim_text_for_prompt(text: str, limit: int = 6000) -> str:
    """Prepare text for prompt by trimming whitespace and clipping length."""

    cleaned = (text or "").strip()
    if len(cleaned) > limit:
        logging.debug("邮件正文长度 %d 超出 %d，截断后再发送 DeepSeek。", len(cleaned), limit)
        return cleaned[:limit]
    return cleaned


def _build_imap_id_params(config: dict | None = None) -> dict[str, str]:
    """根据配置构建 IMAP ID 指令参数。"""

    config = config or {}
    
    def _resolve_value(config_key: str, default: str) -> str:
        value = config.get(config_key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        return default

    params: dict[str, str] = {
        'name': _resolve_value('imap_id_name', DEFAULT_IMAP_ID['name']),
        'version': _resolve_value('imap_id_version', DEFAULT_IMAP_ID['version']),
        'vendor': _resolve_value('imap_id_vendor', DEFAULT_IMAP_ID['vendor']),
        'support-url': _resolve_value('imap_id_support_url', DEFAULT_IMAP_ID['support-url']),
    }

    support_email = config.get('imap_id_support_email') or config.get('forward_email')
    if isinstance(support_email, str) and support_email.strip():
        params['support-email'] = support_email.strip()

    return {k: v for k, v in params.items() if isinstance(v, str) and v.strip()}


def _send_imap_id(mail: imaplib.IMAP4, id_params: dict[str, str] | None = None) -> None:
    """向 IMAP 服务器发送 ID 指令以降低 Unsafe Login 触发概率。"""
    if not id_params:
        id_params = _build_imap_id_params()

    pieces: list[str] = []
    for key, value in id_params.items():
        key = str(key).strip()
        value = str(value).strip()
        if not key or not value:
            continue
        safe_key = key.replace('"', '\\"')
        safe_value = value.replace('"', '\\"')
        pieces.append(f'"{safe_key}" "{safe_value}"')

    if not pieces:
        logging.debug("未生成有效的 IMAP ID 参数，跳过发送。")
        return

    payload = f'({" ".join(pieces)})'
    try:
        mail._simple_command('ID', payload)
        mail._untagged_response('ID')
        logging.debug("已向 IMAP 服务器发送 ID 指令：%s", payload)
    except Exception as exc:
        logging.debug("发送 IMAP ID 指令失败：%s", exc)


def _imap_encode_utf7(mailbox: str) -> str:
    """Encode mailbox names using IMAP modified UTF-7 with graceful fallback."""
    try:
        encoder = getattr(imaplib.IMAP4, '_encode_utf7')
    except AttributeError:
        encoder = None
    if callable(encoder):
        try:
            return encoder(mailbox)
        except Exception:
            logging.debug("内置 UTF-7 编码失败，使用自定义实现。", exc_info=True)
    # 自定义实现基于 RFC 2060 Modified UTF-7 规则
    pieces: list[str] = []
    buffer: list[str] = []

    def _flush_buffer() -> None:
        if not buffer:
            return
        chunk = ''.join(buffer)
        encoded = base64.b64encode(chunk.encode('utf-16-be')).decode('ascii').rstrip('=')
        pieces.append(f'&{encoded}-')
        buffer.clear()

    for char in mailbox:
        code_point = ord(char)
        if 0x20 <= code_point <= 0x7e and char != '&':
            _flush_buffer()
            pieces.append(char)
        elif char == '&':
            _flush_buffer()
            pieces.append('&-')
        else:
            buffer.append(char)

    _flush_buffer()
    return ''.join(pieces)


def _imap_decode_utf7(mailbox: str) -> str:
    """Decode mailbox names encoded with IMAP modified UTF-7."""
    try:
        decoder = getattr(imaplib.IMAP4, '_decode_utf7')
    except AttributeError:
        decoder = None
    if callable(decoder):
        try:
            return decoder(mailbox)
        except Exception:
            logging.debug("内置 UTF-7 解码失败，使用自定义实现。", exc_info=True)

    result: list[str] = []
    i = 0
    length = len(mailbox)
    while i < length:
        char = mailbox[i]
        if char == '&':
            j = i + 1
            while j < length and mailbox[j] != '-':
                j += 1
            if j == i + 1:
                result.append('&')
            else:
                chunk = mailbox[i + 1:j]
                padding = (-len(chunk)) % 4
                chunk += '=' * padding
                try:
                    decoded = base64.b64decode(chunk)
                    result.append(decoded.decode('utf-16-be'))
                except Exception:
                    logging.debug("自定义 UTF-7 解码失败，保留原始片段: %s", chunk, exc_info=True)
                    result.append('&' + mailbox[i + 1:j] + '-')
            i = j
        else:
            result.append(char)
        i += 1
    return ''.join(result)


def _decode_mailbox_name(raw: bytes) -> str | None:
    """从 IMAP LIST 返回的原始行中提取并解码邮箱名称。"""
    if not raw:
        logging.debug("收到空的 IMAP 邮箱原始数据，返回 None。")
        return None
    mailbox_bytes = None
    match = re.search(br' "/" (.+)$', raw)
    if match:
        logging.debug("通过正则匹配提取邮箱名称: %s", match.group(1))
        mailbox_bytes = match.group(1).strip()
    if mailbox_bytes is None:
        parts = raw.split()
        mailbox_bytes = parts[-1].strip() if parts else b''
        logging.debug("通过分割方式提取邮箱名称: %s", mailbox_bytes)
    if mailbox_bytes.startswith(b'"') and mailbox_bytes.endswith(b'"'):
        logging.debug("去除邮箱名称外层引号。")
        mailbox_bytes = mailbox_bytes[1:-1]
    if not mailbox_bytes:
        logging.debug("邮箱名称字节串为空，返回 None。")
        return None
    try:
        decoded = mailbox_bytes.decode('imap4-utf-7')
        logging.debug("使用 imap4-utf-7 解码邮箱名称成功: %s", decoded)
        return decoded
    except Exception:
        try:
            ascii_source = mailbox_bytes.decode('ascii', errors='ignore')
            logging.debug("imap4-utf-7 解码失败，尝试通过 ASCII -> _decode_utf7: %s", ascii_source)
            decoded_mailbox = _imap_decode_utf7(ascii_source)
            logging.debug("使用 _decode_utf7 解码成功: %s", decoded_mailbox)
            return decoded_mailbox
        except Exception:
            fallback = mailbox_bytes.decode('utf-8', errors='ignore') or None
            logging.debug("所有专用解码失败，退回 UTF-8 忽略错误: %s", fallback)
            return fallback


def _list_imap_mailboxes(mail: imaplib.IMAP4) -> list[str]:
    """列出并解码 IMAP 邮箱名称。"""
    status, raw_mailboxes = mail.list()
    if status != 'OK':
        raise RuntimeError('获取邮箱列表失败')
    mailboxes: list[str] = []
    for raw in raw_mailboxes or []:
        if not raw:
            continue
        logging.info("IMAP 邮箱列表原始响应: %r", raw)
        mailbox = _decode_mailbox_name(raw)
        if not mailbox:
            continue
        logging.info("IMAP 邮箱名称解析结果: %s", mailbox)
        mailboxes.append(mailbox)
    return mailboxes


def _find_inbox_name(mailboxes: list[str]) -> str | None:
    """从邮箱列表中寻找标准收件箱名称。"""
    for mailbox in mailboxes:
        if mailbox.upper() == 'INBOX' or mailbox == '收件箱':
            logging.debug("匹配到标准收件箱名称: %s", mailbox)
            return mailbox
    logging.debug("邮箱列表中未找到标准收件箱名称。")
    return None


def _decode_header_value(value: str | None) -> str:
    """解析邮件头字段，兼容多段编码。"""

    if not value:
        return ""

    decoded_chunks: list[str] = []
    for chunk, encoding in decode_header(value):
        if isinstance(chunk, bytes):
            try:
                decoded_chunks.append(chunk.decode(encoding or 'utf-8', errors='ignore'))
            except Exception:
                decoded_chunks.append(chunk.decode('utf-8', errors='ignore'))
        elif isinstance(chunk, str):
            decoded_chunks.append(chunk)
    return ''.join(decoded_chunks)


def _extract_email_body(msg: email.message.Message) -> str:
    """从邮件对象中提取正文内容，优先纯文本。"""

    body = ""
    if msg.is_multipart():
        logging.debug("邮件 %s 为多部分结构，开始解析。", _decode_header_value(msg.get('Subject')))
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and 'attachment' not in content_disposition:
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    payload = part.get_payload(decode=True)
                    if payload is not None:
                        body = payload.decode(charset, errors='ignore')
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload is not None:
                        body = payload.decode('utf-8', errors='ignore')
                if body:
                    break
        if not body:
            logging.debug("邮件 %s 未找到纯文本部分，尝试解析 HTML。", _decode_header_value(msg.get('Subject')))
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/html" and 'attachment' not in content_disposition:
                    try:
                        charset = part.get_content_charset() or 'utf-8'
                        payload = part.get_payload(decode=True)
                        if payload is not None:
                            body = payload.decode(charset, errors='ignore')
                    except Exception:
                        payload = part.get_payload(decode=True)
                        if payload is not None:
                            body = payload.decode('utf-8', errors='ignore')
                    if body:
                        break
    else:
        content_type = msg.get_content_type()
        logging.debug("邮件 %s 为单部分，内容类型 %s。", _decode_header_value(msg.get('Subject')), content_type)
        if content_type in ["text/plain", "text/html"]:
            try:
                charset = msg.get_content_charset() or 'utf-8'
                payload = msg.get_payload(decode=True)
                if payload is not None:
                    body = payload.decode(charset, errors='ignore')
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload is not None:
                    body = payload.decode('utf-8', errors='ignore')
    return body


def _parse_email_message(
    msg: email.message.Message,
    sender_list: list[str],
) -> dict[str, str] | None:
    """解析邮件对象并根据发件人筛选返回基础信息。"""

    subject = _decode_header_value(msg.get("Subject"))
    from_ = _decode_header_value(msg.get("From"))
    matched = False
    if sender_list:
        logging.debug("启用发件人过滤规则：%s", ', '.join(sender_list))
        for sender in sender_list:
            if sender.lower() in from_.lower():
                matched = True
                break
    else:
        logging.debug("未配置发件人过滤，全部未读邮件均处理。")
        matched = True
    if not matched:
        logging.debug("邮件 %s 不满足发件人过滤条件，跳过。", subject or '(无主题)')
        return None

    body = _extract_email_body(msg)
    if not body:
        logging.debug("邮件 %s 未能提取正文，跳过。", subject or '(无主题)')
        return None

    return {
        "subject": subject,
        "from": from_,
        "original": body,
    }


def load_config():
    """读取配置文件，不存在则返回空字典"""
    with CONFIG_LOCK:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        return {}


def save_config(config: dict) -> None:
    """保存配置到文件"""
    with CONFIG_LOCK:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def load_logs():
    """读取日志文件，返回列表"""
    with LOG_LOCK:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except Exception:
                    return []
        return []


def save_logs(logs: list) -> None:
    """保存日志到文件"""
    with LOG_LOCK:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)


def append_log_entry(message: str, success: bool) -> None:
    """追加单条日志记录。"""

    logs = load_logs()
    logs.append(
        {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "success": success,
            "message": message,
        }
    )
    save_logs(logs)


def init_queue_storage() -> None:
    """初始化任务队列存储。"""

    with sqlite3.connect(QUEUE_DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                trigger TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def enqueue_email_batch(emails: list[dict[str, str]], trigger: str) -> int:
    """将邮件集合加入任务队列，返回任务 ID。"""

    if not emails:
        raise ValueError("邮件列表为空，无法创建任务。")
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = json.dumps({"emails": emails}, ensure_ascii=False)
    with sqlite3.connect(QUEUE_DB_FILE) as conn:
        cursor = conn.execute(
            """
            INSERT INTO task_queue (payload, status, trigger, attempts, created_at, updated_at)
            VALUES (?, 'pending', ?, 0, ?, ?)
            """,
            (payload, trigger, now_str, now_str),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _finalize_task(task_id: int, status: str, *, last_error: str | None = None) -> None:
    """更新任务状态。"""

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(QUEUE_DB_FILE) as conn:
        conn.execute(
            """
            UPDATE task_queue
            SET status = ?, last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, last_error, now_str, task_id),
        )
        conn.commit()


def _reset_task(task_id: int, *, last_error: str | None = None) -> None:
    """将任务恢复为待处理状态。"""

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(QUEUE_DB_FILE) as conn:
        conn.execute(
            """
            UPDATE task_queue
            SET status = 'pending', last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (last_error, now_str, task_id),
        )
        conn.commit()


def claim_pending_tasks(limit: int) -> list[dict[str, object]]:
    """领取待处理的队列任务并标记为处理中。"""

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    claimed: list[dict[str, object]] = []
    with sqlite3.connect(QUEUE_DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        pending = conn.execute(
            "SELECT id FROM task_queue WHERE status = 'pending' ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        for row in pending:
            cursor = conn.execute(
                """
                UPDATE task_queue
                SET status = 'processing', attempts = attempts + 1, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now_str, row['id']),
            )
            if cursor.rowcount:
                task_row = conn.execute(
                    "SELECT id, payload, trigger, attempts FROM task_queue WHERE id = ?",
                    (row['id'],),
                ).fetchone()
                if task_row:
                    claimed.append(dict(task_row))
        conn.commit()
    return claimed


def get_queue_statistics() -> dict[str, int]:
    """返回当前队列的状态统计。"""

    stats = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    with sqlite3.connect(QUEUE_DB_FILE) as conn:
        for status, count in conn.execute(
            "SELECT status, COUNT(1) FROM task_queue GROUP BY status"
        ):
            stats[str(status)] = int(count)
    stats["total"] = sum(stats.values())
    return stats


def _resolve_deepseek_timeout(config: dict) -> float:
    """解析 DeepSeek 超时时间配置。"""

    raw = config.get('deepseek_timeout')
    if raw is None or raw == "":
        return DEFAULT_DEEPSEEK_TIMEOUT
    try:
        value = float(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        logging.warning(
            "DeepSeek 超时时间配置无效：%s，使用默认值 %s 秒。",
            raw,
            DEFAULT_DEEPSEEK_TIMEOUT,
        )
    return DEFAULT_DEEPSEEK_TIMEOUT


def _send_summary_email(
    processed_emails: list[dict[str, str]],
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    forward_email: str,
) -> None:
    """构建并发送汇总邮件。"""

    if not processed_emails:
        return
    message = MIMEMultipart("alternative")
    message["Subject"] = "自动汇总邮件"
    message["From"] = smtp_user
    message["To"] = forward_email

    html = ["<html><body>"]
    html.append("<h2>邮件目录</h2>")
    html.append("<ul>")
    for idx, item in enumerate(processed_emails, start=1):
        anchor = f"email{idx}"
        summary = item.get("summary", "")
        short_summary = summary[:100]
        subject = item.get("subject", "(无主题)")
        html.append(
            f'<li><a href="#{anchor}">{subject}</a> - {short_summary}</li>'
        )
    html.append("</ul><hr/>")
    for idx, item in enumerate(processed_emails, start=1):
        anchor = f"email{idx}"
        subject = item.get("subject", "(无主题)")
        from_ = item.get("from", "")
        chinese = item.get("chinese", "").replace('\n', '<br/>')
        html.append(f'<h3 id="{anchor}">{subject}</h3>')
        html.append(f'<p><strong>发件人:</strong> {from_}</p>')
        html.append(f'<p>{chinese}</p>')
        html.append("<hr/>")
    html.append("</body></html>")

    part = MIMEText("".join(html), "html", "utf-8")
    message.attach(part)

    server = smtplib.SMTP_SSL(smtp_host, smtp_port)
    try:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [forward_email], message.as_string())
    finally:
        try:
            server.quit()
        except Exception:
            pass


def process_task_queue(batch_size: int = QUEUE_BATCH_LIMIT) -> None:
    """异步处理队列中的邮件翻译与转发任务。"""

    global LAST_QUEUE_CONFIG_WARNING

    tasks = claim_pending_tasks(batch_size)
    if not tasks:
        return

    config = load_config()
    missing = [field for field in QUEUE_REQUIRED_FIELDS if not config.get(field)]
    if missing:
        now = datetime.now()
        if (
            LAST_QUEUE_CONFIG_WARNING is None
            or now - LAST_QUEUE_CONFIG_WARNING >= QUEUE_CONFIG_WARNING_INTERVAL
        ):
            append_log_entry(
                f"队列任务无法执行，缺少配置字段：{', '.join(missing)}",
                False,
            )
            LAST_QUEUE_CONFIG_WARNING = now
        for task in tasks:
            _reset_task(int(task['id']), last_error='配置缺失，等待重试')
        return

    smtp_host = config.get('smtp_host')
    smtp_port = int(config.get('smtp_port', 465))
    smtp_user = config.get('smtp_user')
    smtp_pass = config.get('smtp_pass')
    forward_email = config.get('forward_email')
    deepseek_token = config.get('deepseek_token')
    deepseek_timeout = _resolve_deepseek_timeout(config)

    for task in tasks:
        task_id = int(task['id'])
        attempts = int(task.get('attempts', 0))
        try:
            payload = json.loads(task['payload'])
        except (TypeError, ValueError) as exc:
            append_log_entry(
                f"队列任务 #{task_id} 数据解析失败：{exc}",
                False,
            )
            _finalize_task(task_id, 'failed', last_error='数据解析失败')
            continue

        emails = payload.get('emails') if isinstance(payload, dict) else None
        if not emails:
            _finalize_task(task_id, 'done')
            append_log_entry(
                f"队列任务 #{task_id} 无需处理的邮件，已标记完成。",
                True,
            )
            continue

        processed_emails: list[dict[str, str]] = []
        for email_entry in emails:
            subject = email_entry.get('subject') or '(无主题)'
            from_ = email_entry.get('from') or ''
            original = email_entry.get('original') or ''
            chinese = deepseek_translate(original, deepseek_token, timeout=deepseek_timeout)
            summary = deepseek_summarize(chinese, deepseek_token, timeout=deepseek_timeout)
            processed_emails.append(
                {
                    'subject': subject,
                    'from': from_,
                    'chinese': chinese,
                    'summary': summary,
                }
            )

        try:
            _send_summary_email(
                processed_emails,
                smtp_host,
                smtp_port,
                smtp_user,
                smtp_pass,
                forward_email,
            )
        except Exception as exc:
            logging.exception("队列任务 #%s 转发失败", task_id, exc_info=exc)
            if attempts >= QUEUE_MAX_ATTEMPTS:
                _finalize_task(task_id, 'failed', last_error=str(exc))
                append_log_entry(
                    f"队列任务 #{task_id} 达到最大重试次数，已标记失败：{exc}",
                    False,
                )
            else:
                _reset_task(task_id, last_error=str(exc))
            continue

        _finalize_task(task_id, 'done')
        append_log_entry(
            f"队列任务 #{task_id} 已完成，成功转发 {len(processed_emails)} 封邮件。",
            True,
        )


def _normalize_schedule_config(config: dict | None = None) -> dict[str, object]:
    """整理调度配置，返回规范化结果。"""

    config = config or load_config()
    mode = str(config.get('schedule_mode') or 'minutes').lower()
    if mode not in {'minutes', 'days', 'fixed_hours'}:
        mode = 'minutes'
    normalized: dict[str, object] = {'mode': mode}
    if mode == 'minutes':
        minutes_raw = str(config.get('schedule_interval_minutes') or '').strip()
        try:
            minutes = int(minutes_raw)
        except ValueError:
            minutes = 60
        if minutes <= 0:
            minutes = 60
        normalized['minutes'] = minutes
    elif mode == 'days':
        days_raw = str(config.get('schedule_interval_days') or '').strip()
        try:
            days = int(days_raw)
        except ValueError:
            days = 1
        if days <= 0:
            days = 1
        time_raw = str(config.get('schedule_daily_time') or '09:00').strip()
        try:
            hour_str, minute_str = time_raw.split(':', 1)
            hour = int(hour_str)
            minute = int(minute_str)
        except ValueError:
            hour, minute = 9, 0
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        normalized['days'] = days
        normalized['time'] = f"{hour:02d}:{minute:02d}"
    else:
        hours_raw = str(config.get('schedule_fixed_hours') or '')
        hours: list[int] = []
        for token in hours_raw.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                continue
            if 0 <= value <= 23:
                hours.append(value)
        hours = sorted(set(hours))
        if not hours:
            hours = [9]
        normalized['hours'] = hours
    return normalized


def _build_schedule_trigger(
    config: dict | None = None,
) -> tuple[IntervalTrigger | CronTrigger, dict[str, object]]:
    """根据配置生成 APScheduler Trigger。"""

    normalized = _normalize_schedule_config(config)
    mode = normalized['mode']
    if mode == 'minutes':
        trigger = IntervalTrigger(minutes=int(normalized['minutes']))
    elif mode == 'days':
        days = int(normalized['days'])
        hour_str, minute_str = str(normalized['time']).split(':', 1)
        hour = int(hour_str)
        minute = int(minute_str)
        now = datetime.now()
        start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if start <= now:
            start += timedelta(days=days)
        trigger = IntervalTrigger(days=days, start_date=start)
    else:
        trigger = CronTrigger(hour=list(normalized['hours']), minute=0)
    return trigger, normalized


def describe_schedule(config: dict | None = None) -> str:
    """返回人类可读的调度描述。"""

    normalized = _normalize_schedule_config(config)
    mode = normalized['mode']
    if mode == 'minutes':
        return f"每 {normalized['minutes']} 分钟执行一次"
    if mode == 'days':
        return f"每 {normalized['days']} 天在 {normalized['time']} 执行"
    hours = ', '.join(f"{hour:02d}:00" for hour in normalized['hours'])
    return f"每日 {hours} 执行"


def format_datetime(dt: datetime | None) -> str | None:
    """格式化日期时间用于展示。"""

    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def get_next_run_time() -> datetime | None:
    """读取调度任务的下一次执行时间。"""

    if SCHEDULER is None:
        return None
    job = SCHEDULER.get_job(SCHEDULER_JOB_ID)
    if job is None:
        return None
    return job.next_run_time


def refresh_scheduler_job(*, config: dict | None = None) -> None:
    """根据配置刷新定时任务。"""

    if SCHEDULER is None:
        return
    trigger, _ = _build_schedule_trigger(config or load_config())
    try:
        SCHEDULER.remove_job(SCHEDULER_JOB_ID)
    except JobLookupError:
        pass
    SCHEDULER.add_job(
        process_emails,
        trigger=trigger,
        args=('scheduler',),
        id=SCHEDULER_JOB_ID,
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )


def ensure_queue_worker_job() -> None:
    """确保队列轮询任务已注册。"""

    if SCHEDULER is None:
        return
    if SCHEDULER.get_job(QUEUE_WORKER_JOB_ID) is None:
        SCHEDULER.add_job(
            run_queue_worker,
            trigger=IntervalTrigger(seconds=10),
            id=QUEUE_WORKER_JOB_ID,
            coalesce=True,
            max_instances=1,
        )


def run_queue_worker() -> None:
    """后台执行队列任务。"""

    try:
        process_task_queue()
    except Exception:
        logging.exception("后台队列任务执行失败")


def start_scheduler() -> None:
    """启动或刷新调度器。"""

    global SCHEDULER

    if SCHEDULER is None:
        SCHEDULER = BackgroundScheduler()
        SCHEDULER.start()
    refresh_scheduler_job()
    ensure_queue_worker_job()


def _call_deepseek_chat(
    messages: list[dict[str, str]],
    token: str,
    *,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int | None = 1024,
    timeout: float = DEFAULT_DEEPSEEK_TIMEOUT,
) -> str | None:
    """Invoke DeepSeek chat completions API and return the assistant content."""

    if not token:
        logging.debug("未配置 DeepSeek token，跳过 API 调用。")
        return None
    if not messages:
        logging.debug("DeepSeek 请求缺少消息内容，跳过调用。")
        return None

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        resp = requests.post(
            DEEPSEEK_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logging.warning("DeepSeek API 请求异常：%s", exc)
        return None

    if resp.status_code != 200:
        detail = resp.text[:200] if resp.text else resp.status_code
        logging.warning("DeepSeek API 返回状态 %s：%s", resp.status_code, detail)
        return None

    try:
        data = resp.json()
    except ValueError:
        logging.warning("DeepSeek API 响应 JSON 解析失败。")
        return None

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        logging.warning("DeepSeek API 响应中未包含 choices 字段。")
        return None

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        logging.warning("DeepSeek API 响应 message 字段格式异常。")
        return None

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    logging.warning("DeepSeek API 响应缺少有效内容。")
    return None


def deepseek_translate(
    text: str,
    token: str,
    *,
    timeout: float = DEFAULT_DEEPSEEK_TIMEOUT,
) -> str:
    """调用 DeepSeek API 将文本翻译成中文。

    这里给出示例实现，实际使用时请根据 DeepSeek 官方文档调整接口地址和参数。
    若调用失败则返回原文本。
    """
    if not text:
        return text

    prompt_text = _trim_text_for_prompt(text)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名专业的邮件翻译助手，负责将英文或其他语言的邮件精准翻译为自然流畅的简体中文。"
                "请严格保留原文的结构、段落、列表、标点和格式，不要省略或合并任何信息，也不要添加额外说明。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请按照原文格式完整翻译以下邮件正文为简体中文，用于后续邮件转发。"
                "不要生成目录或摘要，直接输出全文译文：\n\n"
                f"{prompt_text}"
            ),
        },
    ]

    translated = _call_deepseek_chat(
        messages,
        token,
        temperature=0.2,
        max_tokens=2048,
        timeout=timeout,
    )
    if translated:
        return translated
    return text


def deepseek_summarize(
    text: str,
    token: str,
    *,
    timeout: float = DEFAULT_DEEPSEEK_TIMEOUT,
) -> str:
    """调用 DeepSeek API 对中文文本进行摘要。

    示例实现，若调用失败则返回原文本的前 200 个字符。
    """
    if not text:
        return text

    prompt_text = _trim_text_for_prompt(text)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名专业的中文邮件助理，需要从邮件正文中提炼重点，突出关键信息和待办事项。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请基于以下邮件正文生成一段不超过 150 字的中文摘要，突出主题、关键信息和可执行事项。"
                "无需重复问候语，若无行动项可以省略：\n\n"
                f"{prompt_text}"
            ),
        },
    ]

    summary = _call_deepseek_chat(
        messages,
        token,
        temperature=0.4,
        max_tokens=400,
        timeout=timeout,
    )
    if summary:
        return summary
    return text[:200]


def fetch_imap_mailboxes(
    host: str,
    port: int,
    username: str,
    password: str,
    id_params: dict[str, str] | None = None,
) -> list[str]:
    """验证 IMAP 连接并返回邮箱文件夹名称列表。"""
    mail = imaplib.IMAP4_SSL(host, int(port))
    try:
        mail.login(username, password)
        _send_imap_id(mail, id_params)
        mailboxes = _list_imap_mailboxes(mail)
        logging.info("IMAP 验证成功，可用邮箱列表：%s", ", ".join(mailboxes))
        return mailboxes
    finally:
        try:
            mail.logout()
        except Exception:
            pass



    
def _trigger_label(trigger: str) -> str:
    """转换任务来源标签。"""

    return {'manual': '手动', 'scheduler': '定时', 'queue': '队列'}.get(trigger, trigger)


def process_emails(trigger: str = 'scheduler') -> str:
    """核心任务：读取邮件并入队等待异步处理。"""

    config = load_config()
    trigger_label = _trigger_label(trigger)

    mail_protocol = str(config.get('mail_protocol') or 'imap').lower()
    if mail_protocol not in {'imap', 'pop3'}:
        mail_protocol = 'imap'

    required_fields = (
        (MAIL_IMAP_REQUIRED if mail_protocol == 'imap' else MAIL_POP3_REQUIRED)
        + MAIL_COMMON_REQUIRED
    )
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        message = f"{trigger_label}任务无法执行，缺少配置字段：{', '.join(missing_fields)}"
        append_log_entry(message, False)
        return message

    sender_list = [s.strip() for s in str(config.get('senders', '')).split(',') if s.strip()]
    pending_emails: list[dict[str, str]] = []

    try:
        if mail_protocol == 'pop3':
            pop_host = config.get('pop3_host')
            pop_port = int(config.get('pop3_port', 995))
            pop_user = config.get('pop3_user')
            pop_pass = config.get('pop3_pass')
            pop_use_ssl_raw = config.get('pop3_use_ssl', 'true')
            use_ssl = str(pop_use_ssl_raw).lower() not in {'false', '0', 'no', 'off'}
            pop_conn: poplib.POP3 | None = None
            try:
                pop_conn = poplib.POP3_SSL(pop_host, pop_port) if use_ssl else poplib.POP3(pop_host, pop_port)
                pop_conn.user(pop_user)
                pop_conn.pass_(pop_pass)
                try:
                    welcome = pop_conn.getwelcome()
                    welcome_text = welcome.decode('utf-8', errors='ignore') if isinstance(welcome, bytes) else str(welcome)
                    logging.info("POP3 登录成功，欢迎语：%s", welcome_text)
                except Exception:
                    logging.debug("POP3 登录成功，但欢迎语解析失败。", exc_info=True)

                processed_uidls = config.get('pop3_processed_uidls')
                if not isinstance(processed_uidls, list):
                    processed_uidls = []
                seen_uidls = set(str(uid) for uid in processed_uidls)

                uidl_supported = True
                try:
                    _, uidl_entries, _ = pop_conn.uidl()
                    raw_uidl_entries = list(uidl_entries or [])
                    logging.debug("POP3 UIDL 响应共 %d 条。", len(raw_uidl_entries))
                except poplib.error_proto as exc:
                    uidl_supported = False
                    logging.warning("POP3 服务器不支持 UIDL 命令：%s，退回使用 LIST。", exc)
                    _, list_entries, _ = pop_conn.list()
                    raw_uidl_entries = []
                    for entry in list_entries or []:
                        try:
                            decoded = entry.decode()
                            number = decoded.split()[0]
                        except Exception:
                            continue
                        raw_uidl_entries.append(f"{number} {number}".encode())

                new_uid_pairs: list[tuple[str, str]] = []
                for raw in raw_uidl_entries:
                    try:
                        decoded = raw.decode()
                    except Exception:
                        continue
                    parts = decoded.split()
                    if len(parts) < 2:
                        continue
                    number, uidl = parts[0], parts[1]
                    normalized_uidl = uidl if uidl_supported else f"list-{uidl}"
                    if normalized_uidl not in seen_uidls:
                        new_uid_pairs.append((number, normalized_uidl))
                new_uid_pairs.sort(key=lambda pair: int(pair[0]))
                logging.info("POP3 共检测到 %d 封新邮件。", len(new_uid_pairs))

                processed_this_run: list[str] = []
                for number, normalized_uidl in new_uid_pairs:
                    try:
                        _, lines, _ = pop_conn.retr(int(number))
                    except Exception as exc:
                        logging.warning("POP3 拉取第 %s 封邮件失败：%s", number, exc)
                        continue
                    processed_this_run.append(normalized_uidl)
                    msg_content = b"\r\n".join(lines)
                    msg = email.message_from_bytes(msg_content)
                    parsed = _parse_email_message(msg, sender_list)
                    if parsed:
                        pending_emails.append(parsed)

                if processed_this_run:
                    for uidl in processed_this_run:
                        if uidl not in seen_uidls:
                            processed_uidls.append(uidl)
                            seen_uidls.add(uidl)
                    max_history = 500
                    if len(processed_uidls) > max_history:
                        processed_uidls = processed_uidls[-max_history:]
                    config['pop3_processed_uidls'] = processed_uidls
                    save_config(config)
            finally:
                if pop_conn:
                    try:
                        pop_conn.quit()
                    except Exception:
                        pass
        else:
            imap_host = config.get('imap_host')
            imap_port = int(config.get('imap_port', 993))
            imap_user = config.get('imap_user')
            imap_pass = config.get('imap_pass')
            imap_id_params = _build_imap_id_params(config)
            target_mailbox = config.get('imap_mailbox') or 'INBOX'
            mail: imaplib.IMAP4 | None = None
            try:
                mail = imaplib.IMAP4_SSL(imap_host, imap_port)
                _send_imap_id(mail, imap_id_params)
                try:
                    encoded_mailbox = _imap_encode_utf7(target_mailbox)
                    if encoded_mailbox != target_mailbox:
                        logging.info("IMAP 邮箱名称 UTF-7 编码: %s -> %s", target_mailbox, encoded_mailbox)
                    else:
                        logging.debug("IMAP 邮箱名称无需 UTF-7 编码: %s", target_mailbox)
                except Exception as exc:
                    logging.warning("IMAP 邮箱名称 UTF-7 编码失败，使用原值: %s，错误: %s", target_mailbox, exc)
                    encoded_mailbox = target_mailbox
                status, data = mail.select(encoded_mailbox)
                if status == 'OK':
                    logging.info("IMAP 邮箱 %s 选择成功。", target_mailbox)
                detail = ''
                if status != 'OK':
                    if data:
                        try:
                            detail = data[0].decode('utf-8', errors='ignore')
                        except Exception:
                            detail = str(data)
                        logging.debug("IMAP 选择失败的原始响应: %s", detail)
                    lower_detail = detail.lower()
                    if 'unsafe login' in lower_detail:
                        logging.warning("选择邮箱 %s 时触发 Unsafe Login，尝试获取邮箱列表后重试。", target_mailbox)
                        try:
                            available_mailboxes = _list_imap_mailboxes(mail)
                        except Exception as exc:
                            logging.warning("获取邮箱列表用于重试失败：%s", exc)
                            available_mailboxes = []
                        else:
                            logging.info("成功获取 %d 个邮箱用于重试。", len(available_mailboxes))
                        fallback_mailbox = _find_inbox_name(available_mailboxes)
                        if fallback_mailbox and fallback_mailbox != target_mailbox:
                            logging.info("使用备用收件箱名称 %s 重试选择。", fallback_mailbox)
                            try:
                                encoded_mailbox = _imap_encode_utf7(fallback_mailbox)
                            except Exception:
                                encoded_mailbox = fallback_mailbox
                                logging.debug("备用邮箱 %s 使用原值进行选择。", fallback_mailbox)
                            retry_status, retry_data = mail.select(encoded_mailbox)
                            if retry_status == 'OK':
                                logging.info("备用邮箱 %s 选择成功。", fallback_mailbox)
                                status = retry_status
                                data = retry_data
                                target_mailbox = fallback_mailbox
                                detail = ''
                            else:
                                retry_detail = ''
                                if retry_data:
                                    try:
                                        retry_detail = retry_data[0].decode('utf-8', errors='ignore')
                                    except Exception:
                                        retry_detail = str(retry_data)
                                logging.warning("备用邮箱 %s 选择失败: %s", fallback_mailbox, retry_detail)
                        elif fallback_mailbox == target_mailbox:
                            logging.info("获取到的备用邮箱与当前目标一致：%s，跳过重试。", fallback_mailbox)
                        else:
                            logging.warning("未找到可用的备用收件箱用于重试。")
                    if status != 'OK':
                        if 'unsafe login' not in lower_detail:
                            logging.error("IMAP 选择邮箱 %s 失败，原因: %s", target_mailbox, detail or '未知')
                        raise Exception(f"IMAP 选择邮箱失败：{target_mailbox}{(' - ' + detail) if detail else ''}")
                status, messages = mail.search(None, 'UNSEEN')
                if status != 'OK':
                    raise Exception('IMAP 搜索失败')
                email_ids = messages[0].split()
                for num in email_ids:
                    res, data = mail.fetch(num, '(RFC822)')
                    if res != 'OK':
                        continue
                    msg = email.message_from_bytes(data[0][1])
                    parsed = _parse_email_message(msg, sender_list)
                    if parsed:
                        pending_emails.append(parsed)
            finally:
                if mail:
                    try:
                        mail.logout()
                    except Exception:
                        pass
    except Exception as exc:
        message = f"{trigger_label}任务处理邮件出现异常: {exc}"
        logging.exception("邮件抓取过程失败", exc_info=exc)
        append_log_entry(message, False)
        return message

    if not pending_emails:
        message = f"{trigger_label}任务执行完毕，没有符合条件的未读邮件。"
        append_log_entry(message, True)
        return message

    try:
        task_id = enqueue_email_batch(pending_emails, trigger)
    except Exception as exc:
        logging.exception("创建队列任务失败", exc_info=exc)
        message = f"{trigger_label}任务发现 {len(pending_emails)} 封邮件但创建队列失败：{exc}"
        append_log_entry(message, False)
        return message

    message = f"{trigger_label}任务检测到 {len(pending_emails)} 封未读邮件，已创建队列任务 #{task_id}。"
    append_log_entry(message, True)
    return message


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page for email_owner user."""

    next_candidate = request.args.get('next') or request.form.get('next')
    next_url = _safe_next_url(next_candidate)
    if is_authenticated():
        return redirect(next_url)

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        if username != AUTH_USER:
            flash('用户名或密码错误。')
        elif not AUTH_PASSWORD_HASH:
            flash('系统尚未配置登录密码，请联系管理员。')
        elif verify_password(password, AUTH_PASSWORD_HASH):
            session[SESSION_KEY] = AUTH_USER
            flash('登录成功。')
            return redirect(_safe_next_url(request.form.get('next')))
        else:
            flash('用户名或密码错误。')
    return render_template('login.html', next_url=next_url)


@app.route('/logout')
def logout():
    """Logout current user and redirect to login page."""

    session.pop(SESSION_KEY, None)
    flash('已退出登录。')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """首页：显示最近一次运行结果和最近日志"""
    logs = load_logs()
    last_run = logs[-1] if logs else None
    # 仅展示最近 10 条
    recent_logs = logs[-10:]
    queue_stats = get_queue_statistics()
    next_run_display = format_datetime(get_next_run_time())
    schedule_desc = describe_schedule()
    return render_template(
        'index.html',
        last_run=last_run,
        logs=recent_logs,
        queue_stats=queue_stats,
        next_run_time=next_run_display,
        schedule_description=schedule_desc,
    )


@app.route('/run', methods=['POST'])
@login_required
def run_now():
    """手动触发任务"""
    message = process_emails('manual')
    flash(message or "任务已执行。")
    return redirect(url_for('index'))


@app.route('/config', methods=['GET', 'POST'])
@login_required
def config_page():
    """配置页面：显示或保存配置"""
    config = load_config()
    mailboxes = config.get('available_mailboxes', [])
    schedule_state = _normalize_schedule_config(config)
    schedule_hours_display = ','.join(
        f"{hour:02d}" for hour in schedule_state.get('hours', [])
    )
    next_run_display = format_datetime(get_next_run_time())
    schedule_desc = describe_schedule(config)
    if request.method == 'POST':
        action = request.form.get('action', 'save')
        if action == 'update_password':
            new_password = (request.form.get('new_password') or '').strip()
            confirm_password = (request.form.get('confirm_password') or '').strip()
            if not new_password:
                flash('新密码不能为空。')
            elif new_password != confirm_password:
                flash('两次输入的密码不一致。')
            else:
                set_auth_password(new_password, config=config)
                flash('登录密码已更新。')
            return redirect(url_for('config_page'))
        # 保存用户提交的配置
        mail_protocol = request.form.get('mail_protocol', 'imap').strip().lower()
        if mail_protocol not in {'imap', 'pop3'}:
            mail_protocol = 'imap'
        config['mail_protocol'] = mail_protocol
        schedule_mode = (request.form.get('schedule_mode') or 'minutes').strip().lower()
        if schedule_mode not in {'minutes', 'days', 'fixed_hours'}:
            schedule_mode = 'minutes'
        config['schedule_mode'] = schedule_mode
        config['imap_host'] = request.form.get('imap_host', '').strip()
        config['imap_port'] = request.form.get('imap_port', '').strip()
        config['imap_user'] = request.form.get('imap_user', '').strip()
        config['imap_pass'] = request.form.get('imap_pass', '').strip()
        config['imap_mailbox'] = request.form.get('imap_mailbox', '').strip()
        config['imap_id_name'] = request.form.get('imap_id_name', '').strip()
        config['imap_id_version'] = request.form.get('imap_id_version', '').strip()
        config['imap_id_vendor'] = request.form.get('imap_id_vendor', '').strip()
        config['imap_id_support_url'] = request.form.get('imap_id_support_url', '').strip()
        config['imap_id_support_email'] = request.form.get('imap_id_support_email', '').strip()
        config['pop3_host'] = request.form.get('pop3_host', '').strip()
        config['pop3_port'] = request.form.get('pop3_port', '').strip()
        config['pop3_user'] = request.form.get('pop3_user', '').strip()
        config['pop3_pass'] = request.form.get('pop3_pass', '').strip()
        config['pop3_use_ssl'] = 'true' if request.form.get('pop3_use_ssl') else 'false'
        config['smtp_host'] = request.form.get('smtp_host', '').strip()
        config['smtp_port'] = request.form.get('smtp_port', '').strip()
        config['smtp_user'] = request.form.get('smtp_user', '').strip()
        config['smtp_pass'] = request.form.get('smtp_pass', '').strip()
        config['forward_email'] = request.form.get('forward_email', '').strip()
        config['senders'] = request.form.get('senders', '').strip()
        config['deepseek_token'] = request.form.get('deepseek_token', '').strip()
        timeout_value = request.form.get('deepseek_timeout', '').strip()
        default_timeout_str = f"{DEFAULT_DEEPSEEK_TIMEOUT:g}"
        config['deepseek_timeout'] = timeout_value or default_timeout_str
        config['schedule_interval_minutes'] = (
            request.form.get('schedule_interval_minutes', '').strip()
        )
        config['schedule_interval_days'] = (
            request.form.get('schedule_interval_days', '').strip()
        )
        config['schedule_daily_time'] = (
            request.form.get('schedule_daily_time', '').strip()
        )
        config['schedule_fixed_hours'] = (
            request.form.get('schedule_fixed_hours', '').strip()
        )

        if action == 'fetch_mailboxes':
            if config['mail_protocol'] == 'pop3':
                flash("POP3 协议不支持获取邮箱文件夹列表。")
                schedule_state = _normalize_schedule_config(config)
                schedule_hours_display = ','.join(
                    f"{hour:02d}" for hour in schedule_state.get('hours', [])
                )
                schedule_desc = describe_schedule(config)
                next_run_display = format_datetime(get_next_run_time())
                return render_template(
                    'config.html',
                    config=config,
                    mailboxes=mailboxes,
                    fetched_mailboxes=None,
                    default_imap_id=DEFAULT_IMAP_ID,
                    deepseek_timeout_default=DEFAULT_DEEPSEEK_TIMEOUT,
                    schedule_state=schedule_state,
                    schedule_hours_display=schedule_hours_display,
                    schedule_description=schedule_desc,
                    next_run_time=next_run_display,
                )
            try:
                id_params = _build_imap_id_params(config)
                mailboxes = fetch_imap_mailboxes(
                    config['imap_host'],
                    int(config['imap_port'] or 993),
                    config['imap_user'],
                    config['imap_pass'],
                    id_params,
                )
                config['available_mailboxes'] = mailboxes
                if mailboxes and config['imap_mailbox'] not in mailboxes:
                    config['imap_mailbox'] = ''
                save_config(config)
                refresh_scheduler_job(config=config)
                flash("邮箱验证成功，已获取可用列表。")
            except Exception as exc:
                flash(f"获取邮箱列表失败：{exc}")
            schedule_state = _normalize_schedule_config(config)
            schedule_hours_display = ','.join(
                f"{hour:02d}" for hour in schedule_state.get('hours', [])
            )
            schedule_desc = describe_schedule(config)
            next_run_display = format_datetime(get_next_run_time())
            return render_template(
                'config.html',
                config=config,
                mailboxes=mailboxes,
                fetched_mailboxes=mailboxes,
                default_imap_id=DEFAULT_IMAP_ID,
                deepseek_timeout_default=DEFAULT_DEEPSEEK_TIMEOUT,
                schedule_state=schedule_state,
                schedule_hours_display=schedule_hours_display,
                schedule_description=schedule_desc,
                next_run_time=next_run_display,
            )

        config['available_mailboxes'] = mailboxes
        save_config(config)
        refresh_scheduler_job(config=config)
        flash("配置已保存。")
        return redirect(url_for('config_page'))
    return render_template(
        'config.html',
        config=config,
        mailboxes=mailboxes,
        fetched_mailboxes=None,
        default_imap_id=DEFAULT_IMAP_ID,
        deepseek_timeout_default=DEFAULT_DEEPSEEK_TIMEOUT,
        schedule_state=schedule_state,
        schedule_hours_display=schedule_hours_display,
        schedule_description=schedule_desc,
        next_run_time=next_run_display,
    )


@app.route('/logs')
@login_required
def logs_page():
    """日志页面：查看全部日志"""
    logs = load_logs()
    return render_template('logs.html', logs=logs)


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
    parser = argparse.ArgumentParser(description='邮件汇总系统后台服务')
    parser.add_argument('--pwd', dest='default_password', help='email_owner 用户默认密码')
    args = parser.parse_args()

    initialize_auth(args.default_password)

    init_queue_storage()
    # 开启定时任务
    start_scheduler()
    # 启动 Flask
    host = os.environ.get('APP_HOST', '0.0.0.0')
    port = get_runtime_port()
    app.run(host=host, port=port)
else:
    initialize_auth(None)
    init_queue_storage()
    start_scheduler()
