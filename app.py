import os
import json
import logging
import re
import base64
import argparse
import hashlib
from functools import wraps
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, render_template, request, redirect, url_for, flash, session
from apscheduler.schedulers.background import BackgroundScheduler
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

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"


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
    deepseek_token: str,
) -> dict[str, str] | None:
    """解析邮件对象并根据发件人筛选返回处理结果。"""

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

    chinese = deepseek_translate(body, deepseek_token)
    summary = deepseek_summarize(chinese, deepseek_token)
    return {
        "subject": subject,
        "from": from_,
        "original": body,
        "chinese": chinese,
        "summary": summary,
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


def _call_deepseek_chat(
    messages: list[dict[str, str]],
    token: str,
    *,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int | None = 1024,
    timeout: int = 30,
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


def deepseek_translate(text: str, token: str) -> str:
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

    translated = _call_deepseek_chat(messages, token, temperature=0.2, max_tokens=2048)
    if translated:
        return translated
    return text


def deepseek_summarize(text: str, token: str) -> str:
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

    summary = _call_deepseek_chat(messages, token, temperature=0.4, max_tokens=400)
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


def process_emails():
    """核心任务：读取邮件、翻译、汇总并转发"""
    config = load_config()
    logs = load_logs()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {"time": timestamp, "success": False, "message": ""}

    mail_protocol = str(config.get('mail_protocol') or 'imap').lower()
    if mail_protocol not in {'imap', 'pop3'}:
        mail_protocol = 'imap'

    common_required = [
        'smtp_host', 'smtp_port', 'smtp_user', 'smtp_pass',
        'forward_email', 'senders', 'deepseek_token'
    ]
    protocol_required = [
        'imap_host', 'imap_port', 'imap_user', 'imap_pass'
    ] if mail_protocol == 'imap' else [
        'pop3_host', 'pop3_port', 'pop3_user', 'pop3_pass'
    ]
    required_fields = protocol_required + common_required
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        logging.error("任务配置缺失字段：%s", ', '.join(missing_fields))
        log_entry['message'] = f"配置项缺失：{', '.join(missing_fields)}。请在配置页面完善。"
        logs.append(log_entry)
        save_logs(logs)
        return
    logging.debug("所有必要配置均已填写。")

    smtp_host = config.get('smtp_host')
    smtp_port = int(config.get('smtp_port', 465))
    smtp_user = config.get('smtp_user')
    smtp_pass = config.get('smtp_pass')
    forward_email = config.get('forward_email')
    sender_list = [s.strip() for s in config.get('senders', '').split(',') if s.strip()]
    deepseek_token = config.get('deepseek_token')

    processed_emails: list[dict[str, str]] = []
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
                    parsed = _parse_email_message(msg, sender_list, deepseek_token)
                    if parsed:
                        processed_emails.append(parsed)

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
                mail.login(imap_user, imap_pass)
                _send_imap_id(mail, imap_id_params)
                encoded_mailbox = target_mailbox
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
                    parsed = _parse_email_message(msg, sender_list, deepseek_token)
                    if parsed:
                        processed_emails.append(parsed)
            finally:
                if mail:
                    try:
                        mail.logout()
                    except Exception:
                        pass
    except Exception as e:
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
    return render_template('index.html', last_run=last_run, logs=recent_logs)


@app.route('/run', methods=['POST'])
@login_required
def run_now():
    """手动触发任务"""
    process_emails()
    flash("任务已执行。")
    return redirect(url_for('index'))


@app.route('/config', methods=['GET', 'POST'])
@login_required
def config_page():
    """配置页面：显示或保存配置"""
    config = load_config()
    mailboxes = config.get('available_mailboxes', [])
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

        if action == 'fetch_mailboxes':
            if config['mail_protocol'] == 'pop3':
                flash("POP3 协议不支持获取邮箱文件夹列表。")
                return render_template(
                    'config.html',
                    config=config,
                    mailboxes=mailboxes,
                    fetched_mailboxes=None,
                    default_imap_id=DEFAULT_IMAP_ID,
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
                flash("邮箱验证成功，已获取可用列表。")
            except Exception as exc:
                flash(f"获取邮箱列表失败：{exc}")
            return render_template(
                'config.html',
                config=config,
                mailboxes=mailboxes,
                fetched_mailboxes=mailboxes,
                default_imap_id=DEFAULT_IMAP_ID,
            )

        config['available_mailboxes'] = mailboxes
        save_config(config)
        flash("配置已保存。")
        return redirect(url_for('config_page'))
    return render_template(
        'config.html',
        config=config,
        mailboxes=mailboxes,
        fetched_mailboxes=None,
        default_imap_id=DEFAULT_IMAP_ID,
    )


@app.route('/logs')
@login_required
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
    parser = argparse.ArgumentParser(description='邮件汇总系统后台服务')
    parser.add_argument('--pwd', dest='default_password', help='email_owner 用户默认密码')
    args = parser.parse_args()

    initialize_auth(args.default_password)

    # 开启定时任务
    start_scheduler()
    # 启动 Flask
    host = os.environ.get('APP_HOST', '0.0.0.0')
    port = get_runtime_port()
    app.run(host=host, port=port)
else:
    initialize_auth(None)
