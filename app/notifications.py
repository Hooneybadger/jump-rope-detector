from __future__ import annotations

import smtplib
from email.message import EmailMessage

from .config import get_settings


class DeliveryNotConfigured(RuntimeError):
    pass


def send_email(destination: str, subject: str, body: str) -> str:
    settings = get_settings()
    if not settings.smtp_host or not settings.smtp_from:
        raise DeliveryNotConfigured("이메일 발송 설정이 필요합니다.")
    message = EmailMessage()
    message["From"] = settings.smtp_from
    message["To"] = destination
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        if settings.smtp_starttls:
            smtp.starttls()
        if settings.smtp_username:
            smtp.login(settings.smtp_username, settings.smtp_password)
        smtp.send_message(message)
    return message.get("Message-ID", "smtp-accepted")

