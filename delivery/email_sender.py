"""Email delivery via Resend API.

Resend is used over smtplib for reliability, deliverability, and a clean
free tier (3,000 emails/month). The twice-daily digest for ≤5 recipients
will never exceed this limit.

Docs: https://resend.com/docs/api-reference/emails/send-email

SDK compatibility note
----------------------
Resend's Python SDK has two incompatible call signatures across versions:
  - v0.x / v1.x: resend.Emails.send({"from": ..., "to": ..., ...})  (plain dict)
  - v2.x+:       resend.Emails.send(SendParams(from_=..., to=..., ...))

We use the plain dict form because it works on both old and new SDK versions.
The key must be "from" (not "from_") in the dict.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Literal

import resend

from config.loader import get_settings

logger = logging.getLogger(__name__)

DigestPeriod = Literal["morning", "evening"]


def _format_date(dt: datetime) -> str:
    """Return a date string like 'May 22, 2026' without a leading zero on the day.

    strftime's %-d (no-pad day) is Linux-only; %#d is Windows-only.
    Stripping the leading zero from %d is portable across all platforms.
    """
    return dt.strftime("%B %d, %Y").replace(" 0", " ")


class EmailSender:
    def __init__(self) -> None:
        self.settings = get_settings()
        api_key = os.getenv("RESEND_API_KEY", "")
        if not api_key:
            raise EnvironmentError("RESEND_API_KEY is not set. Add it to your .env file.")
        resend.api_key = api_key
        self._from_addr = os.getenv("NEWSBOT_EMAIL_FROM", "")
        if not self._from_addr:
            raise EnvironmentError("NEWSBOT_EMAIL_FROM is not set. Add it to your .env file.")
        raw_to = os.getenv("NEWSBOT_EMAIL_TO", "")
        if not raw_to:
            raise EnvironmentError("NEWSBOT_EMAIL_TO is not set. Add it to your .env file.")
        self._to_addrs = [addr.strip() for addr in raw_to.split(",") if addr.strip()]
        max_recipients = self.settings["delivery"]["email"]["max_recipients"]
        if len(self._to_addrs) > max_recipients:
            logger.warning(
                "Recipient list (%d) exceeds max_recipients (%d). Truncating.",
                len(self._to_addrs), max_recipients
            )
            self._to_addrs = self._to_addrs[:max_recipients]

    def send(self, html: str, period: DigestPeriod, run_date: datetime) -> bool:
        """Send the digest. Returns True on success, False on failure."""
        delivery_cfg = self.settings["delivery"]["email"]
        subject_template = (
            delivery_cfg["subject_morning"]
            if period == "morning"
            else delivery_cfg["subject_evening"]
        )
        subject = subject_template.format(date=_format_date(run_date))

        try:
            # Use plain dict form — compatible with both resend SDK v0/v1 and v2+.
            # "from" (not "from_") is the required key name in the dict interface.
            params: dict = {
                "from": self._from_addr,
                "to": self._to_addrs,
                "subject": subject,
                "html": html,
            }
            result = resend.Emails.send(params)
            logger.info(
                "Email sent: id=%s to=%s subject='%s'",
                result.get("id", "unknown"), self._to_addrs, subject
            )
            return True
        except Exception as exc:
            logger.error("Failed to send email: %s", exc)
            return False
