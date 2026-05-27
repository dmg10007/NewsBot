"""Telegram delivery stub.

Disabled by default. Enable by setting:
  delivery.telegram.enabled: true  in settings.yaml
  NEWSBOT_TELEGRAM_TOKEN and NEWSBOT_TELEGRAM_CHAT_ID in .env

When enabled, sends a plain-text version of the digest as a series of
Telegram messages (one per topic section). Telegram's 4096-char message
limit means we split long digests into chunks.

Full implementation requires: python-telegram-bot>=21.0
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Literal

from domain.models import DigestStory

logger = logging.getLogger(__name__)

TOPIC_ORDER = ["politics", "economy", "current_events"]
TOPIC_EMOJI = {
    "politics": "\U0001f3db",       # classical building
    "economy": "\U0001f4b9",        # chart with upward trend
    "current_events": "\U0001f4f0", # newspaper
}
TELEGRAM_MAX_CHARS = 4096


class TelegramSender:
    def __init__(self) -> None:
        self._token = os.getenv("NEWSBOT_TELEGRAM_TOKEN", "")
        self._chat_id = os.getenv("NEWSBOT_TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)

    def send_digest(self, summaries: list[DigestStory], period: str) -> bool:
        """Send a full digest as chunked Telegram messages."""
        return self._send(summaries, period=period, run_date=datetime.now())

    def send_alert(self, message: str) -> None:
        """Send a plain-text alert message (e.g. failure notification)."""
        if not self._enabled:
            logger.debug("Telegram not configured — skipping alert")
            return
        try:
            import telegram
            import asyncio

            async def _send() -> None:
                bot = telegram.Bot(token=self._token)
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=message,
                    disable_web_page_preview=True,
                )

            asyncio.run(_send())
        except ImportError:
            logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot>=21.0")
        except Exception as exc:
            logger.error("Telegram alert failed: %s", exc)

    def _send(self, summaries: list[DigestStory], period: str, run_date: datetime) -> bool:
        if not self._enabled:
            logger.debug("Telegram delivery is disabled or unconfigured. Skipping.")
            return False

        try:
            import telegram
            import asyncio
        except ImportError:
            logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot>=21.0")
            return False

        period_label = "\U0001f305 Morning" if period == "morning" else "\U0001f307 Evening"
        date_str = run_date.strftime("%b %d, %Y")
        messages = self._build_messages(summaries, period_label, date_str)

        async def _send_all() -> None:
            bot = telegram.Bot(token=self._token)
            for msg in messages:
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=msg,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )

        try:
            asyncio.run(_send_all())
            logger.info("Telegram: sent %d message(s) to chat %s", len(messages), self._chat_id)
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def _build_messages(
        self,
        summaries: list[DigestStory],
        period_label: str,
        date_str: str,
    ) -> list[str]:
        header = f"<b>NewsBot {period_label} Briefing</b> \u2014 {date_str}\n"
        grouped: dict[str, list[DigestStory]] = {t: [] for t in TOPIC_ORDER}
        for s in summaries:
            topic = s.topic or "current_events"
            bucket = topic if topic in grouped else "current_events"
            grouped[bucket].append(s)

        chunks: list[str] = []
        current = header
        for topic in TOPIC_ORDER:
            stories = grouped[topic]
            if not stories:
                continue
            emoji = TOPIC_EMOJI.get(topic, "")
            section_header = f"\n<b>{emoji} {topic.replace('_', ' ').title()}</b>\n"
            for s in stories:
                source_note = f"({s.source_count} source{'s' if s.source_count > 1 else ''})"
                entry = f"\n\u2022 <b>{s.headline}</b> {source_note}\n{s.summary}\n"
                if len(current) + len(section_header) + len(entry) > TELEGRAM_MAX_CHARS:
                    chunks.append(current)
                    current = entry
                else:
                    if section_header not in current:
                        current += section_header
                    current += entry

        if current.strip():
            chunks.append(current)
        return chunks
