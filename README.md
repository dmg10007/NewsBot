# NewsBot

Bias-aware, source-diverse news aggregation bot for US national, North Carolina state, and Lee County local news.

Delivers a twice-daily (6 AM / 6 PM ET) email digest with optional Telegram support.

## Pipeline

```
Sources → Ingest → Deduplicate → Parse/NLP → Cluster → Bias Analysis → Summarize → Deliver
```

1. **Ingestion** — RSS feeds + HTML scrapers (national, NC state, Lee County)
2. **Deduplication** — URL hash + semantic headline similarity
3. **Parsing** — spaCy entity extraction, VADER sentiment, keyword topic classification
4. **Normalization** — canonical entity aliases across sources
5. **Clustering** — sentence-transformers cosine similarity grouping
6. **Bias Analysis** — lexicon scan → framing comparison → Perplexity Sonar (escalation) → llama.cpp fallback
7. **Summarization** — neutral digest paragraphs via local llama.cpp
8. **Delivery** — HTML email via Resend, optional Telegram bot

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — see the Environment Variables section below
```

### 3. Start your local llama.cpp server

```bash
./llama-server -m your-model.gguf --port 8080
```

### 4. Run

```bash
# Test ingestion only (no delivery)
python main.py --test-ingest

# Run a single morning digest immediately
python main.py --period morning

# Run a single evening digest immediately
python main.py --period evening

# Start the scheduler (6 AM / 6 PM ET, runs indefinitely)
python main.py --schedule
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below. **Never commit `.env` to version control** — it is already listed in `.gitignore`.

### Required

These must be set before the bot will run.

| Variable | Description | Where to get it |
|---|---|---|
| `PPLX_API_KEY` | Perplexity Sonar API key — used for LLM-assisted bias analysis on escalated story clusters | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) |
| `RESEND_API_KEY` | Resend API key — used to send the HTML email digest | [resend.com](https://resend.com) — free tier is 3,000 emails/month |
| `NEWSBOT_EMAIL_FROM` | The sender address for digest emails (e.g. `newsbot@yourdomain.com`) | Must be a verified domain in your Resend account |
| `NEWSBOT_EMAIL_TO` | Recipient email address(es) — comma-separated for multiple (max 5) | Your own email |

### Local LLM

The bot uses a locally-running [llama.cpp](https://github.com/ggerganov/llama.cpp) server for summarization. These variables control how it connects.

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_BASE_URL` | `http://localhost:8080` | Base URL of the llama.cpp HTTP server |
| `LLAMA_CPP_MODEL` | `llama3` | Model name passed in the API request body |

Start the server before running the bot:

```bash
./llama-server -m your-model.gguf --port 8080
```

If the server is unreachable, the summarizer falls back to a headline + entity list instead of a full neutral paragraph.

### Optional — Telegram

Telegram delivery is disabled by default. To enable it, set `delivery.telegram.enabled: true` in `settings.yaml` and provide both variables below.

| Variable | Description | Where to get it |
|---|---|---|
| `NEWSBOT_TELEGRAM_TOKEN` | Bot token from BotFather | Message [@BotFather](https://t.me/botfather) on Telegram |
| `NEWSBOT_TELEGRAM_CHAT_ID` | Chat or channel ID to post digests to | Send a message to your bot, then call `getUpdates` on the Bot API |

### Optional — Anthropic

If you have an Anthropic API key, you can swap it in for higher-quality summarization by setting this variable and updating the summarizer config in `settings.yaml`.

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key — used as an alternative summarization provider |

### Full `.env` reference

```env
# Required
PPLX_API_KEY=your_perplexity_api_key_here
RESEND_API_KEY=your_resend_api_key_here
NEWSBOT_EMAIL_FROM=newsbot@yourdomain.com
NEWSBOT_EMAIL_TO=you@example.com

# Local LLM (llama.cpp)
LLAMA_CPP_BASE_URL=http://localhost:8080
LLAMA_CPP_MODEL=llama3

# Optional: Telegram
NEWSBOT_TELEGRAM_TOKEN=
NEWSBOT_TELEGRAM_CHAT_ID=

# Optional: Anthropic
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Sources

| Tier | Sources |
|---|---|
| National | AP News, Reuters (via Google News), NPR, PBS NewsHour, Axios (via Google News), The Hill, Fox News, CNN, WSJ (via Google News) |
| NC State | WRAL, NC Policy Watch, Carolina Public Press, WCNC, Charlotte Observer |
| Lee County | Sanford Herald, The Rant NC |

## Bias Detection

Four-stage escalation chain:
1. Loaded word lexicon scan (always runs)
2. Sentiment variance across sources (always runs)
3. Entity omission + framing comparison (on escalated clusters)
4. Perplexity Sonar `sonar-reasoning` LLM analysis → llama.cpp fallback (on escalated clusters, capped at 20 calls/run)

The LLM is a **labeler, not a judge** — it identifies framing differences without declaring which source is correct.

## Delivery

- **Email**: HTML digest via [Resend](https://resend.com) (free tier: 3,000 emails/month)
- **Telegram**: Optional — set `delivery.telegram.enabled: true` in `settings.yaml` and add credentials to `.env`

## Running as a service (Linux)

```ini
# /etc/systemd/system/newsbot.service
[Unit]
Description=NewsBot News Digest Scheduler
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/NewsBot
ExecStart=/path/to/venv/bin/python main.py --schedule
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable newsbot
sudo systemctl start newsbot
```
