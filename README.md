# NewsBot

Bias-aware, source-diverse news aggregation bot for US national, North Carolina state, and Lee County local news.

Delivers a twice-daily (6 AM / 6 PM ET) email digest with optional Telegram support.

## Pipeline

```
Sources → Ingest → Deduplicate → Parse/NLP → Cluster → Bias Enrichment → Bias Analysis → Summarize → Deliver
```

1. **Ingestion** — RSS feeds + HTML scrapers (national, NC state, Lee County)
2. **Deduplication** — URL hash + semantic headline similarity
3. **Parsing** — spaCy entity extraction, VADER sentiment, keyword topic classification
4. **Normalization** — canonical entity aliases across sources
5. **Clustering** — sentence-transformers ANN batch search (`util.semantic_search`) with cosine similarity grouping; single-article fast path skips clustering entirely
6. **Bias Enrichment** — domain-level ratings from AllSides + MBFC loaded at startup via async scraper; every article gets `bias_lean`, `factuality`, and a confidence score before any NLP runs
7. **Bias Analysis** — lexicon scan → framing comparison → Perplexity Sonar (escalation, `sonar` model) → llama.cpp fallback
8. **Summarization** — neutral digest paragraphs via local llama.cpp; single-source stories skip LLM and use a headline + entity fallback directly
9. **Delivery** — HTML email via Resend, optional Telegram bot

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

### 3. Refresh bias ratings cache

On first run, build the source ratings cache from AllSides and Media Bias Fact Check:

```bash
python -m bias.refresh
```

This writes `config/bias_ratings_cache.json`. The scraper runs AllSides and MBFC **concurrently** (up to 5 parallel MBFC requests) and completes in ~10 seconds for the default 30-domain list. It only needs to re-run weekly — add it to cron:

```bash
# Every Monday at 4 AM
0 4 * * 1 /path/to/venv/bin/python -m bias.refresh
```

### 4. Start your local llama.cpp server

```bash
./llama-server -m your-model.gguf --port 8080
```

### 5. Run

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
| National | AP News (via Google News), Reuters (via Google News), NPR, PBS NewsHour, Axios (via Google News), The Hill, Fox News, Fox Business, CNN, WSJ (via Google News) |
| NC State | WRAL News, WRAL Politics, NC Policy Watch, Carolina Public Press, WCNC, Charlotte Observer (via Google News) |
| Lee County | Sanford Herald, The Rant NC |

> **Note:** Several outlets (AP News, Reuters, Axios, WSJ, Charlotte Observer) have no public RSS feed or have paywalled/deprecated feeds. These are proxied through Google News `site:` and `allinurl:` search feeds, which return the same articles without requiring a subscription.

## Bias & Factuality System

NewsBot uses a two-layer approach: **source-level ratings** loaded at startup, and **article-level analysis** run during the pipeline.

### Layer 1 — Source Ratings (AllSides + MBFC)

At startup (or after running `python -m bias.refresh`), the bot scrapes two independent third-party rating services:

| Service | What it provides | Methodology |
|---|---|---|
| [AllSides](https://www.allsides.com/media-bias/ratings) | Bias lean (Left → Right, 5-point scale) | Editorial review + blind surveys + community feedback |
| [Media Bias Fact Check](https://mediabiasfactcheck.com) | Bias lean + factuality reporting score | Analyst review of sourcing, headlines, and story selection |

Ratings from both services are **merged per domain** with a confidence score:

| Agreement | Confidence | Outcome |
|---|---|---|
| Both agree | `1.0` | High confidence |
| Off by one step | `0.67` | AllSides used, minor disagreement noted |
| Differ by 2+ steps | `0.33` | AllSides used, significant disagreement flagged |
| Only one source | `0.75` | Single-source rating |
| Neither has data | `0.50` | Hardcoded fallback table |

This mirrors the methodology Ground News uses internally, built from the same primary sources.

Every `RawArticle` carries a `bias_metadata` field after ingestion:

```python
article.bias_metadata.bias_lean      # 'right'
article.bias_metadata.factuality     # 'mixed'
article.bias_metadata.confidence     # 0.67
article.bias_metadata.allsides_bias  # 'right'
article.bias_metadata.mbfc_bias      # 'center-right'
article.bias_metadata.notes          # ['minor disagreement: ...']
```

The cache refreshes weekly via cron and is stored in `config/bias_ratings_cache.json` (git-ignored).

**Async scraper:** The MBFC scraper runs up to 5 concurrent domain lookups behind an `asyncio.Semaphore` with a 1.0s polite delay. AllSides (single request) runs concurrently in a thread via `asyncio.to_thread`. Combined startup time for the default 30-domain list is ~10 seconds, down from 90+ seconds in the original sequential implementation. Synchronous wrappers are kept for backwards compatibility with tests and CLI use.

### Layer 2 — Article-Level Analysis

Four-stage escalation chain run on each story cluster:

1. **Loaded word lexicon scan** — always runs; flags emotionally charged language
2. **Sentiment variance** — always runs; compares VADER scores across sources covering the same story
3. **Entity omission + framing comparison** — runs on escalated clusters; detects what facts different sources omit
4. **LLM analysis** — Perplexity Sonar (`sonar` model) → llama.cpp fallback; capped at 20 calls/run; article summaries truncated to 300 chars per prompt to reduce token usage by ~60%

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

## Changelog

### 2026-05-22
- **perf: async MBFC scraper** — `bias/source_ratings.py` and `bias/resolver.py` refactored to run MBFC domain lookups concurrently (`asyncio.Semaphore`, max 5 parallel). `BiasResolver.refresh_async()` runs AllSides + MBFC via `asyncio.gather`. Startup bias scraping: ~10s vs 90s+ previously. Sync wrappers kept for backwards compatibility.

### 2026-05-21 (patch batch)
- **fix: scheduler duplicate `load_dotenv`** — removed redundant `load_dotenv()` call and unused import from `scheduler.py`
- **fix: `bias/resolver.py` import order** — moved `import re` to module top level
- **fix: deprecated Perplexity model** — `sonar-reasoning` → `sonar` in `config/settings.yaml`
- **perf: token reduction ~60%** — article summaries truncated to 300 chars in `bias/llm_analyzer.py` LLM prompt
- **perf: single-source summarizer fast path** — `summarizer/summarizer.py` skips local LLM for single-source stories, uses headline + entity fallback directly
- **perf: clustering O(n²) → ANN** — `clustering/clusterer.py` replaced pairwise loop with `util.semantic_search()` batched ANN; single-article fast path added
- **fix: dead httpx.Client removed** — `ingestion/fetcher.py` cleaned up; cache scope documented
- **fix: `bias/framing.py` import** — `_HEDGE_WORDS` moved to module level
- **fix: `config/loader.py` lru_cache note** — restart requirement documented
- **fix: requirements.txt** — direct `torch` pin removed
