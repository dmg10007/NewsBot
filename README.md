# NewsBot v0.0.2

Bias-aware, source-diverse news aggregation bot for US national, North Carolina state, and Lee County local news.

Delivers a configurable-schedule email digest with optional Telegram support.

## Pipeline

```
Sources → Ingest → Deduplicate → Parse/NLP → Cluster → Bias Enrichment → Bias Analysis → Summarize → Deliver
```

1. **Ingestion** — RSS feeds + HTML scrapers (national, NC state, Lee County). All sources fetched via `httpx` with enforced timeouts and configurable per-request rate limiting. `feedparser` never touches a URL directly.
2. **Deduplication** — Full 256-bit SHA-256 URL hash + semantic headline ANN similarity (`util.semantic_search`). Includes cross-source wire-syndication dedup to prevent AP/Reuters republications from appearing as false corroboration.
3. **Parsing** — spaCy entity extraction, VADER sentiment, keyword topic classification.
4. **Normalization** — canonical entity aliases across sources.
5. **Clustering** — sentence-transformers ANN batch search (`util.semantic_search`) with cosine similarity grouping; single-article fast path skips clustering entirely. Shared model registry ensures the embedding model is loaded once per process (~400 MB, not twice).
6. **Bias Enrichment** — domain-level ratings from AllSides + MBFC loaded at startup via async scraper; every article gets `bias_lean`, `factuality`, and a confidence score before any NLP runs.
7. **Bias Analysis** — lexicon scan → framing comparison → Perplexity Sonar (escalation, `sonar` model) → llama.cpp fallback. LLM prompt contains only anonymized article content — source names and lean labels are intentionally excluded to prevent identity-based reasoning.
8. **Summarization** — neutral digest paragraphs via Perplexity Sonar or local llama.cpp. Single-source stories skip LLM and use a headline + entity fallback. All HTTP clients released cleanly after each run.
9. **Delivery** — HTML email via Resend, optional Telegram bot.

## Setup

### 1. Install dependencies

```bash
# Use the lock file for reproducible installs
pip install pip-tools
pip-compile requirements.txt -o requirements.lock
pip install -r requirements.lock
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
python main.py test-ingest

# Run a single morning digest immediately
python main.py run --period morning

# Run a single afternoon digest immediately
python main.py run --period afternoon

# Run a single evening digest immediately
python main.py run --period evening

# Start the scheduler (runs indefinitely per settings.yaml cron)
python main.py schedule
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below. **Never commit `.env` to version control** — it is already listed in `.gitignore`.

> **Note:** `load_dotenv()` is called once at startup in `main.py`, before any other module is imported. It is not called anywhere else in the codebase. If you run pipeline modules directly (e.g. in tests), ensure your environment is loaded by your test runner.

### Required

| Variable | Description | Where to get it |
|---|---|---|
| `PPLX_API_KEY` | Perplexity Sonar API key — used for LLM-assisted bias analysis and summarization | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) |
| `RESEND_API_KEY` | Resend API key — used to send the HTML email digest | [resend.com](https://resend.com) — free tier is 3,000 emails/month |
| `NEWSBOT_EMAIL_FROM` | Sender address for digest emails (e.g. `newsbot@yourdomain.com`) | Must be a verified domain in your Resend account |
| `NEWSBOT_EMAIL_TO` | Recipient email address(es) — comma-separated for multiple (max 5) | Your own email |

### Local LLM

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_BASE_URL` | `http://localhost:8080` | Base URL of the llama.cpp HTTP server |
| `LLAMA_CPP_MODEL` | `llama3` | Model name passed in the API request body |

If the server is unreachable, the summarizer falls back to a headline + entity list.

### Optional — Telegram

Telegram delivery is disabled by default. To enable it, set `delivery.telegram.enabled: true` in `settings.yaml` and provide both variables below.

| Variable | Description | Where to get it |
|---|---|---|
| `NEWSBOT_TELEGRAM_TOKEN` | Bot token from BotFather | Message [@BotFather](https://t.me/botfather) on Telegram |
| `NEWSBOT_TELEGRAM_CHAT_ID` | Chat or channel ID to post digests to | Send a message to your bot, then call `getUpdates` on the Bot API |

### Optional — Brave Search

Enables context enrichment for high-importance clusters during summarization.

| Variable | Description | Where to get it |
|---|---|---|
| `BRAVE_SEARCH_API_KEY` | Brave Search API key | [brave.com/search/api](https://brave.com/search/api) |

### Optional — Anthropic

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

# Optional: Brave Search enrichment
BRAVE_SEARCH_API_KEY=

# Optional: Anthropic
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Configuration

### settings.yaml — key options

```yaml
ingestion:
  request_timeout_seconds: 15    # httpx timeout per source fetch
  request_delay_seconds: 1.0     # polite inter-request delay
  max_retries: 3
  retry_backoff_seconds: 2
  max_articles_per_source: 20
  user_agent: "NewsBot/0.0.2"

deduplication:
  headline_similarity_threshold: 0.85   # same-source dedup threshold
  wire_syndication_threshold: 0.95      # cross-source wire dedup threshold

bias:
  max_llm_calls_per_run: 50

delivery:
  email:
    bias_tag_colors:           # optional — customize without touching Python
      left:         "#1565c0"
      center-left:  "#1976d2"
      center:       "#388e3c"
      center-right: "#e64a19"
      right:        "#b71c1c"
      unknown:      "#757575"
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

Every `RawArticle` carries a `bias_metadata` field after ingestion:

```python
article.bias_metadata.bias_lean      # 'right'
article.bias_metadata.factuality     # 'mixed'
article.bias_metadata.confidence     # 0.67
article.bias_metadata.allsides_bias  # 'right'
article.bias_metadata.mbfc_bias      # 'center-right'
article.bias_metadata.notes          # ['minor disagreement: ...']
```

### Layer 2 — Article-Level Analysis

Four-stage escalation chain run on each story cluster:

1. **Loaded word lexicon scan** — always runs; flags emotionally charged language
2. **Sentiment variance** — always runs; compares VADER scores across sources covering the same story
3. **Entity omission + framing comparison** — runs on escalated clusters; detects what facts different sources omit
4. **LLM analysis** — Perplexity Sonar (`sonar` model) → llama.cpp fallback; capped per run via `bias.max_llm_calls_per_run` in `settings.yaml`

**Prompt design principle:** Source names and bias-lean labels are intentionally excluded from the LLM prompt. The model sees only anonymized `ARTICLE_1`, `ARTICLE_2` … labels with headlines and summaries. This prevents the LLM from reasoning about a source’s political identity rather than the actual content. The LLM is a **labeler, not a judge** — it identifies framing differences without declaring which source is correct.

## Delivery

- **Email**: HTML digest via [Resend](https://resend.com) (free tier: 3,000 emails/month)
- **Telegram**: Optional — set `delivery.telegram.enabled: true` in `settings.yaml` and add credentials to `.env`

### Failure Alerting

If a digest run fails with an unhandled exception, NewsBot will attempt to send an alert via Telegram first, then fall back to email. Operators are notified immediately without needing to tail logs.

## Running as a service (Linux)

```ini
# /etc/systemd/system/newsbot.service
[Unit]
Description=NewsBot News Digest Scheduler
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/NewsBot
ExecStart=/path/to/venv/bin/python main.py schedule
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable newsbot
sudo systemctl start newsbot
```

## Architecture Notes

### Import safety

All pipeline modules are imported at the top of `scheduler.py` (not deferred inside functions). `_validate_imports()` is called at scheduler startup to catch broken modules immediately, not at the first scheduled run.

### Configuration safety

`get_settings()` and `get_sources()` return `types.MappingProxyType` objects. The cached config is **read-only** — callers cannot accidentally mutate the shared instance. A process restart is required to pick up config file changes.

### Dependency management

All dependencies in `requirements.txt` are pinned to exact versions. Use `pip-tools` to generate `requirements.lock` for reproducible deploys:

```bash
pip install pip-tools
pip-compile requirements.txt -o requirements.lock
pip install -r requirements.lock
```

Never `pip install -r requirements.txt` directly in production.

## Changelog

### v0.0.2 — 2026-05-23 (Code Review Pass)

**Security**
- `config/loader.py` — Removed `load_dotenv()` from module import time. `main.py` is the sole entry point that loads `.env`. Fixes environment leakage into tests and double-loading.
- `bias/llm_analyzer.py` — Added `httpx` request hook that redacts the `Authorization: Bearer` header before it can appear in debug-level logs. Prevents accidental API key exposure.

**Reliability**
- `ingestion/fetcher.py` — `feedparser` previously used `urllib` with no socket timeout and could hang indefinitely. All RSS sources are now pre-fetched via `httpx` and raw content is passed to `feedparser`.
- `ingestion/fetcher.py` — Added configurable `request_delay_seconds` inter-request delay and enforced `User-Agent` on all outbound requests.
- `scheduler/scheduler.py` — All pipeline imports moved to module level. `_validate_imports()` called at startup to fail fast on any broken module.
- `summarizer/summarizer.py` — Added `close()` and `__enter__`/`__exit__` context manager. The scheduler now uses `with Summarizer() as s` to guarantee connection pool release.
- `scheduler/scheduler.py` — Added `_send_failure_alert()`: on any unhandled exception, operators are notified via Telegram → Email before the exception re-raises.

**Correctness**
- `ingestion/fetcher.py` — `url_hash` upgraded from 16-char (64-bit) to full 64-char SHA-256 hex digest. Eliminates birthday collision risk at scale.
- `ingestion/deduplicator.py` — Added cross-source wire-syndication dedup at `wire_syndication_threshold` (default 0.95). Prevents AP/Reuters republications from appearing as independent corroboration in clusters.
- `bias/llm_analyzer.py` — Source names and `bias_lean` labels removed from LLM prompt. Model now sees only anonymized article content, preventing identity-based reasoning.
- `config/loader.py` — `get_settings()` and `get_sources()` now return `types.MappingProxyType`. Shared cached config is read-only; caller mutations no longer corrupt the cache.

**Performance**
- `utils/model_registry.py` *(new)* — Shared `SentenceTransformer` model registry. `Deduplicator` and `StoryClusterer` now share one model instance (~400 MB saved per run).
- `ingestion/deduplicator.py` — Replaced O(n²) pairwise loop with `util.semantic_search()` ANN. ~125,000 individual calls for 500 articles reduced to one vectorized matrix operation.

**Code Quality**
- `ingestion/pipeline.py` *(new)* — Shared ingest pipeline helper `ingest_all_sources()`. The duplicated tier loop that existed in both `scheduler.py` and `main.py` is now in one place.
- `__init__.py` added to all packages (`ingestion`, `parsing`, `clustering`, `bias`, `summarizer`, `delivery`, `scheduler`, `monitoring`, `config`, `utils`) for reliable import resolution with pytest, mypy, and coverage.
- `requirements.txt` — All dependencies pinned to exact versions. `pip-tools` workflow documented.
- `delivery/email_renderer.py` — Bias tag colors moved from hardcoded Python to `settings.yaml` under `delivery.email.bias_tag_colors`. Palette is now configurable without touching Python.

---

### v0.0.1 — 2026-05-22
- **perf: async MBFC scraper** — `bias/source_ratings.py` and `bias/resolver.py` refactored to run MBFC domain lookups concurrently (`asyncio.Semaphore`, max 5 parallel). `BiasResolver.refresh_async()` runs AllSides + MBFC via `asyncio.gather`. Startup bias scraping: ~10s vs 90s+ previously. Sync wrappers kept for backwards compatibility.
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
