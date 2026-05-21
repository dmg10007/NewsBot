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
# Edit .env and fill in your API keys
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
