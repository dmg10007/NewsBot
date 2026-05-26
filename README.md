# NewsBot

NewsBot builds a twice-daily national, state, and local news digest. It ingests configured RSS feeds and scraper sources, groups articles reporting the same story, compares reporting across sources, writes neutral summaries, links every source, and labels each source with its configured or resolved media-bias rating.

Default geography is US national, North Carolina, and Lee County/Sanford. The geography labels and matching keywords live in `config/settings.yaml`, so another state/local profile can be configured without rewriting the pipeline.

## Pipeline

```text
Collect articles
  -> normalize URLs and persist to SQLite
  -> classify geography
  -> cluster same-story coverage
  -> score and filter stories
  -> compare reporting across sources
  -> summarize neutrally
  -> render/send digest
  -> record run history
```

The main orchestrator is `pipeline.DigestPipeline`. Scheduler and CLI commands are thin wrappers around it.

## Main Concepts

- `domain.models.Source` is the typed source configuration.
- `domain.models.Article` is the normalized/persisted article record.
- `domain.models.StoryCluster` groups articles covering one news event.
- `domain.models.ReportingComparison` stores shared facts, source-specific claims, omissions, framing differences, and bias notes.
- `domain.models.DigestStory` is the only contract renderers consume.
- `domain.models.DigestRun` records one morning or evening run.

SQLite persistence is implemented in `storage.SQLiteStore` and defaults to `data/newsbot.sqlite`.

## Installation

NewsBot requires Python 3.11 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you plan to refresh live media-bias ratings with Playwright-backed scrapers, install the browser runtime:

```bash
python -m playwright install chromium
```

Create your local environment file:

```bash
copy .env.example .env
```

Then edit `.env` with at least the email variables if you want delivery enabled.

Initialize the SQLite database:

```bash
python main.py db migrate
```

Validate source configuration:

```bash
python main.py sources check
```

Run a non-delivery ingestion smoke test:

```bash
python main.py ingest --dry-run
```

## Startup

Run one digest immediately:

```bash
python main.py run --period morning
python main.py run --period evening
```

Start the blocking scheduler:

```bash
python main.py schedule
```

By default, scheduled runs use the cron expressions in `config/settings.yaml`:

```yaml
scheduler:
  schedule:
    morning: "0 6 * * *"
    evening: "0 18 * * *"
```

For a persistent deployment, run `python main.py schedule` under your normal process manager, such as Task Scheduler on Windows, systemd on Linux, or a container supervisor.

## Commands

```bash
python main.py db migrate
python main.py sources check
python main.py ingest --dry-run
python main.py run --period morning
python main.py run --period evening
python main.py schedule
```

`run` sends delivery if enabled in `config/settings.yaml`. `ingest --dry-run` collects and classifies articles without writing them to SQLite.

## Configuration

`config/settings.yaml` contains schedule, geography, storage, scoring, LLM, and delivery settings.

Important defaults:

```yaml
storage:
  sqlite_path: data/newsbot.sqlite

geography:
  profile: nc_lee_county
  labels:
    national: "National"
    state: "North Carolina"
    local: "Local - Lee County"

scheduler:
  schedule:
    morning: "0 6 * * *"
    evening: "0 18 * * *"
```

`config/sources.yaml` contains RSS and scraper sources. Each source should include `name`, `url`, `bias_lean`, `credibility`, and `topics`; scraper sources also need `scraper_class`.

## LLM Behavior

NewsBot keeps separate LLM clients:

- `ComparisonLLMClient` compares reporting across clustered articles.
- `SummaryLLMClient` writes neutral digest summaries.

Both clients use Perplexity when `PPLX_API_KEY` is set, fall back to a local OpenAI-compatible llama.cpp server when `LLAMA_CPP_BASE_URL` is set, and finally fall back to deterministic heuristic/extractive output. Prompts explicitly treat “bias-free” as minimizing loaded language and clearly labeling attribution, not as a guarantee of perfect objectivity.

## Delivery

Email is the primary delivery path. The renderer groups stories by geography and includes:

- headline and neutral summary,
- all source links,
- source bias badges,
- reporting-difference notes,
- single-source labeling when applicable.

Telegram remains optional and uses the same `DigestStory` objects in a compact format.

## Environment

Required for email delivery:

```env
RESEND_API_KEY=
NEWSBOT_EMAIL_FROM=
NEWSBOT_EMAIL_TO=
```

Optional:

```env
PPLX_API_KEY=
LLAMA_CPP_BASE_URL=http://localhost:8080
LLAMA_CPP_MODEL=llama3
NEWSBOT_TELEGRAM_TOKEN=
NEWSBOT_TELEGRAM_CHAT_ID=
```

If email delivery is enabled in `config/settings.yaml`, the `RESEND_API_KEY`, `NEWSBOT_EMAIL_FROM`, and `NEWSBOT_EMAIL_TO` variables must be present. If you only want to test ingestion and rendering, keep delivery disabled or use `python main.py ingest --dry-run`.

## Tests

```bash
pytest
```

The test suite covers URL normalization, source/geography classification, clustering compatibility, SQLite persistence, email rendering, source-bias resolver behavior, ingestion compatibility, and monitoring.
