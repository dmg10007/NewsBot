# NewsBot

Bias-aware, source-diverse news aggregation bot for US national, North Carolina state, and Lee County local news.

## Planned Pipeline

1. Ingestion — RSS, APIs, and HTML scrapers
2. Parsing — entity extraction and normalization
3. Clustering — group the same story across outlets
4. Bias filtering — lexicon, sentiment, framing, LLM escalation
5. Summarization — neutral digest generation
6. Delivery — twice-daily email digest, optional Telegram bot

## Current Status

Initial scaffold is in place for:
- configuration loading
- RSS fetching
- scraper framework
- deduplication

## Next Steps

- Add parsing and normalization
- Add story clustering
- Add bias analysis layer
- Add summarization and email delivery
