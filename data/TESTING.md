# Test Data

**Category:** project-management tools (Asana, Trello, monday.com, ClickUp, Notion, Linear, Jira, Basecamp).

The corpus is 12 public articles, saved on 2026-11-04. Each one is available as a URL and as text, so you can test either way. Prices are as written by those sites on that date, and several sites earn affiliate fees.

The text copies are other people's articles, so keep them in a private repo.

## Files

| File | Use |
|---|---|
| `corpus/*.txt` | Plain text of each article |
| `samples/ingest_urls.json` | All 12 as URLs (the service fetches them) |
| `samples/ingest_text.json` | All 12 as text, with `source` set to the URL |
| `samples/ingest_mixed.json` | 1 URL plus 1 text, for a quick check |
| `samples/ingest_bad_url.json` | Domain that doesn't exist |
| `samples/ingest_blocked_url.json` | Real page that returns 403 to scripts |
| `samples/reingest_long.json`, `reingest_short.json` | Same document, 18 chunks then 1 |
| `samples/questions.json` | Test questions with expected results |

## Commands

Run these from the project root, with the service on port 8000. `/ask` needs a question in the body, so set this helper first:

```bash
ask() { curl -s -X POST localhost:8000/tenants/$1/ask -H 'Content-Type: application/json' -d "{\"question\": \"$2\"}" | python3 -m json.tool; }
ingest() { curl -s -X POST localhost:8000/tenants/$1/ingest -H 'Content-Type: application/json' -d @data/samples/$2 | python3 -m json.tool; }
```

### Ingest

```bash
ingest acme ingest_urls.json    # by URL
ingest acme ingest_text.json    # or by text
ingest acme ingest_mixed.json   # or a quick mix
```

Text gives `documents_stored: 12` and `chunks_stored: 156` at 400 tokens with 50 overlap. URLs can differ slightly if a page has changed since it was saved.

### Check there are no duplicates

| Test | Expected |
|---|---|
| Run `ingest_urls.json` twice | Pinecone count for `acme` doesn't grow |
| Run `ingest_urls.json`, then `ingest_text.json` | Count doesn't grow, because `source` gives the same `doc_id` |

### Check stale chunk cleanup

This uses its own tenant so it doesn't affect `acme`.

```bash
ingest shrinktest reingest_long.json    # 18 chunks
ingest shrinktest reingest_short.json   # 1 chunk
```

The Pinecone count for `shrinktest` should go from 18 to 1. If it stays at 18, cleanup isn't working.

### Ask

```bash
ask acme "What do reviewers say about Asana's pricing compared to Trello?"   # answer with citations
ask acme "What is the capital of France?"                                    # found_in_corpus false, retry in trace
ask globex "What do reviewers say about Asana's pricing compared to Trello?" # found_in_corpus false, no citations
```

More questions are in `samples/questions.json`:
- **normal:** should answer with citations. `must_contain` is the corpus text that backs the answer.
- **conflicting_sources:** sources disagree on Asana's free plan (2 or 15 users), so see how the answer handles it.
- **off_topic:** should return `found_in_corpus: false`, and the answer shouldn't include the `must_not_contain` words.
- **retry:** vague wording that may trigger the rewrite step. Check the trace.
- **tenant_isolation:** must return nothing for `globex`.

### Errors

```bash
ingest acme ingest_bad_url.json       # clear error, nothing stored
ingest acme ingest_blocked_url.json   # clear error, nothing stored
```
