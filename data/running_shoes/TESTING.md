# Running Shoes Test Data

**Focus:** Nike vs ASICS, with Brooks, HOKA and others in the wider articles.

The corpus is 12 public articles, saved on 2026-11-04. Each one is available as a URL and as text. Prices and specs are as written by those sites on that date, and model years vary across articles. Some sites earn affiliate fees.

The text copies are other people's articles, so keep them in a private repo.

Files match the project-tools dataset described in `data/TESTING.md`, with `corpus/` and `samples/` in this folder.

## Commands

Run these from the project root, with the service on port 8000.

```bash
ask() { curl -s -X POST localhost:8000/tenants/$1/ask -H 'Content-Type: application/json' -d "{\"question\": \"$2\"}" | python3 -m json.tool; }
ingest() { curl -s -X POST localhost:8000/tenants/$1/ingest -H 'Content-Type: application/json' -d @data/running_shoes/samples/$2 | python3 -m json.tool; }
```

Each dataset should go in its own tenant, otherwise their answers mix.

### Ingest

```bash
ingest acme ingest_urls.json    # by URL
ingest acme ingest_text.json    # or by text
ingest acme ingest_mixed.json   # or a quick mix
```

For the smallest test, use `ingest_simple.json`: two short notes, 1 chunk each, with facts taken from the articles. Put it in its own tenant, for example `ingest demo ingest_simple.json`.

Text gives `documents_stored: 12` and `chunks_stored: 101` at 400 tokens with 50 overlap. URLs can differ slightly if a page has changed since it was saved.

### Check there are no duplicates

| Test | Expected |
|---|---|
| Run `ingest_urls.json` twice | Pinecone count for `acme` doesn't grow |
| Run `ingest_urls.json`, then `ingest_text.json` | Count doesn't grow, because `source` gives the same `doc_id` |

### Check stale chunk cleanup

```bash
ingest shrinktest reingest_long.json    # 10 chunks
ingest shrinktest reingest_short.json   # 1 chunk
```

The Pinecone count for `shrinktest` should go from 10 to 1.

### Ask

```bash
ask acme "What do reviewers say about ASICS durability compared to Nike?"     # answer with citations
ask acme "What is the capital of France?"                                     # found_in_corpus false, retry in trace
ask acme "What is Nike's stock price today?"                                  # found_in_corpus false
ask globex "What do reviewers say about ASICS durability compared to Nike?"   # found_in_corpus false, no citations
```

The corpus includes a shoe called the Metaspeed Sky Paris, so the France question may retrieve those chunks. Grading must reject them.

More questions are in `samples/questions.json`, including one where sources disagree on Vaporfly and Metaspeed prices.

### Errors

```bash
ingest acme ingest_bad_url.json       # clear error, nothing stored
ingest acme ingest_blocked_url.json   # clear error, nothing stored
```
