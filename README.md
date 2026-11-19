# Retrieval Lens

A RAG demo where each visitor gets their own workspace. They add web pages or text, ask
questions, and watch the agent work step by step in a visualizer at the bottom of the page.

- **Search modes:** semantic search (Pinecone, cosine) is always on. Hybrid adds BM25 keyword
  search merged with reciprocal rank fusion. Rerank reorders candidates with
  `bge-reranker-v2-m3`.
- **Agent loop (LangGraph):** retrieve → grade each chunk → answer, or rewrite the question and
  retry once, or say it's not in the sources.
- **Visualizer:** the server streams each graph step as server-sent events, including the
  ranking after every search stage and the grade for each chunk. The page replays them
  with play, pause and step controls. Chunks slide between rankings so you can see what
  fusion and reranking change.

## Pages and endpoints

| Path | What it is |
|---|---|
| `/` | The public app: sign-up, sources, ask, visualizer |
| `/console` | The original developer test page (logs, evaluation, mode comparison) |
| `POST /users` | Create a user from name + email; the returned `id` is their tenant id |
| `POST /tenants/{id}/ingest` | Add URLs or text |
| `POST /tenants/{id}/ask/stream` | Ask, streaming one event per graph step |
| `POST /tenants/{id}/ask` | Ask, with a single JSON response |
| `GET /tenants/{id}/chunks` | Stored chunks |
| `DELETE /tenants/{id}/sources?source=…` | Remove one source |

## Run locally

```bash
cp .env.example .env      # add GOOGLE_API_KEY and PINECONE_API_KEY
make index                # creates the Pinecone index once
make run                  # http://127.0.0.1:8000
make test-all
```

Without `DATABASE_URL`, users are kept in memory and are lost when the server restarts.

## Deploy to Vercel

1. **Postgres for users.** In the Vercel project, open *Storage → Marketplace → Neon* and connect
   a database. That sets `DATABASE_URL`. The `demo_users` table is created on first use.
2. **Environment variables** (Project → Settings → Environment Variables):

   | Name | Value |
   |---|---|
   | `GOOGLE_API_KEY` | Gemini key |
   | `PINECONE_API_KEY` | Pinecone key (run `make index` once locally first) |
   | `DATABASE_URL` | Set by Neon; use the pooled connection string |
   | `REQUIRE_REGISTRATION` | `true`, so only ids made by `POST /users` work |
   | `REQUESTS_PER_MINUTE` | e.g. `10` |
   | `MAX_CHUNKS_PER_TENANT` | e.g. `300` |

3. **Deploy:**
   ```bash
   npm i -g vercel
   vercel link
   vercel deploy --prod
   ```

How it fits together: `api/index.py` exposes the FastAPI app, and `vercel.json` sends every path
to it, allows up to 300 s per request (the agent loop makes several Gemini calls), and bundles
the static page and sample data.

### Limits on serverless

- **No RAGAS evaluation.** `ragas` is in `requirements-eval.txt`, not `requirements.txt`, so the
  bundle stays small, and `/evaluate` returns 501 on Vercel. Run evaluation locally.
- **Best-effort rate limit.** The per-minute limit is counted in memory on each instance, so
  separate instances count separately. The chunk quota is enforced exactly, because it reads
  Pinecone.
- **Unreliable `/logs`.** Logs are kept in memory, so `/logs` on the console page can miss
  entries. The visualizer doesn't use it.
- **Open free tier.** Gemini's free tier rate-limits quickly once several people use the demo.
  Use a paid key for anything public.

### Safety

- **No internal fetches.** URL ingestion refuses hosts that resolve to private, loopback,
  link-local or metadata addresses, checked again on every redirect.
- **Untrusted text.** Ingested text is treated as untrusted in every prompt.
- **No real login.** There is no password: the workspace id stays in the browser. Don't let
  people add private documents.
