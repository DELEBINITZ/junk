# Infrastructure & deployment

The app is **cloud-agnostic** (containers + standard backends). It runs three ways:

## 1. Zero-infra (laptop / CI)
```bash
uvicorn app.main:app --reload
```
Deterministic LLM + in-memory stores. No GPU/keys. This is what tests + eval use.

## 2. Docker Compose (self-hosted single box)
```bash
cd infra
docker compose up                          # app + postgres + redis + qdrant
docker compose --profile rag up            # + TEI embeddings/reranker (CPU)
docker compose --profile gpu up            # + SGLang (Qwen) — needs NVIDIA GPUs
docker compose --profile observability up  # + Langfuse
# run migrations once Postgres is up:
docker compose exec app asi-migrate
```
Flip providers in `docker-compose.yml` env (`LLM_PROVIDER=sglang`,
`EMBEDDING_PROVIDER=tei`, `RERANK_ENABLED=true`) as each backend comes online.

## 3. Kubernetes (production, multi-tenant)
The image (`infra/Dockerfile`) is a standard stateless service — deploy as a
Deployment behind a Service/Ingress. Recommended topology (blueprint §12, §16):

| Tier | Notes |
|------|-------|
| **chat-service** (this app) | HPA on CPU/RPS; ≥2 replicas; stateless |
| **SGLang** lanes (fast/standard/deep) | GPU node pools; standard = 2×H100 ×N replicas; deep opt-in + quota-capped; KEDA on queue depth |
| **TEI** embeddings + reranker | GPU; separate Deployments |
| **Qdrant** | StatefulSet, sharded/replicated, quantization on |
| **Postgres 16** | HA (Patroni/CloudNativePG); RLS enforced; run `asi-migrate` as a Job |
| **Redis 7** | semantic cache, rate limits, revocation |
| **Neo4j** (Zep/Graphiti) | when KG is enabled (Stage 2); shard by tenant |
| **Langfuse** | traces; its own datastore |

Production checklist before launch (blueprint §16): measured capacity model,
load + soak + chaos tests green, per-tenant fairness + isolation tests green,
canary/blue-green with fast rollback, SLOs + alerts + on-call ready.

### Key production env (set real values)
- `JWT_SECRET` (32+ bytes), or `AUTH_PROVIDER=oidc` + `OIDC_*`
- `STORE_BACKEND=postgres`, `DATABASE_URL`
- `RETRIEVAL_BACKEND=qdrant`, `QDRANT_URL`
- `LLM_PROVIDER=sglang`, `EMBEDDING_PROVIDER=tei`, `RERANK_ENABLED=true`
- `AGENT_ENGINE=langgraph` (durable checkpointing → resumable approval gate)
- `TRACING_PROVIDER=langfuse` + keys; `SEED_DEMO_DATA=false`
- `CAP_*_ENABLED` per the modules this deployment serves

### Scaling notes
- **Inference dominates cost** — three lanes so most traffic runs cheap; deep lane opt-in + quota-capped.
- **Admission control** is built in (`ConcurrencyMiddleware`): bounded concurrency → graceful 503 instead of brownout.
- **Per-tenant fairness**: extend with per-org quotas/token budgets (orchestrator seam).
- **Isolation is non-negotiable**: Postgres RLS + Qdrant org filter + per-org KG namespace; isolation tests are a launch gate.
