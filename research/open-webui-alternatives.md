# Alternatives to Open WebUI (search + agent support)

Researched 2026-10-11. Requirement: self-hosted chat UI over an OpenAI-compatible
endpoint, with (a) basic web search and (b) some form of agent/tool-calling support.

## TL;DR

| Rank | Pick | Why |
|---|---|---|
| 1 | **LibreChat** | Closest 1:1 replacement. MIT, native Agents + MCP, native web search with **SearXNG** support. Cost: MongoDB + Meilisearch, ~1.5 GB RAM. |
| 2 | **Onyx** (ex-Danswer) | Best if "search" is the real product. Native SearXNG web search (v2), agents + actions, MCP, code interpreter, OIDC/SAML. Heavier stack. |
| 3 | **AnythingLLM** | Lightest path to search + agent skills in one container. MIT, desktop or Docker multi-user. Agent skills: web search, scrape, charts, file I/O. |
| 4 | **LobeChat / LobeHub** | Design-first Next.js, MCP marketplace, agent presets, web search w/ citations. Weakest on hard enterprise controls. |
| 5 | **Diy: thin UI over pi SDK / RPC** | You already run pi with SearXNG + crawl4ai tools. A thin web frontend gives you *real* agent behaviour, not a chat-UI's toy agent loop. |

If your complaint is specifically "Open WebUI config is a nightmare" — that is the
dominant community complaint right now (r/LocalLLaMA, Feb 2026, 50↑/54 comments about
tools not showing up in the UI). LibreChat is what most of that thread lands on.

## Tier 1 — drop-in web UI replacements

### LibreChat — https://github.com/danny-avila/LibreChat
- **Search**: built-in web search with pluggable providers — SearXNG, Tavily, Brave,
  Keyenable (keyless). Optional reranker (Jina/Cohere) or `rerankerType: "none"`.
  Private/loopback endpoints are blocked by default; add the host:port to
  `webSearch.allowedAddresses` (i.e. your `maceo.local:8888` needs one config line).
  Also has RAG API (chat with files), full-text message search (Meilisearch), user memory.
- **Agents**: "Agents" feature = OpenAI Assistants/GPTs equivalent but model-agnostic and
  no-code. Tools + Skills, **MCP servers**, agent plugins (v1.0 bundles deploy skills +
  MCP servers), subagents, scheduled chats, agents API (beta), code interpreter,
  generative-UI artifacts.
- **Ops**: MongoDB + Meilisearch (+ a vector DB if you want memory/agent RAG). ~1.5 GB idle.
  Multi-user: local, OAuth, OIDC, **LDAP** + per-user model permissions — the only one of
  the big three that ships that without middleware.
- **License**: MIT. Note: docs banner currently says "LibreChat is joining ClickHouse" —
  check governance/fork status before committing long-term.
- **Watch out for**: heaviest install of the group; agents are the newest surface and
  release notes still show perf/security churn (v0.8.8, Sep 2026).

### Onyx — https://github.com/onyx-dot-app/onyx
- **Search**: this is its core competency — hybrid search, contextual retrieval,
  knowledge graph, 40+ connectors (Slack/Drive/Confluence/…), citations.
  Web search providers include Serper and **native SearXNG** (added in Onyx v2, Jan 2026).
- **Agents**: custom agents with instructions/knowledge/actions, Actions against
  connected apps, MCP (both as MCP server for other agents and to consume tools),
  code interpreter, web research agent.
- **Ops**: Postgres + index/search service + model container; OIDC + SAML in v2. Heavier
  than Open WebUI, lighter than Dify.
- **License**: MIT core + `ee/` under enterprise licence.
- **Best for**: knowledge/search-first deployments. Overkill if you just want chat + a
  search button.

### AnythingLLM — https://github.com/Mintplex-Labs/anything-llm
- **Search**: agent "skills" include web search (SearXNG supported; You.com is the hosted
  default) and website scraping. Local-first RAG with LanceDB default, pluggable
  Chroma/Milvus/Qdrant/etc.
- **Agents**: agent mode + custom agent skills, "Private Browser Tool" API, model router,
  embedded chat widget, multi-user via Docker (or single-user desktop app).
- **Ops**: single container, SQLite + LanceDB default. Closest to "one command and done".
- **License**: MIT (Pro tier for some team features).
- **Best for**: solo/small-team document + agent workspace; the lowest migration effort.

### LobeChat / LobeHub — https://github.com/lobehub/lobehub
- **Search**: web search with citations, image gen, voice, artifacts panel.
- **Agents**: agent presets/marketplace, MCP marketplace (large), skills, scheduled
  agent tasks, Slack/MCP relay via lobehub-gateway.
- **Ops**: Next.js + Postgres; ~400 MB idle RAM (lightest of the big three).
- **Best for**: individuals/small teams who value the UI. Auth is lighter (OAuth/Clerk/
  Logto) — weakest fit if you need LDAP and per-model ACLs.

## Tier 2 — if "agent" means workflows, not chat

- **Dify** — visual workflow builder + RAG + API layer, publishes a chat app UI. Source
  available (modified Apache 2.0). Right choice for multi-step pipelines, overkill for chat.
- **RAGFlow** — best-in-class document parsing + agent workflows; search/agent oriented,
  less of a general ChatGPT replacement.
- **n8n / OpenClaw-style personal agents** — automation-first; use only if the trigger is
  schedules/webhooks rather than a chat window.

## Tier 3 — desktop / minimal (no server to run)

- **Jan** (Apache 2.0) — clean open-source local desktop app; web search/agents limited.
- **Cherry Studio** — cross-platform desktop, good **MCP** client, RAG, agent workflows.
  Desktop only; no hosted multi-user.
- **LM Studio** — best model lab/server, thin chat layer.
- **NextChat / Chatbox / Big-AGI** — very light; web search via plugins (SearXNG plugin
  exists for NextChat), agent support minimal.

## Tier 4 — build the 200-line UI you actually want

You already have pi pointed at `mothership:8082` with SearXNG + crawl4ai as tools. Every
chat UI above re-implements a weaker version of that agent loop. Options:

- Thin web frontend over pi's SDK/RPC (`docs/sdk.md`, `docs/rpc.md`) — keeps one agent
  runtime, one tool set, session JSONL history. Search = your existing searxng skill.
- Vercel AI SDK chat template if you want a plain React chat app calling the
  OpenAI-compatible endpoint + a `searxng` tool.

Worth ~half a day if your requirements really are "basic search + agent"; you skip the
MongoDB/Postgres/Meilisearch fleet entirely.

## Fit against your environment

| Need | LibreChat | Onyx | AnythingLLM | LobeHub |
|---|---|---|---|---|
| OpenAI-compatible base URL (mothership:8082) | ✅ custom endpoints | ✅ (Ollama/LiteLLM/OpenAI-compat) | ✅ generic OpenAI provider | ✅ |
| SearXNG without an API key | ✅ (needs `allowedAddresses`) | ✅ native (v2) | ✅ (needs `format=json` enabled) | ⚠️ plugin/3rd-party |
| Agent w/ tool calling | ✅ Agents + MCP | ✅ Agents + Actions + MCP | ✅ Agent skills + MCP | ✅ MCP marketplace |
| Single-container ops | ⚠️ compose (3+ svc) | ❌ | ✅ | ⚠️ (+Postgres) |
| Multi-user + SSO | ✅ LDAP/OIDC | ✅ OIDC/SAML | ✅ (Docker) | ⚠️ |
| RAM idle (reported) | ~1.5 GB | >1 GB | ~600 MB | ~400 MB |

RAM figures from third-party comparisons (blog.elest.io, May 2026), not measured.

## Recommended experiment

1. `docker compose up` **LibreChat** side-by-side with Open WebUI for a week, same model
   endpoint, `SEARXNG` search provider + `webSearch.allowedAddresses` for maceo.local:8888.
2. In parallel, spend 2 hours on **AnythingLLM** — if it covers 90 % with one container,
   that is usually the right answer for a home network.
3. If both feel like fighting a framework, spend those hours on the pi-SDK web frontend
   instead.

## Sources

- https://docs.openwebui.com/alternatives/ (Open WebUI's own list)
- https://blog.elest.io/librechat-vs-openwebui-vs-lobe-chat-which-to-self-host-in-2026/
- https://www.switchmytool.com/blog/best-open-webui-alternatives
- https://www.truefoundry.com/blog/librechat-vs-open-webui
- https://neuralchainai.com/blog/onyx-vs-librechat-vs-open-webui-comparison/
- https://docs.onyx.app/welcome , https://docs.onyx.app/overview/core_features/web_search
- https://www.librechat.ai/docs/features/agents , /docs/features/web_search , changelog v0.8.8
- https://docs.anythingllm.com/ (agent skills, providers, MCP)
- https://github.com/lobehub/lobehub , https://aiidelist.com/blog/lobehub
- r/LocalLLaMA "Are there any alternatives to Open WebUI that don't have terrible UX?" (Feb 2026)
