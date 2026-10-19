# LibreChat on corsair — local setup notes

Compose project: `~/build/LibreChat` (git checkout, v0.8.8, `registry.librechat.ai/librechat-ai/librechat-dev:latest`)
UI: http://localhost:3080 · admin panel: http://localhost:3081 · Open WebUI untouched on :3000

## Files added (all untracked; `git pull` safe)

| File | Purpose |
|---|---|
| `librechat.yaml` | The only provider (Mothership) + SearXNG web search. Mounted to `/app/librechat.yaml`. |
| `docker-compose.override.yml` | Bind-mounts the above; injects `extra_hosts` for LAN names. |
| `lan-hosts.sh` | Resolves `mothership.local` / `maceo.local` on the host and writes `LAN_*_IP` into `.env`; recreates the api container only when an address changed. |
| cron `*/5 * * * *` (user `marc`) | Runs `lan-hosts.sh`, logs to `logs/lan-hosts.log`. No root needed. |

## .env changes
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_KEY`, `ASSISTANTS_API_KEY`: `user_provided` → blank. That blanking is what removes them from the model picker; there is no "disable provider" flag.
- `SEARXNG_INSTANCE_URL=http://maceo.local:8888`
- `SCHEDULES_SINGLE_PROCESS=true` — single api replica, otherwise the scheduler refuses to arm and schedule writes 503.
- `ADMIN_PANEL_SESSION_SECRET=<64 hex>` — was empty, panel crash-looped with "SESSION_SECRET must be at least 32 characters".
- `LAN_MOTHERSHIP_IP` / `LAN_MACEO_IP` — generated, do not hand-edit.
- Backup of the original: `.env.bak.<epoch>`

## librechat.yaml gotchas (learned the hard way)
- `version: 1.3.17` must match the image; mismatch = container exits and names the expected version.
- URL fields must be `${VAR}` references, not literals (`searxngInstanceUrl: "http://..."` warns and is rejected as unsourced).
- `allowedAddresses` is the opposite: literal `host:port` only, no `${}` interpolation, private IPs only, and it is required or the SSRF guard refuses `maceo.local`.
- Current schema keys for a custom endpoint: `name tenantId apiKey baseURL provider iconURL models{default,fetch} userIdQuery titleConvo titleTiming titleMethod titleModel titlePrompt titlePromptTemplate titleEndpoint modelDisplayLabel addParams dropParams customParams tokenConfig headers directEndpoint titleMessageRole`. `contextTokens` / `forcePrompt` / `addDefaults` are gone; unknown keys fail validation.

## Adding another model server
Append to `endpoints.custom` in `librechat.yaml`, then `docker compose restart api`:

```yaml
    - name: "Boxname"
      apiKey: "local"
      baseURL: "http://boxname.local:PORT/v1"
      models: { default: ["model-id"], fetch: true }
      modelDisplayLabel: "Label"
      titleConvo: true
      titleModel: current_model
      titleMessageRole: user
```
If that box needs web-search/URL-fetch access, add it to `lan-hosts.sh` `NAMES=()`.

## Why names need help at all
`.local` is mDNS: link-scoped multicast. Containers get Docker's unicast resolver (127.0.0.11 → 127.0.0.53 → router 192.168.1.1), which has no idea about these names, and the container image has `hosts: files dns` with no `libnss-mdns`, so glibc can't ask avahi either. Dead ends, in order tried:
1. `MulticastDNS=yes` in `/etc/systemd/resolved.conf` — no effect, NetworkManager overrides per link.
2. `nmcli connection modify netplan-eno1 ipv4.mdns yes` — this NM build has no `mdns` property.
3. Remaining paths to "real" mDNS in containers: swap netplan to `renderer: networkd` + a `.network.d` drop-in, or bake `libnss-mdns` + nsswitch into the image and bind-mount `/var/run/avahi-daemon/socket`. Both worse than `extra_hosts`.

Verified after `extra_hosts`: from inside the api container `GET http://mothership.local:8082/v1/models` → 200 `think/Qwen3.8-Flash-Next`; `GET http://maceo.local:8888/search?q=…&format=json` → 200, 20 results.

## rag_api and embeddings
`rag_api` uses `env_file: .env`, so blanking `OPENAI_API_KEY` there made it crash-loop on import ("Missing credentials"). Fixed with a container-scoped placeholder in `docker-compose.override.yml` (the api container never sees it, so the picker stays clean).
Still not functional for real work: llama.cpp on mothership refuses embeddings — `{"error":{"code":501,"message":"This server does not support embeddings. Start it with `--embeddings`"}}`. File upload / RAG needs a separate embeddings endpoint (e.g. llama.cpp with `--embeddings` serving `bge-m3` or `Qwen3-Embedding`), then point RAG at it. Until then RAG features return 501.

## Loose ends
- `docker compose` warns `UID`/`GID` unset (compose sets `user: "${UID}:${GID}"`), so the api container runs as root; add `UID=`/`GID=` (your `id -u`/`id -g`) to `.env` if you want files it writes owned by you.
- End-to-end chat + web-search smoke test needs an authenticated session; run it in the browser while tailing `docker compose logs -f api`.

## llama.cpp target
`mothership.local:8082`, model alias `think/Qwen3.8-Flash-Next`, `n_ctx` 98304, build b11546, 2 slots.
`/v1/models` `chat_template_caps`: `supports_tools`, `supports_parallel_tool_calls`, `supports_reasoning_effort`, `supports_preserve_reasoning` all true — so LibreChat agents/tool calling work.
