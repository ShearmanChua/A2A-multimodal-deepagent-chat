# MCP Tool Authentication — Per-User Permission Design

## Problem

Multiple users chat with the agent concurrently. Each user has different access rights to Weaviate collections, Postgres tables, and SharePoint folders. Credentials for each user must not be stored in the agent or MCP tool containers.

## Recommended Architecture: JWT Permission Claims via MCP HTTP Headers

The core idea is to have an auth service issue short-lived JWTs containing permission manifests, and flow those tokens from the frontend through to the MCP server as HTTP headers. Downstream data stores use **service accounts** (env vars on the MCP container), but the JWT controls what each user is allowed to see.

### Data Flow

```
Frontend
  → A2A message metadata: { auth_token: "eyJ..." }
  → AgentExecutor: extracts token from context.metadata
  → Agent: creates per-session MCP client with Authorization header
  → MCP server: validates JWT, extracts claims, filters tool results
  → Weaviate / Postgres / SharePoint (service accounts, permission-filtered)
```

---

## Components

### 1. Auth Service — JWT with Permission Claims

Issue tokens that act as "permission manifests" signed by your own service. The JWT carries which resources the user is allowed to access — no actual upstream credentials are embedded.

```json
{
  "sub": "alice",
  "exp": 1716000000,
  "weaviate_collections": ["ProjectA_Docs", "SharedKnowledge"],
  "postgres_tables": ["reports", "metrics"],
  "sharepoint_folders": ["/Marketing/", "/Finance/Public/"]
}
```

The MCP container only needs:
- Service-account env vars for Weaviate / Postgres / SharePoint
- `JWT_SECRET_KEY` env var to validate incoming tokens

---

### 2. Frontend → A2A Metadata

Attach the JWT to every A2A request in `MessageSendParams.metadata`:

```json
{ "auth_token": "<jwt>" }
```

Refresh the token before it expires; on 401 responses from the MCP server, re-authenticate and retry.

---

### 3. AgentExecutor — Extract Token

In `agent_executor.py`, `params_meta` already reads `system_prompt` from `context.metadata`. Extract `auth_token` the same way and pass it into `agent.stream()`.

```python
# agent_executor.py — _extract_parts()
auth_token = params_meta.get("auth_token", "")
```

```python
# agent_executor.py — execute()
async for item in self.agent.stream(
    query=parts.query,
    context_id=task.context_id,
    auth_token=auth_token,   # new parameter
    ...
):
```

---

### 4. Agent — Per-Session MCP Client Cache

The current `_ensure_initialized()` creates a shared singleton MCP client — this must change. Replace it with a **per-session agent cache** keyed by `context_id`. Each session gets its own `MultiServerMCPClient` with the user's JWT in the `Authorization` header.

```python
# agent.py
import time

_session_cache: dict[str, tuple[Any, float]] = {}  # context_id → (agent, last_used)
_SESSION_TTL = 3600  # seconds


class MultimodalAgent:

    async def _get_or_create_agent(self, context_id: str, auth_token: str):
        now = time.time()

        # Evict stale sessions
        expired = [k for k, (_, t) in _session_cache.items() if now - t > _SESSION_TTL]
        for k in expired:
            del _session_cache[k]

        if context_id in _session_cache:
            agent, _ = _session_cache[context_id]
            _session_cache[context_id] = (agent, now)
            return agent

        # Build a new MCP client carrying the user's token
        mcp_client = MultiServerMCPClient({
            "mcp": {
                "transport": "http",
                "url": f"{MCP_SERVER_URL}/mcp",
                "headers": {"Authorization": f"Bearer {auth_token}"},
            }
        })
        tools = wrap_tools_with_error_handling(await mcp_client.get_tools())
        agent = create_deep_agent(model=self.model, tools=tools, ...)

        _session_cache[context_id] = (agent, now)
        return agent

    async def stream(self, ..., auth_token: str = "", ...):
        agent = await self._get_or_create_agent(context_id, auth_token)
        async for chunk in agent.astream(...):
            ...
```

The LangGraph graph structure is the same for all users — only the baked-in MCP client (and therefore the `Authorization` header) differs per session.

---

### 5. MCP Server — JWT Validation + Permission Filtering

FastMCP's `Context` object exposes the raw Starlette HTTP request. Tool functions accept an optional `ctx: Context` parameter. A shared `_get_user_claims()` helper validates the JWT and returns the permission claims.

```python
# server.py
import os
import jwt  # PyJWT
from fastmcp import Context

_JWT_SECRET = os.environ["JWT_SECRET_KEY"]
_JWT_ALGORITHM = "HS256"


def _get_user_claims(ctx: Context) -> dict:
    """Validate the bearer token and return its claims."""
    auth = ctx.request_context.request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if not token:
        raise PermissionError("Missing auth token")
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise PermissionError("Auth token has expired")
    except jwt.InvalidTokenError as exc:
        raise PermissionError(f"Invalid auth token: {exc}")


@mcp.tool(name="list_weaviate_collections")
def list_weaviate_collections(ctx: Context) -> list[dict]:
    claims = _get_user_claims(ctx)
    allowed = set(claims.get("weaviate_collections", []))
    results = weaviate_list_collections()
    return [r for r in results if r["name"] in allowed]


@mcp.tool(name="get_weaviate_collection_schema")
def get_weaviate_collection_schema(collection_name: str, ctx: Context) -> dict:
    claims = _get_user_claims(ctx)
    allowed = set(claims.get("weaviate_collections", []))
    if collection_name not in allowed:
        raise PermissionError(f"Access to collection '{collection_name}' denied")
    return weaviate_get_collection_schema(collection_name)


@mcp.tool(name="query_weaviate")
def query_weaviate(
    collection_name: str,
    query: str,
    ctx: Context,
    limit: int = 10,
    alpha: float = 0.5,
    properties: list[str] | None = None,
) -> list[dict]:
    claims = _get_user_claims(ctx)
    allowed = set(claims.get("weaviate_collections", []))
    if collection_name not in allowed:
        raise PermissionError(f"Access to collection '{collection_name}' denied")
    return weaviate_hybrid_query(collection_name, query, limit, alpha, properties)
```

Apply the same pattern to Postgres tools (check `postgres_tables` claim) and SharePoint tools (check `sharepoint_folders` claim).

---

## Decision Table

| Concern | How it's handled |
|---|---|
| Credentials not stored in containers | MCP uses service accounts via env vars; user permissions travel as signed JWTs |
| Multiple concurrent users | Per-session agent cache, each with its own MCP client + JWT |
| Token expiry / refresh | Frontend refreshes token; new `context_id` or session recreation picks up the new token |
| Unauthorised collection / table access | MCP server enforces claims on every tool call — access denied before any query runs |
| Secrets scope | Only `JWT_SECRET_KEY` needs to be shared between auth service and MCP container |

---

## Tradeoff: Per-Session Connections vs Single Client

The session-cache approach means **N active sessions = N MCP client connections**. For most enterprise deployments this is acceptable.

If connection overhead becomes a concern (hundreds of concurrent sessions), replace the session cache with a single shared MCP client and inject the token at the HTTP transport level using a custom `httpx.AsyncBaseTransport` that reads from a `contextvars.ContextVar` set before each agent invocation. This is more complex but eliminates the per-session connection cost.

---

## Implementation Checklist

- [ ] Auth service: issue JWTs with `weaviate_collections`, `postgres_tables`, `sharepoint_folders` claims
- [ ] Frontend: attach `auth_token` to every A2A `MessageSendParams.metadata`
- [ ] `agent_executor.py`: extract `auth_token` from `context.metadata`, pass to `agent.stream()`
- [ ] `agent.py`: replace singleton `_ensure_initialized()` with `_get_or_create_agent(context_id, auth_token)`
- [ ] `mcp_tools/server.py`: add `_get_user_claims(ctx)` helper and `ctx: Context` param to all tools
- [ ] MCP container: add `JWT_SECRET_KEY` env var; remove any user-credential env vars
- [ ] Add `PyJWT` to MCP tool dependencies
