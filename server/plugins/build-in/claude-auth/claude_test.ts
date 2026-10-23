import type {
  JsonValue,
  NetworkEventStream,
  NetworkRequestInit,
  NetworkResponse,
  PluginContext,
} from "cursor-byok:plugin";
import type { ForwardEvent, LlmRequest, ModelEvent } from "cursor-byok:provider";
import type { ResourceSnapshot } from "cursor-byok:resource";
import {
  accountData,
  CLIENT_ID,
  OAuthError,
  parseTokens,
  refreshTokens,
  RESOURCE_TYPE,
  TOKEN_URL,
} from "./auth.ts";
import { CLAUDE_CODE_SYSTEM_PROMPT, forwardBody, forwardHeaders } from "./messages.ts";
import { claudeOAuth } from "./oauth.ts";
import { claudeAccounts } from "./resources.ts";
import { claudeModels } from "./models.ts";
import { claudeProvider, retryAtMs } from "./provider.ts";
import { parseUsage, quotaResetAtMs } from "./usage.ts";

function assert(condition: unknown, message = "assertion failed"): asserts condition {
  if (!condition) throw new Error(message);
}
function equal(actual: unknown, expected: unknown) {
  assert(
    JSON.stringify(actual) === JSON.stringify(expected),
    `expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`,
  );
}
async function rejects(fn: () => unknown, contains?: string) {
  try {
    await fn();
  } catch (error) {
    assert(error instanceof Error);
    if (contains) assert(error.message.includes(contains), error.message);
    return error;
  }
  throw new Error("expected rejection");
}
function response(body: unknown, status = 200): NetworkResponse {
  return { status, headers: {}, body: JSON.stringify(body) };
}
function context(handlers: {
  fetch?: (url: string, init?: NetworkRequestInit) => NetworkResponse | Promise<NetworkResponse>;
  stream?: (url: string, init?: NetworkRequestInit) => NetworkEventStream;
} = {}): PluginContext {
  return {
    signal: new AbortController().signal,
    network: {
      fetch: (url, init) => {
        assert(handlers.fetch, "unexpected fetch");
        return Promise.resolve(handlers.fetch(url, init));
      },
      stream: (url, init) => {
        assert(handlers.stream, "unexpected stream");
        return Promise.resolve(handlers.stream(url, init));
      },
    },
  };
}
let sequence = 0;
function resource(expired = false): ResourceSnapshot {
  const id = ++sequence;
  return {
    id: `resource-${id}`,
    type: RESOURCE_TYPE,
    key: "claude:org:account",
    state: { status: "ready" },
    privateData: {
      accessToken: `access-secret-${id}`,
      refreshToken: `refresh-secret-${id}`,
      expiresAtMs: Date.now() + (expired ? -1000 : 3_600_000),
      accountId: "account",
      organizationId: "org",
      displayName: "person@example.com",
    },
  };
}
function tokens(extra: Record<string, unknown> = {}) {
  return {
    access_token: "new-access-secret",
    refresh_token: "new-refresh-secret",
    expires_in: 3600,
    token_type: "Bearer",
    scope: "user:profile user:inference",
    account: { uuid: "account", email_address: "person@example.com" },
    organization: { uuid: "org" },
    ...extra,
  };
}
/** Trimmed real `/api/oauth/usage` response. */
function usage() {
  return {
    five_hour: { utilization: 18.0, resets_at: "2026-10-08T21:59:59.957393+00:00" },
    limits: [
      {
        kind: "session",
        group: "session",
        percent: 18,
        resets_at: "2026-10-08T21:59:59.957393+00:00",
        scope: null,
      },
      {
        kind: "weekly_all",
        group: "weekly",
        percent: 23,
        resets_at: "2026-10-15T06:59:59.957415+00:00",
        scope: null,
      },
      {
        kind: "weekly_scoped",
        group: "weekly",
        percent: 11,
        resets_at: "2026-10-15T06:59:59.957625+00:00",
        scope: { model: { id: null, display_name: "Opus" }, surface: null },
      },
    ],
  };
}
function request(): LlmRequest {
  return {
    instructions: "Help with code.",
    messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    tools: [],
    reasoning: { enabled: false, effort: null },
    latency: "standard",
    maxOutputTokens: 1024,
    cacheKey: "conversation",
  };
}
async function* lines(values: string[]) {
  for (const value of values) yield value;
}
function stream(status = 200, headers: Record<string, string> = {}): NetworkEventStream {
  const events = [
    { type: "message_start", message: { usage: { input_tokens: 5, output_tokens: 0 } } },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
    { type: "message_stop" },
  ];
  return {
    status,
    headers,
    lines: lines(
      status === 200
        ? events.flatMap((event) => [`data: ${JSON.stringify(event)}`, ""])
        : ["sensitive upstream error"],
    ),
  };
}
const model = { id: "claude-test", displayName: "Claude test" };

Deno.test("OAuth delegates PKCE and state to the host and uses a localhost callback", async () => {
  const begin = await claudeOAuth.begin({
    redirectUri: "http://127.0.0.1:43210/callback",
    state: "expected-state",
    codeChallenge: "host-challenge",
  }, context());
  const url = new URL(begin.authorizationUrl);
  equal(url.origin + url.pathname, "https://claude.ai/oauth/authorize");
  equal(url.searchParams.get("redirect_uri"), "http://localhost:43210/callback");
  equal(url.searchParams.get("client_id"), CLIENT_ID);
  equal(url.searchParams.get("code_challenge_method"), "S256");
  equal(url.searchParams.get("code_challenge"), "host-challenge");
  equal(url.searchParams.get("state"), "expected-state");
  assert(!url.searchParams.get("scope")?.includes("org:create_api_key"));
  const drafts = await claudeOAuth.complete(
    begin.session,
    {
      code: "authorization-code",
      codeVerifier: "host-verifier",
      redirectUri: "http://127.0.0.1:43210/callback",
    },
    context({
      fetch: (url, init) => {
        if (url === "https://api.anthropic.com/api/oauth/profile") {
          equal(init?.headers?.authorization, "Bearer new-access-secret");
          return response({
            account: { uuid: "account", email: "person@example.com" },
            organization: { uuid: "org" },
          });
        }
        equal(url, TOKEN_URL);
        equal(init?.headers?.["content-type"], "application/json");
        const body = JSON.parse(init?.body ?? "{}");
        equal(body.state, "expected-state");
        equal(body.code_verifier, "host-verifier");
        equal(body.redirect_uri, "http://localhost:43210/callback");
        return response(tokens({ account: undefined, organization: undefined }));
      },
    }),
  );
  equal(drafts[0].key, "claude:org:account");
  const view = claudeAccounts.present({ ...resource(), ...drafts[0] });
  assert(!JSON.stringify(view).includes("secret"));
  equal(view.displayName, "person@example.com");
});

Deno.test("OAuth rejects expired or changed sessions before network access", async () => {
  const input = {
    code: "code",
    codeVerifier: "verifier",
    redirectUri: "http://127.0.0.1:12345/callback",
  };
  await rejects(
    () =>
      claudeOAuth.complete(
        { state: "state", redirectUri: "http://localhost:12345/callback", expiresAtMs: 1 },
        input,
        context(),
      ),
    "expired",
  );
  await rejects(
    () =>
      claudeOAuth.complete(
        {
          state: "state",
          redirectUri: "http://localhost:9999/callback",
          expiresAtMs: Date.now() + 60_000,
        },
        input,
        context(),
      ),
    "changed",
  );
  await rejects(
    () =>
      claudeOAuth.begin({
        redirectUri: "https://example.com/callback",
        state: "state",
        codeChallenge: "challenge",
      }, context()),
    "local callback",
  );
});

Deno.test("token validation rejects missing identity, invalid expiry, and missing inference scope", async () => {
  for (
    const invalid of [{ account: null }, { expires_in: 0 }, { expires_in: Infinity }, {
      token_type: "Basic",
    }, { scope: "user:profile" }]
  ) {
    await rejects(() => parseTokens(tokens(invalid)));
  }
  const data = parseTokens(tokens(), undefined, 1000);
  equal(data.expiresAtMs, 3_601_000);
  const refreshed = parseTokens(
    tokens({ account: undefined, organization: undefined, refresh_token: undefined }),
    data,
    2000,
  );
  equal(refreshed.accountId, data.accountId);
  equal(refreshed.refreshToken, data.refreshToken);
  await rejects(
    () => parseTokens(tokens({ account: { uuid: "different" } }), data),
    "different account",
  );
});

Deno.test("OAuth errors never expose token response bodies", async () => {
  const error = await rejects(() =>
    refreshTokens(
      accountData(resource(true)),
      context({
        fetch: () =>
          response({
            error: "invalid_grant",
            error_description: "access-secret should never be shown",
          }, 400),
      }),
    )
  );
  assert(error instanceof OAuthError && error.invalidCredentials);
  assert(!error.message.includes("access-secret"));
});

Deno.test("refresh coalesces concurrent and stale snapshots without repeating token rotation", async () => {
  const data = accountData(resource(true));
  let calls = 0;
  const ctx = context({
    fetch: () => {
      calls++;
      return response(tokens());
    },
  });
  const [first, second] = await Promise.all([refreshTokens(data, ctx), refreshTokens(data, ctx)]);
  equal(first, second);
  equal(await refreshTokens(data, ctx), first);
  equal(calls, 1);
});

Deno.test("transient refresh failures allow a later retry", async () => {
  const data = accountData(resource(true));
  await rejects(() => refreshTokens(data, context({ fetch: () => response({}, 503) })));
  const refreshed = await refreshTokens(data, context({ fetch: () => response(tokens()) }));
  equal(refreshed.accessToken, "new-access-secret");
});

Deno.test("provider uses Bearer OAuth and emits a complete response", async () => {
  const events: ModelEvent[] = [];
  const selected = resource();
  const result = await claudeProvider.invoke(
    { model, resource: selected, request: request() },
    { emit: (event) => events.push(event) },
    context({
      stream: (url, init) => {
        equal(url, "https://api.anthropic.com/v1/messages");
        equal(init?.headers?.authorization, `Bearer ${accountData(selected).accessToken}`);
        equal(init?.headers?.["anthropic-beta"], "oauth-2025-04-20");
        assert(!init?.headers?.["x-api-key"]);
        return stream();
      },
    }),
  );
  equal(result.status, "completed");
  equal(events.at(-1), { type: "done", reason: "stop" });
});

Deno.test("expired token is refreshed before inference and persisted even on HTTP 429", async () => {
  let fetches = 0;
  const result = await claudeProvider.invoke(
    { model, resource: resource(true), request: request() },
    { emit: () => {} },
    context({
      fetch: (_url, init) => {
        fetches++;
        equal(JSON.parse(init?.body ?? "{}").grant_type, "refresh_token");
        return response(tokens());
      },
      stream: (_url, init) => {
        equal(init?.headers?.authorization, "Bearer new-access-secret");
        return stream(429, { "retry-after": "120" });
      },
    }),
  );
  equal(fetches, 1);
  equal(result.status, "resource-error");
  assert(result.patch?.state?.status === "cooling");
  assert(result.patch.state.retryAtMs! > Date.now() + 119_000);
  equal((result.patch.privateData as Record<string, JsonValue>).refreshToken, "new-refresh-secret");
});

Deno.test("HTTP 401 retries once after refresh and never loops", async () => {
  let requests = 0;
  let refreshes = 0;
  const result = await claudeProvider.invoke(
    { model, resource: resource(), request: request() },
    { emit: () => {} },
    context({
      fetch: () => {
        refreshes++;
        return response(tokens());
      },
      stream: () => {
        requests++;
        return stream(401);
      },
    }),
  );
  equal(requests, 2);
  equal(refreshes, 1);
  equal(result.status, "resource-error");
  equal(result.patch?.state?.status, "invalid");
  assert(result.patch?.privateData);
});

Deno.test("HTTP 403 reports denied access without a refresh or silent fallback", async () => {
  const result = await claudeProvider.invoke({ model, resource: resource(), request: request() }, {
    emit: () => {},
  }, context({ stream: () => stream(403) }));
  equal(result.status, "resource-error");
  assert(result.status === "resource-error" && result.message.includes("blocked"));
});

Deno.test("HTTP 503 refresh failure does not invalidate an account", async () => {
  const result = await claudeProvider.invoke(
    { model, resource: resource(true), request: request() },
    { emit: () => {} },
    context({
      fetch: () => response({ error: "upstream" }, 503),
    }),
  );
  equal(result.status, "request-error");
  assert(!result.patch);
});

Deno.test("stream failure after refresh retains replacement credentials without replaying partial output", async () => {
  let requests = 0;
  const result = await claudeProvider.invoke(
    { model, resource: resource(true), request: request() },
    { emit: () => {} },
    context({
      fetch: () => response(tokens()),
      stream: () => {
        requests++;
        return {
          status: 200,
          headers: {},
          lines: lines([
            'data: {"type":"message_start","message":{}}',
            "",
            'data: {"type":"error","error":{"type":"api_error","message":"access-secret"}}',
            "",
          ]),
        };
      },
    }),
  );
  equal(result.status, "request-error");
  assert(result.patch?.privateData);
  equal(requests, 1);
  assert(result.status !== "completed" && !result.message.includes("access-secret"));
});

Deno.test("retry-after supports seconds, HTTP dates, and malformed headers", () => {
  equal(retryAtMs({ "Retry-After": "30" }, 1000), 31_000);
  equal(retryAtMs({ "retry-after": "Thu, 01 Jan 1970 00:01:00 GMT" }, 1000), 60_000);
  equal(retryAtMs({ "retry-after": "invalid" }, 1000), 61_000);
});

Deno.test("models paginate account-visible catalog with capability metadata", async () => {
  let pages = 0;
  const models = await claudeModels.list(
    { resource: resource() },
    context({
      fetch: (url) => {
        pages++;
        const parsed = new URL(url);
        if (pages === 1) {
          assert(!parsed.searchParams.has("after_id"));
          return response({
            data: [{
              id: "claude-a",
              display_name: "Claude A",
              max_tokens: 64000,
              capabilities: {
                image_input: { supported: true },
                thinking: { supported: true, types: { adaptive: { supported: true } } },
                effort: { supported: true, high: { supported: true } },
              },
            }],
            has_more: true,
            last_id: "claude-a",
          });
        }
        equal(parsed.searchParams.get("after_id"), "claude-a");
        return response({
          data: [{ id: "claude-b", display_name: "Claude B" }],
          has_more: false,
          last_id: "claude-b",
        });
      },
    }),
  );
  equal(pages, 2);
  equal(models.map((model) => model.id), ["claude-a", "claude-b"]);
  equal(models[0].privateData, { thinking: "adaptive", efforts: ["high"] });
  equal(models[0].capabilities, { images: true });
});

Deno.test("model sync refuses to rotate unpersistable credentials or fabricate a catalog", async () => {
  await rejects(
    () => claudeModels.list({ resource: resource(true) }, context()),
    "Refresh the account",
  );
  await rejects(
    () => claudeModels.list({ resource: resource() }, context({ fetch: () => response({}, 403) })),
    "403",
  );
  await rejects(
    () =>
      claudeModels.list(
        { resource: resource() },
        context({ fetch: () => response({ data: [], has_more: true, last_id: "same" }) }),
      ),
    "pagination",
  );
});

Deno.test("refresh reads usage without rotating fresh tokens", async () => {
  const urls: string[] = [];
  const snapshot = resource();
  const refreshed = await claudeAccounts.refresh!(
    snapshot,
    context({
      fetch: (url, init) => {
        urls.push(url);
        equal(init?.headers?.["user-agent"]?.startsWith("claude-code/"), true);
        return response(usage());
      },
    }),
  );
  equal(urls, [
    "https://api.anthropic.com/api/oauth/usage",
    "https://api.anthropic.com/api/oauth/organizations/org/prepaid/credits",
  ]);
  equal(refreshed.state, { status: "ready" });
  const data = refreshed.privateData as Record<string, JsonValue>;
  equal(data.accessToken, (snapshot.privateData as Record<string, JsonValue>).accessToken);
  const view = claudeAccounts.present({ ...snapshot, privateData: refreshed.privateData! });
  equal(view.metrics?.map((metric) => [metric.id, metric.value]), [
    ["session", 82],
    ["weekly_all", 77],
    ["weekly_scoped:Opus", 89],
  ]);
  equal(view.metrics?.[0].resetAtMs, Date.parse("2026-10-08T21:59:59.957393+00:00"));
});

Deno.test("refresh rotates expiring tokens and keeps them when usage fails", async () => {
  const fetches: string[] = [];
  const refreshed = await claudeAccounts.refresh!(
    resource(true),
    context({
      fetch: (url) => {
        fetches.push(url);
        return url === TOKEN_URL ? response(tokens()) : response({}, 429);
      },
    }),
  );
  equal(fetches, [TOKEN_URL, "https://api.anthropic.com/api/oauth/usage"]);
  equal((refreshed.privateData as Record<string, JsonValue>).refreshToken, "new-refresh-secret");
  equal(refreshed.state, undefined);
});

Deno.test("refresh retries usage once after a rejected access token", async () => {
  let usageCalls = 0;
  const refreshed = await claudeAccounts.refresh!(
    resource(),
    context({
      fetch: (url) => {
        if (url === TOKEN_URL) return response(tokens());
        if (url.endsWith("/prepaid/credits")) return response(credits());
        usageCalls++;
        return usageCalls === 1 ? response({}, 401) : response(usage());
      },
    }),
  );
  equal(usageCalls, 2);
  equal(refreshed.state, { status: "ready" });
  equal((refreshed.privateData as Record<string, JsonValue>).accessToken, "new-access-secret");
});

Deno.test("refresh cools an account whose all-model limit is exhausted", async () => {
  const resetsAt = new Date(Date.now() + 3_600_000).toISOString();
  const refreshed = await claudeAccounts.refresh!(
    resource(),
    context({
      fetch: () =>
        response({
          limits: [
            { kind: "session", percent: 100, resets_at: resetsAt, scope: null },
            {
              kind: "weekly_scoped",
              percent: 100,
              resets_at: new Date(Date.now() + 9_000_000).toISOString(),
              scope: { model: { display_name: "Opus" } },
            },
          ],
        }),
    }),
  );
  equal(refreshed.state?.status, "cooling");
  equal(refreshed.state?.status === "cooling" && refreshed.state.retryAtMs, Date.parse(resetsAt));
  equal(refreshed.quotaResetAtMs, null);
});

Deno.test("refresh reports the weekly reset so the host spends expiring quota first", async () => {
  const resetsAt = new Date(Date.now() + 86_400_000).toISOString();
  const refreshed = await claudeAccounts.refresh!(
    resource(),
    context({
      fetch: () =>
        response({
          limits: [{ kind: "weekly_all", percent: 40, resets_at: resetsAt, scope: null }],
        }),
    }),
  );
  equal(refreshed.quotaResetAtMs, Date.parse(resetsAt));
});

Deno.test("quota reset is the weekly all-models reset while the plan has room", () => {
  const quota = parseUsage(usage(), 0);
  const weekly = Date.parse("2026-10-15T06:59:59.957415+00:00");
  equal(quotaResetAtMs(quota, weekly - 1), weekly);
  // A passed reset is unknown until the next refresh reads the following week.
  equal(quotaResetAtMs(quota, weekly), null);
  // With an all-models limit used up, only paid extra usage would serve the account.
  const session = { ...quota.limits[0], usedPercent: 100 };
  const exhausted = { ...quota, limits: [session, ...quota.limits.slice(1)] };
  equal(quotaResetAtMs(exhausted, weekly - 1), null);
});

/** Trimmed real `/api/oauth/organizations/:org/prepaid/credits` response. */
function credits(amount = 11063) {
  return {
    amount,
    currency: "USD",
    balance: { money: null, credits: { amount_minor: amount, exponent: 2 } },
    auto_reload_settings: { enabled: true },
  };
}
function extraUsageContext(extraUsage: Record<string, unknown>, balance: NetworkResponse) {
  const resetsAt = new Date(Date.now() + 3_600_000).toISOString();
  return context({
    fetch: (url) =>
      url.endsWith("/prepaid/credits") ? balance : response({
        limits: [{ kind: "weekly_all", percent: 100, resets_at: resetsAt, scope: null }],
        extra_usage: extraUsage,
      }),
  });
}

Deno.test("refresh reports a capped extra usage month and the credit balance", async () => {
  const snapshot = resource();
  const refreshed = await claudeAccounts.refresh!(
    snapshot,
    extraUsageContext({
      is_enabled: false,
      monthly_limit: 4000,
      used_credits: 4264.0,
      currency: "USD",
      decimal_places: 2,
      disabled_reason: "org_level_disabled_until",
      spend_limit_reached: true,
    }, response(credits())),
  );
  equal(refreshed.state?.status, "cooling");
  const view = claudeAccounts.present({ ...snapshot, privateData: refreshed.privateData! });
  equal(
    view.metrics?.map((metric) => [
      metric.id,
      metric.unit,
      metric.value,
      metric.currency,
      metric.limit,
    ]),
    [
      ["weekly_all", "percent", 0, undefined, undefined],
      ["extra_usage", "currency", 42.64, "USD", 40],
      ["credits", "currency", 110.63, "USD", undefined],
    ],
  );
});

Deno.test("enabled extra usage keeps an exhausted plan ready", async () => {
  const snapshot = resource();
  const capped = await claudeAccounts.refresh!(
    snapshot,
    extraUsageContext(
      { is_enabled: true, monthly_limit: 15000, used_credits: 4264, currency: "USD" },
      response(credits()),
    ),
  );
  equal(capped.state, { status: "ready" });
  const view = claudeAccounts.present({ ...snapshot, privateData: capped.privateData! });
  const extra = view.metrics?.find((metric) => metric.id === "extra_usage");
  equal([extra?.value, extra?.limit], [42.64, 150]);

  const uncapped = await claudeAccounts.refresh!(
    snapshot,
    extraUsageContext(
      { is_enabled: true, monthly_limit: null, used_credits: 4264, currency: "USD" },
      response(credits()),
    ),
  );
  const spent = claudeAccounts.present({ ...snapshot, privateData: uncapped.privateData! })
    .metrics?.find((metric) => metric.id === "extra_usage");
  equal([spent?.value, spent?.limit], [42.64, undefined]);
});

Deno.test("refresh keeps the last credit balance when its lookup fails", async () => {
  const first = resource();
  const before = await claudeAccounts.refresh!(
    first,
    extraUsageContext({ is_enabled: true, currency: "USD" }, response(credits(500))),
  );
  const after = await claudeAccounts.refresh!(
    { ...first, privateData: before.privateData! },
    extraUsageContext({ is_enabled: true, currency: "USD" }, response({}, 403)),
  );
  const view = claudeAccounts.present({ ...first, privateData: after.privateData! });
  equal(view.metrics?.find((metric) => metric.id === "credits")?.value, 5);
});

Deno.test("token rotation keeps the last usage reading", () => {
  const data = { ...accountData(resource()), quota: parseUsage(usage(), 1) };
  equal(parseTokens(tokens(), data).quota, data.quota);
});

Deno.test("refresh marks only invalid grants invalid", async () => {
  const invalid = await claudeAccounts.refresh!(
    resource(true),
    context({ fetch: () => response({ error: "invalid_grant" }, 400) }),
  );
  equal(invalid.state?.status, "invalid");
  await rejects(() =>
    claudeAccounts.refresh!(resource(), context({ fetch: () => response({}, 503) }))
  );
});

Deno.test("SSE authentication errors do not trigger the HTTP refresh retry", async () => {
  let requests = 0;
  const result = await claudeProvider.invoke(
    { model, resource: resource(), request: request() },
    { emit: () => {} },
    context({
      stream: () => {
        requests++;
        return {
          status: 200,
          headers: {},
          lines: lines(['data: {"type":"error","error":{"type":"authentication_error"}}', ""]),
        };
      },
    }),
  );
  equal(result.status, "resource-error");
  equal(result.patch?.state?.status, "invalid");
  equal(requests, 1);
});

Deno.test("HTTP failures close the response iterator before completing", async () => {
  let closed = false;
  async function* failedBody() {
    try {
      yield "sensitive body";
    } finally {
      closed = true;
    }
  }
  await claudeProvider.invoke(
    { model, resource: resource(), request: request() },
    { emit: () => {} },
    context({
      stream: () => ({ status: 503, headers: {}, lines: failedBody() }),
    }),
  );
  assert(closed, "failed HTTP response was left open");
});

Deno.test("cancelled invocation makes no network calls", async () => {
  const ctx = context();
  ctx.signal = AbortSignal.abort();
  const result = await claudeProvider.invoke(
    { model, resource: resource(true), request: request() },
    { emit: () => {} },
    ctx,
  );
  equal(result.status, "request-error");
  assert(!result.patch);
});

function forwardInput(selected: ResourceSnapshot, body: Record<string, JsonValue> = {}) {
  return {
    model,
    resource: selected,
    body: {
      model: "plugin:dev.cursorbyok.examples.claude-auth:claude/claude-test",
      max_tokens: 100,
      stream: true,
      system: [{ type: "text", text: "Be terse.", cache_control: { type: "ephemeral" } }],
      context_management: {
        edits: [{ type: "compact_20260112", trigger: { type: "input_tokens", value: 50000 } }],
      },
      messages: [
        { role: "user", content: "Hi" },
        { role: "assistant", content: [{ type: "compaction", content: "Summary." }] },
        { role: "user", content: "Continue" },
      ],
      ...body,
    },
    headers: { "anthropic-version": "2023-06-01", "anthropic-beta": "compact-2026-01-12" },
  };
}

Deno.test("Messages forwarding keeps the client's request and relays every upstream line", async () => {
  const relayed: ForwardEvent[] = [];
  const selected = resource();
  let sent: Record<string, JsonValue> = {};
  const upstream = ["event: message_start", 'data: {"type":"message_start"}', ""];
  const result = await claudeProvider.forwardMessages!(
    forwardInput(selected),
    { emit: (event) => relayed.push(event) },
    context({
      stream: (url, init) => {
        equal(url, "https://api.anthropic.com/v1/messages");
        equal(init?.headers?.authorization, `Bearer ${accountData(selected).accessToken}`);
        equal(init?.headers?.["anthropic-beta"], "oauth-2025-04-20,compact-2026-01-12");
        equal(init?.headers?.accept, "text/event-stream");
        sent = JSON.parse(init?.body ?? "{}");
        return { status: 200, headers: { "request-id": "req_1" }, lines: lines(upstream) };
      },
    }),
  );
  equal(result.status, "completed");
  equal(sent.model, "claude-test");
  equal(sent.system, [
    { type: "text", text: CLAUDE_CODE_SYSTEM_PROMPT },
    { type: "text", text: "Be terse.", cache_control: { type: "ephemeral" } },
  ]);
  const original = forwardInput(selected).body;
  equal(sent.context_management, original.context_management);
  equal(sent.messages, original.messages);
  equal(relayed, [
    { type: "response-start", status: 200, headers: { "request-id": "req_1" } },
    ...upstream.map((line) => ({ type: "response-line", line })),
  ]);
});

Deno.test("Messages forwarding adds the identity block once to text or absent system prompts", () => {
  const identity = { type: "text", text: CLAUDE_CODE_SYSTEM_PROMPT };
  equal(forwardBody(model, { system: "Be terse." }).system, [
    identity,
    { type: "text", text: "Be terse." },
  ]);
  equal(forwardBody(model, {}).system, [identity]);
  equal(forwardBody(model, { system: [identity] }).system, [identity]);
  equal(
    forwardHeaders({ "anthropic-version": "2023-06-01", "anthropic-beta": "oauth-2025-04-20" }, {
      "anthropic-beta": "oauth-2025-04-20, context-management-2025-06-27",
    }, false),
    {
      "anthropic-version": "2023-06-01",
      "anthropic-beta": "oauth-2025-04-20,context-management-2025-06-27",
      accept: "application/json",
      "content-type": "application/json",
    },
  );
});

Deno.test("Messages forwarding withholds a 401, refreshes once, and relays the retry", async () => {
  const relayed: ForwardEvent[] = [];
  let requests = 0;
  const result = await claudeProvider.forwardMessages!(
    forwardInput(resource()),
    { emit: (event) => relayed.push(event) },
    context({
      fetch: () => response(tokens()),
      stream: (_url, init) => {
        requests++;
        if (requests === 1) return stream(401);
        equal(init?.headers?.authorization, "Bearer new-access-secret");
        return { status: 200, headers: {}, lines: lines(["data: {}", ""]) };
      },
    }),
  );
  equal(requests, 2);
  equal(result.status, "completed");
  assert(result.patch?.privateData);
  equal(relayed.map((event) => event.type), ["response-start", "response-line", "response-line"]);
});

Deno.test("Messages forwarding relays a rate limit to the client and cools the account", async () => {
  const relayed: ForwardEvent[] = [];
  const error = '{"type":"error","error":{"type":"rate_limit_error","message":"Slow down"}}';
  const result = await claudeProvider.forwardMessages!(
    forwardInput(resource()),
    { emit: (event) => relayed.push(event) },
    context({
      stream: () => ({ status: 429, headers: { "retry-after": "30" }, lines: lines([error]) }),
    }),
  );
  equal(relayed, [
    { type: "response-start", status: 429, headers: { "retry-after": "30" } },
    { type: "response-line", line: error },
  ]);
  equal(result.status, "resource-error");
  assert(result.status === "resource-error" && result.patch.state?.status === "cooling");
});

Deno.test("Messages forwarding relays a rejection of the client's request without touching the account", async () => {
  const relayed: ForwardEvent[] = [];
  const error = '{"type":"error","error":{"type":"permission_error","message":"Beta not allowed"}}';
  const result = await claudeProvider.forwardMessages!(
    forwardInput(resource()),
    { emit: (event) => relayed.push(event) },
    context({ stream: () => ({ status: 403, headers: {}, lines: lines([error]) }) }),
  );
  equal(relayed.at(-1), { type: "response-line", line: error });
  equal(result, { status: "request-error", message: "Anthropic Messages 403" });
});
