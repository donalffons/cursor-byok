import type { PluginContext } from "cursor-byok:plugin";
import type { ResourceMetric, ResourceState } from "cursor-byok:resource";
import { apiHeaders, object, text } from "./auth.ts";

const USAGE_URL = "https://api.anthropic.com/api/oauth/usage";
/** The usage endpoint rate-limits unknown clients aggressively; identify as Claude Code. */
const USAGE_USER_AGENT = "claude-code/2.1.287";

/** One subscription limit as Claude Code's `/usage` reports it. */
export type QuotaLimit = {
  /** `session` (5-hour window), `weekly_all`, or `weekly_scoped`. */
  kind: string;
  /** Model name for a scoped weekly limit; null for limits covering every model. */
  model: string | null;
  usedPercent: number;
  resetAtMs: number | null;
};

export type AccountQuota = {
  limits: QuotaLimit[];
  updatedAtMs: number;
};

export class UsageError extends Error {
  constructor(readonly status: number) {
    super(`Claude usage lookup failed (HTTP ${status}).`);
  }
}

export async function fetchUsage(
  accessToken: string,
  context: PluginContext,
): Promise<AccountQuota> {
  context.signal.throwIfAborted();
  const response = await context.network.fetch(USAGE_URL, {
    headers: { ...apiHeaders(accessToken), "user-agent": USAGE_USER_AGENT },
  });
  if (response.status < 200 || response.status >= 300) throw new UsageError(response.status);
  let body: unknown;
  try {
    body = JSON.parse(response.body);
  } catch {
    throw new Error("Claude usage lookup returned invalid JSON.");
  }
  return parseUsage(body);
}

export function parseUsage(body: unknown, nowMs = Date.now()): AccountQuota {
  const limits: QuotaLimit[] = [];
  const entries = object(body)?.limits;
  for (const entry of Array.isArray(entries) ? entries : []) {
    const limit = object(entry);
    const kind = text(limit?.kind);
    const percent = limit?.percent;
    if (!kind || typeof percent !== "number" || !Number.isFinite(percent)) continue;
    const resetAtMs = Date.parse(text(limit?.resets_at) ?? "");
    limits.push({
      kind,
      model: text(object(object(limit?.scope)?.model)?.display_name),
      usedPercent: Math.max(0, Math.min(100, percent)),
      resetAtMs: Number.isFinite(resetAtMs) ? resetAtMs : null,
    });
  }
  return { limits, updatedAtMs: nowMs };
}

/** Only limits covering every model block the account; a scoped limit blocks one model. */
export function quotaState(quota: AccountQuota, nowMs = Date.now()): ResourceState {
  const resets = quota.limits
    .filter((limit) => limit.model === null && limit.usedPercent >= 100)
    .map((limit) => limit.resetAtMs ?? nowMs + 60 * 60 * 1000)
    .filter((resetAtMs) => resetAtMs > nowMs);
  return resets.length === 0 ? { status: "ready" } : {
    status: "cooling",
    retryAtMs: Math.max(...resets),
    message: "Claude subscription usage limit reached.",
  };
}

export function quotaMetrics(quota: AccountQuota | null | undefined): ResourceMetric[] {
  return (quota?.limits ?? []).map((limit) => ({
    id: limit.model ? `${limit.kind}:${limit.model}` : limit.kind,
    label: limitLabel(limit),
    unit: "percent",
    value: 100 - limit.usedPercent,
    ...(limit.resetAtMs !== null ? { resetAtMs: limit.resetAtMs } : {}),
  }));
}

function limitLabel(limit: QuotaLimit): ResourceMetric["label"] {
  if (limit.kind === "session") {
    return { "en-US": "5-hour window", "ru-RU": "5-часовое окно", "zh-CN": "5 小时窗口" };
  }
  if (limit.kind === "weekly_all") {
    return { "en-US": "Weekly, all models", "ru-RU": "Неделя, все модели", "zh-CN": "周额度(全部模型)" };
  }
  if (limit.model) {
    return {
      "en-US": `Weekly, ${limit.model}`,
      "ru-RU": `Неделя, ${limit.model}`,
      "zh-CN": `周额度(${limit.model})`,
    };
  }
  return limit.kind;
}
