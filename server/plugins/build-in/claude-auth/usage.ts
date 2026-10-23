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

/** Pay-as-you-go spending past the plan limits, in major units of `currency`. */
export type ExtraUsage = {
  /** False while extra usage cannot cover requests: turned off or the monthly cap is reached. */
  enabled: boolean;
  currency: string;
  used: number;
  /** Monthly spending cap; null when uncapped. */
  monthlyLimit: number | null;
};

/** Prepaid usage credits that extra usage draws from. */
export type CreditBalance = {
  currency: string;
  amount: number;
};

export type AccountQuota = {
  limits: QuotaLimit[];
  /** Absent until extra usage was ever set up for the account. */
  extraUsage?: ExtraUsage;
  credits?: CreditBalance;
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

/** Prepaid balance as Claude Code's extra-usage screen reads it; null when unavailable. */
export async function fetchCredits(
  accessToken: string,
  organizationId: string,
  context: PluginContext,
): Promise<CreditBalance | null> {
  context.signal.throwIfAborted();
  try {
    const response = await context.network.fetch(
      `https://api.anthropic.com/api/oauth/organizations/${
        encodeURIComponent(organizationId)
      }/prepaid/credits`,
      {
        headers: {
          ...apiHeaders(accessToken),
          "user-agent": USAGE_USER_AGENT,
          "x-organization-uuid": organizationId,
        },
      },
    );
    if (response.status < 200 || response.status >= 300) return null;
    return parseCredits(JSON.parse(response.body));
  } catch {
    context.signal.throwIfAborted();
    return null;
  }
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
  const extraUsage = parseExtraUsage(object(body)?.extra_usage);
  return { limits, ...(extraUsage ? { extraUsage } : {}), updatedAtMs: nowMs };
}

/** Amounts arrive in minor units (cents) with `decimal_places` defaulting to 2. */
function parseExtraUsage(value: unknown): ExtraUsage | null {
  const extra = object(value);
  const currency = text(extra?.currency);
  if (!extra || !currency) return null;
  const places = number(extra.decimal_places) ?? 2;
  const limit = number(extra.monthly_limit);
  return {
    enabled: extra.is_enabled === true,
    currency,
    used: (number(extra.used_credits) ?? 0) / 10 ** places,
    monthlyLimit: limit === null ? null : limit / 10 ** places,
  };
}

function parseCredits(value: unknown): CreditBalance | null {
  const body = object(value);
  const currency = text(body?.currency);
  const amount = number(body?.amount);
  if (!currency || amount === null) return null;
  const places = number(object(object(body?.balance)?.credits)?.exponent) ?? 2;
  return { currency, amount: amount / 10 ** places };
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Only limits covering every model block the account; a scoped limit blocks one model.
 * Enabled extra usage keeps serving requests past every plan limit.
 */
export function quotaState(quota: AccountQuota, nowMs = Date.now()): ResourceState {
  if (quota.extraUsage?.enabled) return { status: "ready" };
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

/**
 * The weekly all-models reset, after which unused plan quota is lost: the host spends the
 * account whose week ends first. Null once a limit covering every model is used up, because the
 * account then serves only through paid extra usage, or while the reset time is unknown.
 */
export function quotaResetAtMs(quota: AccountQuota, nowMs = Date.now()): number | null {
  const limits = quota.limits.filter((limit) => limit.model === null);
  if (limits.some((limit) => limit.usedPercent >= 100)) return null;
  const resetAtMs = limits.find((limit) => limit.kind === "weekly_all")?.resetAtMs ?? null;
  return resetAtMs !== null && resetAtMs > nowMs ? resetAtMs : null;
}

export function quotaMetrics(quota: AccountQuota | null | undefined): ResourceMetric[] {
  const metrics: ResourceMetric[] = (quota?.limits ?? []).map((limit) => ({
    id: limit.model ? `${limit.kind}:${limit.model}` : limit.kind,
    label: limitLabel(limit),
    unit: "percent",
    value: 100 - limit.usedPercent,
    ...(limit.resetAtMs !== null ? { resetAtMs: limit.resetAtMs } : {}),
  }));
  const extra = quota?.extraUsage;
  if (extra) {
    metrics.push({
      id: "extra_usage",
      label: {
        "en-US": "Extra usage this month",
        "ru-RU": "Доп. использование за месяц",
        "zh-CN": "本月额外用量",
      },
      unit: "currency",
      currency: extra.currency,
      value: extra.used,
      ...(extra.monthlyLimit !== null ? { limit: extra.monthlyLimit } : {}),
    });
  }
  if (quota?.credits) {
    metrics.push({
      id: "credits",
      label: { "en-US": "Usage credits", "ru-RU": "Кредиты использования", "zh-CN": "用量余额" },
      unit: "currency",
      currency: quota.credits.currency,
      value: quota.credits.amount,
    });
  }
  return metrics;
}

function limitLabel(limit: QuotaLimit): ResourceMetric["label"] {
  if (limit.kind === "session") {
    return { "en-US": "5-hour window", "ru-RU": "5-часовое окно", "zh-CN": "5 小时窗口" };
  }
  if (limit.kind === "weekly_all") {
    return {
      "en-US": "Weekly, all models",
      "ru-RU": "Неделя, все модели",
      "zh-CN": "周额度(全部模型)",
    };
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
