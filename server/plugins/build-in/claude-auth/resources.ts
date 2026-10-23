import type { ResourcePatch, ResourceSupport, ResourceView } from "cursor-byok:resource";
import {
  type AccountData,
  accountData,
  isExpiring,
  OAuthError,
  privateData,
  refreshTokens,
  RESOURCE_TYPE,
} from "./auth.ts";
import { claudeOAuth } from "./oauth.ts";
import {
  fetchCredits,
  fetchUsage,
  quotaMetrics,
  quotaResetAtMs,
  quotaState,
  UsageError,
} from "./usage.ts";

export const claudeAccounts: ResourceSupport = {
  type: RESOURCE_TYPE,
  displayName: { "en-US": "Claude accounts", "ru-RU": "Аккаунты Claude", "zh-CN": "Claude 账号" },
  add: [claudeOAuth],
  present(resource): ResourceView {
    const data = accountData(resource);
    const metrics = quotaMetrics(data.quota);
    return {
      displayName: data.displayName,
      description: {
        "en-US": "Full Claude subscription account.",
        "ru-RU": "Полноценный аккаунт подписки Claude.",
        "zh-CN": "完整的 Claude 订阅账号。",
      },
      ...(metrics.length > 0 ? { metrics } : {}),
    };
  },
  /**
   * Reads subscription usage. Tokens rotate only when they expire or the usage endpoint
   * rejects them, so periodic refreshes do not churn refresh tokens.
   */
  async refresh(resource, context): Promise<ResourcePatch> {
    let data: AccountData = accountData(resource);
    let rotated = false;
    const rotate = async () => {
      data = await refreshTokens(data, context);
      rotated = true;
    };
    try {
      if (isExpiring(data)) await rotate();
      let quota;
      try {
        quota = await fetchUsage(data.accessToken, context);
      } catch (error) {
        if (!(error instanceof UsageError) || error.status !== 401 || rotated) throw error;
        await rotate();
        quota = await fetchUsage(data.accessToken, context);
      }
      // A failed balance lookup keeps the previous balance, like a failed usage lookup does.
      const credits = await fetchCredits(data.accessToken, data.organizationId, context) ??
        data.quota?.credits;
      if (credits) quota.credits = credits;
      return {
        privateData: privateData({ ...data, quota }),
        state: quotaState(quota),
        quotaResetAtMs: quotaResetAtMs(quota),
      };
    } catch (error) {
      if (error instanceof OAuthError && error.invalidCredentials) {
        return { state: { status: "invalid", message: error.message } };
      }
      if (error instanceof UsageError && error.status === 401) {
        const message = "Claude authorization was rejected. Sign in again.";
        return { privateData: privateData(data), state: { status: "invalid", message } };
      }
      // Rotated tokens must persist even when the usage lookup fails; the old ones are revoked.
      if (rotated) return { privateData: privateData(data) };
      throw error;
    }
  },
};
