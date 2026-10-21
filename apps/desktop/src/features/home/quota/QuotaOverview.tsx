import type { PluginDescriptor, PluginResourceMetric, PluginResourceView } from "../../../shared/api";
import { pluginMetricAmount, pluginText } from "../../../shared/api";
import { useI18n } from "../../../i18n/store";
import styles from "./QuotaOverview.module.scss";

type QuotaAccount = {
  key: string;
  icon: string;
  pluginName: string;
  account: PluginResourceView;
};

const DAY_MS = 24 * 60 * 60_000;

/** Every plugin account that reports at least one quota metric. */
export function quotaAccounts(plugins: PluginDescriptor[]): QuotaAccount[] {
  return plugins.flatMap((plugin) => plugin.resources.flatMap((resource) =>
    resource.resources
      .filter((account) => account.metrics.length > 0)
      .map((account) => ({
        key: `${plugin.id}/${resource.type}/${account.id}`,
        icon: plugin.icon,
        pluginName: plugin.name,
        account,
      }))));
}

function formatReset(resetAtMs: number, locale: string, nowMs: number) {
  const options: Intl.DateTimeFormatOptions = resetAtMs - nowMs < DAY_MS
    ? { hour: "2-digit", minute: "2-digit" }
    : { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" };
  return new Intl.DateTimeFormat(locale, options).format(resetAtMs);
}

function levelClass(remaining: number) {
  if (remaining < 10) return styles.critical;
  if (remaining < 25) return styles.low;
  return "";
}

/** Remaining share 0..100 for metrics drawn as a bar; null for plain values. */
function remainingPercent(metric: PluginResourceMetric): number | null {
  if (metric.unit === "percent") return metric.value;
  if (metric.unit === "currency" && metric.limit != null && metric.limit > 0) {
    return 100 * (1 - metric.value / metric.limit);
  }
  return null;
}

function MetricRow({ metric, locale, nowMs }: { metric: PluginResourceMetric; locale: string; nowMs: number }) {
  const label = pluginText(metric.label, locale);
  const share = remainingPercent(metric);
  const remaining = share === null ? null : Math.max(0, Math.min(100, share));
  return <div className={styles.metric}>
    <div className={styles.metricHeader}>
      <span>{label}</span>
      <span className={styles.value}>
        {metric.unit === "percent" ? t("剩余 {percent}%", { percent: Math.round(metric.value) }) : pluginMetricAmount(metric, locale)}
      </span>
    </div>
    {remaining !== null && <div className={styles.bar} role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={remaining}>
      <div className={[styles.fill, levelClass(remaining)].filter(Boolean).join(" ")} style={{ width: `${remaining}%` }} />
    </div>}
    {metric.resetAtMs != null && metric.resetAtMs > nowMs && <div className={styles.reset}>
      {t("重置于 {time}", { time: formatReset(metric.resetAtMs, locale, nowMs) })}
    </div>}
  </div>;
}

function AccountState({ state }: { state: PluginResourceView["state"] }) {
  if (state.status === "cooling") return <span className={styles.cooling} title={state.message ?? undefined}>{t("冷却中")}</span>;
  if (state.status === "invalid") return <span className={styles.invalid} title={state.message ?? undefined}>{t("已失效")}</span>;
  return null;
}

/** Remaining subscription quota of every plugin account, one card per account. */
export function QuotaOverview({ accounts }: { accounts: QuotaAccount[] }) {
  const { locale } = useI18n();
  const nowMs = Date.now();
  return <section className={styles.root} aria-label={t("订阅额度")}>
    {accounts.map(({ key, icon, pluginName, account }) => <article key={key} className={styles.account}>
      <header className={styles.header}>
        {icon && <img className={styles.icon} src={icon} alt="" />}
        <div className={styles.identity}>
          <strong title={account.displayName}>{account.displayName}</strong>
          <span>{pluginName}</span>
        </div>
        <AccountState state={account.state} />
      </header>
      {account.metrics.map((metric) => <MetricRow key={metric.id} metric={metric} locale={locale} nowMs={nowMs} />)}
    </article>)}
  </section>;
}
