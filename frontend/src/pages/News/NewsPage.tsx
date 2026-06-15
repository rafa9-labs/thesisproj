import { useNewsStatus } from "@/api/queries";

// ─── Primitives ────────────────────────────────────────────────────────────────

function Dot({ color }: { color: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 6,
        height: 6,
        borderRadius: "50%",
        backgroundColor: color,
        flexShrink: 0,
      }}
    />
  );
}

function RibbonItem({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="flex items-center gap-2" style={{ borderRight: "1px solid #2A2E39", paddingRight: 16, marginRight: 16 }}>
      <span style={{ fontSize: 10, color: "#4B5563", letterSpacing: "0.08em", textTransform: "uppercase" }}>
        {label}
      </span>
      <span
        style={{
          fontSize: 11,
          fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
          color: ok === undefined ? "#D1D4DC" : ok ? "#089981" : "#787B86",
          letterSpacing: "0.04em",
        }}
      >
        {value}
      </span>
      {ok !== undefined && <Dot color={ok ? "#089981" : "#2A2E39"} />}
    </div>
  );
}

const TH: React.CSSProperties = {
  padding: "6px 12px",
  textAlign: "left",
  fontSize: 9,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  color: "#787B86",
  borderBottom: "1px solid #2A2E39",
  backgroundColor: "#1E222D",
  fontWeight: 500,
};

const TD: React.CSSProperties = {
  padding: "7px 12px",
  fontSize: 11,
  borderBottom: "1px solid #1E222D",
  color: "#787B86",
  fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
};

// ─── Mock placeholder rows ─────────────────────────────────────────────────────

const FEED_PLACEHOLDER = [
  { time: "09:32:14", headline: "ECB signals potential rate pause amid inflation data", sentiment: "+0.42" },
  { time: "09:18:07", headline: "USD strengthens ahead of NFP release Friday", sentiment: "+0.61" },
  { time: "08:55:43", headline: "EUR/USD breaks 1.0850 resistance on eurozone PMI beat", sentiment: "+0.38" },
  { time: "08:41:22", headline: "BoE governor warns of prolonged restrictive stance", sentiment: "-0.29" },
  { time: "08:12:09", headline: "Risk-off sentiment persists; JPY gaining across the board", sentiment: "-0.55" },
  { time: "07:44:58", headline: "Oil slips on demand concerns; commodity currencies weaker", sentiment: "-0.47" },
];

const EVENTS_PLACEHOLDER = [
  { date: "2025-05-23", event: "EUR CPI (Flash)", impact: "HIGH" },
  { date: "2025-05-23", event: "US Jobless Claims", impact: "MED" },
  { date: "2025-05-24", event: "US NFP", impact: "HIGH" },
  { date: "2025-05-26", event: "BoE Minutes", impact: "MED" },
  { date: "2025-05-27", event: "EUR GDP (Final)", impact: "LOW" },
  { date: "2025-05-28", event: "FOMC Member Speech", impact: "LOW" },
];

const IMPACT_COLOR: Record<string, string> = {
  HIGH: "#F23645",
  MED: "#F59E0B",
  LOW: "#787B86",
};

// ─── Main page ─────────────────────────────────────────────────────────────────

export function NewsPage() {
  const { data: status, isLoading } = useNewsStatus();

  const backend = status?.sentiment_backend?.toUpperCase() ?? "VADER";
  const finbertOk = status?.finbert_available ?? false;
  const articles = isLoading ? "…" : String(status?.cached_articles ?? 0);
  const eventTypes = status?.event_types ?? [];

  return (
    <div className="flex flex-col gap-4" style={{ minHeight: 0 }}>

      {/* Status ribbon */}
      <div
        className="flex items-center flex-wrap"
        style={{
          height: 36,
          backgroundColor: "#1E222D",
          border: "1px solid #2A2E39",
          borderRadius: 4,
          padding: "0 14px",
        }}
      >
        <RibbonItem label="Engine" value={backend} />
        <RibbonItem label="finBERT" value={finbertOk ? "Ready" : "Not Installed"} ok={finbertOk} />
        <RibbonItem label="Articles Cached" value={articles} />
        {eventTypes.length > 0 && (
          <RibbonItem label="Event Types" value={String(eventTypes.length)} />
        )}
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 10, color: "#2A2E39", fontFamily: "var(--font-mono)" }}>
          {isLoading ? "CONNECTING…" : status ? "LIVE" : "BACKEND OFFLINE"}
        </span>
      </div>

      {/* Split view */}
      <div className="flex gap-4" style={{ flex: 1, minHeight: 0 }}>

        {/* Left: Live Feed */}
        <div
          style={{
            flex: "0 0 60%",
            backgroundColor: "#1E222D",
            border: "1px solid #2A2E39",
            borderRadius: 4,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            className="flex items-center justify-between"
            style={{ padding: "8px 12px", borderBottom: "1px solid #2A2E39" }}
          >
            <span style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "#4B5563", fontWeight: 600 }}>
              Live Feed
            </span>
            <span style={{ fontSize: 9, color: "#2A2E39", fontFamily: "var(--font-mono)", letterSpacing: "0.06em" }}>
              PLACEHOLDER DATA
            </span>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ ...TH, width: 72 }}>Time</th>
                <th style={TH}>Headline</th>
                <th style={{ ...TH, width: 72, textAlign: "right" as const }}>Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {FEED_PLACEHOLDER.map((row, i) => {
                const sentVal = parseFloat(row.sentiment);
                const sentColor = sentVal > 0 ? "#089981" : sentVal < 0 ? "#F23645" : "#787B86";
                return (
                  <tr key={i} style={{ backgroundColor: i % 2 === 0 ? "transparent" : "#131722" }}>
                    <td style={{ ...TD, color: "#4B5563" }}>{row.time}</td>
                    <td style={{ ...TD, color: "#787B86", fontFamily: "inherit", fontSize: 11 }}>{row.headline}</td>
                    <td style={{ ...TD, textAlign: "right", color: sentColor }}>{row.sentiment}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Right: Event Calendar */}
        <div
          style={{
            flex: "0 0 40%",
            backgroundColor: "#1E222D",
            border: "1px solid #2A2E39",
            borderRadius: 4,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            className="flex items-center justify-between"
            style={{ padding: "8px 12px", borderBottom: "1px solid #2A2E39" }}
          >
            <span style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "#4B5563", fontWeight: 600 }}>
              Event Calendar
            </span>
            <span style={{ fontSize: 9, color: "#2A2E39", fontFamily: "var(--font-mono)", letterSpacing: "0.06em" }}>
              PLACEHOLDER DATA
            </span>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ ...TH, width: 90 }}>Date</th>
                <th style={TH}>Event</th>
                <th style={{ ...TH, width: 56, textAlign: "right" as const }}>Impact</th>
              </tr>
            </thead>
            <tbody>
              {EVENTS_PLACEHOLDER.map((row, i) => (
                <tr key={i} style={{ backgroundColor: i % 2 === 0 ? "transparent" : "#131722" }}>
                  <td style={{ ...TD, color: "#4B5563" }}>{row.date}</td>
                  <td style={{ ...TD, color: "#787B86", fontFamily: "inherit", fontSize: 11 }}>{row.event}</td>
                  <td style={{ ...TD, textAlign: "right", color: IMPACT_COLOR[row.impact] ?? "#787B86" }}>
                    {row.impact}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
