interface Tab {
  key: string;
  label: string;
}

interface Props {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (key: string) => void;
  disabledTabs?: Set<string>;
}

export function TabBar({ tabs, activeTab, onTabChange, disabledTabs }: Props) {
  return (
    <div
      className="flex items-end"
      style={{ borderBottom: "1px solid var(--color-glass-border)" }}
    >
      {tabs.map((tab) => {
        const disabled = disabledTabs?.has(tab.key);
        const isActive = activeTab === tab.key;

        return (
          <button
            key={tab.key}
            onClick={() => !disabled && onTabChange(tab.key)}
            disabled={disabled}
            className="relative px-4 pb-3 pt-2 text-[11px] font-semibold uppercase tracking-[0.08em] transition-colors duration-150"
            style={{
              background: "transparent",
              border: "none",
              color: isActive
                ? "var(--color-text-primary)"
                : disabled
                ? "var(--color-text-muted)"
                : "rgba(255,255,255,0.38)",
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.4 : 1,
              outline: "none",
              whiteSpace: "nowrap",
            }}
          >
            {tab.label}
            {/* Active underline */}
            {isActive && (
              <span
                style={{
                  position: "absolute",
                  bottom: -1,
                  left: 0,
                  right: 0,
                  height: 2,
                  backgroundColor: "var(--color-brand)",
                  borderRadius: "2px 2px 0 0",
                }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}
