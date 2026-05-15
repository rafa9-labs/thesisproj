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
      className="flex items-center gap-0.5 rounded-lg p-0.5"
      style={{ backgroundColor: "var(--color-elevated)" }}
    >
      {tabs.map((tab) => {
        const disabled = disabledTabs?.has(tab.key);
        const isActive = activeTab === tab.key;
        return (
          <button
            key={tab.key}
            onClick={() => !disabled && onTabChange(tab.key)}
            disabled={disabled}
            className="rounded-md px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all duration-150"
            style={{
              backgroundColor: isActive ? "var(--color-surface)" : "transparent",
              color: isActive
                ? "var(--color-brand)"
                : disabled
                  ? "var(--color-text-muted)"
                  : "var(--color-text-secondary)",
              boxShadow: isActive ? "0 1px 3px rgba(0,0,0,0.15)" : "none",
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.5 : 1,
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
