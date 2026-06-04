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
      className="flex items-end w-full justify-between px-4"
      role="tablist"
    >
      {tabs.map((tab) => {
        const disabled = disabledTabs?.has(tab.key);
        const isActive = activeTab === tab.key;

        return (
          <button
            key={tab.key}
            onClick={() => !disabled && onTabChange(tab.key)}
            disabled={disabled}
            role="tab"
            id={`tab-${tab.key}`}
            aria-selected={isActive}
            aria-controls={`tabpanel-${tab.key}`}
            className="relative pb-2 px-0 pt-2 text-[11px] font-semibold uppercase tracking-[0.08em] transition-colors duration-150"
            style={{
              background: "transparent",
              border: "none",
              color: isActive
                ? "var(--color-text-primary)"
                : disabled
                ? "var(--color-text-muted)"
                : "var(--color-text-muted)",
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.4 : 1,
              outline: "none",
              whiteSpace: "nowrap",
            }}
          >
            {tab.label}
            {isActive && (
              <span
                className="absolute bottom-0 left-0 right-0 h-0.5"
                style={{
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
