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
    <nav
      className="grid grid-cols-3 gap-x-2 gap-y-4 border-b border-(--color-glass-border) pb-px xl:flex xl:flex-row xl:gap-8"
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
            className="relative flex items-center justify-center px-0 pt-2 pb-2 text-[10px] font-semibold tracking-[0.06em] whitespace-nowrap uppercase transition-colors duration-150 md:text-xs"
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
            }}
          >
            {tab.label}
            {isActive && (
              <span
                className="absolute right-0 bottom-0 left-0 h-0.5 bg-(--color-brand)"
                style={{ borderRadius: "2px 2px 0 0" }}
              />
            )}
          </button>
        );
      })}
    </nav>
  );
}
