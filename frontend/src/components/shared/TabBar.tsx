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
  const activeIndex = tabs.findIndex((t) => t.key === activeTab);

  return (
    <nav
      className="flex flex-row gap-0.5 border-b border-(--color-glass-border) px-2 sm:px-6"
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
            className="relative flex flex-1 min-w-0 items-center justify-center px-1 pt-2.5 pb-2 text-[9px] font-semibold tracking-[0.05em] uppercase transition-colors duration-150 sm:text-[10px] md:px-2"
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
            <span className="truncate">{tab.label}</span>
            {isActive && (
              <span
                className="absolute right-0 bottom-0 left-0 h-0.5 bg-(--color-brand)"
                style={{ borderRadius: "2px 2px 0 0" }}
              />
            )}
          </button>
        );
      })}

      {/* Compact step indicator on small screens */}
      <span className="ml-2 hidden max-[480px]:inline-flex items-center text-[9px] text-(--color-text-muted)/50 font-mono shrink-0">
        {activeIndex + 1}/{tabs.length}
      </span>
    </nav>
  );
}
