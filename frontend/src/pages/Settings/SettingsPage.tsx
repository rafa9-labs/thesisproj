import { Settings as SettingsIcon, Cpu, Database, Key, Info } from "lucide-react";

const settingsSections = [
  {
    icon: SettingsIcon,
    title: "General",
    description: "Theme, verbose mode, data directory, API URL",
  },
  {
    icon: Cpu,
    title: "GPU & Compute",
    description: "CUDA status, thread budget, mixed precision",
  },
  {
    icon: Database,
    title: "Data Sources",
    description: "OANDA API key, pair downloads, data integrity",
  },
  {
    icon: Key,
    title: "License",
    description: "License activation, trial status, machine ID",
  },
  {
    icon: Info,
    title: "About",
    description: "Version info, pipeline stats, check for updates",
  },
];

export function SettingsPage() {
  return (
    <div className="flex flex-col gap-6">
      <h2
        className="text-base font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Settings
      </h2>

      <div className="grid gap-4">
        {settingsSections.map((section) => (
          <div
            key={section.title}
            className="flex items-start gap-4 rounded-lg border p-4"
            style={{
              backgroundColor: "var(--color-surface)",
              borderColor: "var(--color-border)",
            }}
          >
            <section.icon size={20} style={{ color: "var(--color-text-muted)", marginTop: 2 }} />
            <div className="flex flex-col gap-1">
              <span className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
                {section.title}
              </span>
              <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
                {section.description}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
