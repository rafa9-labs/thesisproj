interface Step {
  key: string;
  label: string;
}

interface Props {
  steps: Step[];
  activeStep: string;
  onStepChange: (key: string) => void;
  disabledSteps?: Set<string>;
}

import { Check } from "lucide-react";

export function StepTracker({ steps, activeStep, onStepChange, disabledSteps }: Props) {
  const activeIndex = steps.findIndex((s) => s.key === activeStep);

  return (
    <div className="flex w-full items-start" role="tablist">
      {steps.map((step, i) => {
        const disabled = disabledSteps?.has(step.key);
        const isActive = i === activeIndex;
        const isComplete = i < activeIndex;
        const isLast = i === steps.length - 1;

        const circleBg = isActive
          ? "var(--color-brand)"
          : isComplete
          ? "rgba(0,229,255,0.12)"
          : "transparent";
        const circleBorder = isActive || isComplete
          ? "var(--color-brand)"
          : "var(--color-text-dim)";
        const circleColor = isActive
          ? "var(--color-text-inverse)"
          : isComplete
          ? "var(--color-brand)"
          : "var(--color-text-muted)";

        return (
          <div key={step.key} className="flex flex-1 items-start" style={{ minWidth: 0 }}>
            <button
              onClick={() => !disabled && onStepChange(step.key)}
              disabled={disabled}
              role="tab"
              id={`step-${step.key}`}
              aria-selected={isActive}
              aria-controls={`tabpanel-${step.key}`}
              className="flex flex-col items-center gap-2 transition-opacity duration-150"
              style={{
                cursor: disabled ? "not-allowed" : "pointer",
                opacity: disabled ? 0.4 : 1,
                background: "transparent",
                border: "none",
                outline: "none",
                minWidth: 0,
              }}
            >
              <span
                className="flex items-center justify-center rounded-full text-[12px] font-bold transition-all duration-150"
                style={{
                  width: 34,
                  height: 34,
                  backgroundColor: circleBg,
                  border: `1.5px solid ${circleBorder}`,
                  color: circleColor,
                  fontFamily: "var(--font-mono)",
                  boxShadow: isActive ? "0 0 14px rgba(0,229,255,0.35)" : "none",
                }}
              >
                {isComplete ? <Check size={16} strokeWidth={2.5} /> : i + 1}
              </span>
              <span
                className="text-[11px] font-semibold tracking-[0.02em] text-center"
                style={{
                  color: isActive
                    ? "var(--color-brand)"
                    : isComplete
                    ? "var(--color-text-secondary)"
                    : "var(--color-text-muted)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  maxWidth: "100%",
                }}
              >
                {step.label}
              </span>
            </button>

            {!isLast && (
              <div
                className="mt-[17px] h-px flex-1"
                style={{
                  minWidth: 16,
                  backgroundColor: isComplete ? "var(--color-brand)" : "var(--color-text-dim)",
                  opacity: isComplete ? 0.6 : 0.4,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
