import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div
          className="flex h-full items-center justify-center p-8"
          style={{ backgroundColor: "var(--color-app)" }}
        >
          <div
            className="flex flex-col items-center gap-4 rounded-sm border p-8"
            style={{
              maxWidth: 480,
              borderColor: "var(--color-accent-danger)",
              backgroundColor: "var(--color-surface)",
            }}
          >
            <span className="text-lg font-bold" style={{ color: "var(--color-accent-danger)" }}>
              Something went wrong
            </span>
            <p
              className="text-center text-xs"
              style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
            >
              {this.state.error?.message ?? "An unexpected error occurred."}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="rounded-md border px-4 py-2 text-xs font-bold uppercase"
              style={{
                borderColor: "var(--color-brand)",
                backgroundColor: "rgba(41,98,255,0.1)",
                color: "var(--color-brand)",
                cursor: "pointer",
              }}
            >
              Reload
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
