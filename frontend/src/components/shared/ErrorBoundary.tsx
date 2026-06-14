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
        <div className="flex h-full items-center justify-center bg-(--color-app) p-8">
          <div
            className="flex flex-col items-center gap-4 rounded-sm border border-(--color-accent-danger) bg-(--color-surface) p-8"
            style={{ maxWidth: 480 }}
          >
            <span className="text-lg font-bold text-(--color-accent-danger)">
              Something went wrong
            </span>
            <p className="text-center font-mono text-xs text-(--color-text-muted)">
              {this.state.error?.message ?? "An unexpected error occurred."}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="rounded-md border border-(--color-brand) px-4 py-2 text-xs font-bold text-(--color-brand) uppercase"
              style={{ backgroundColor: "rgba(41,98,255,0.1)" }}
              className="cursor-pointer"
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
