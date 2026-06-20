import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";

const queryFn = vi.fn();

function buildQueryClient(staleMs = 5 * 60_000): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: staleMs,
        gcTime: 10 * 60_000,
        retry: 0,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
      },
    },
  });
}

function TestConsumer({ label }: { label: string }) {
  const { data } = useQuery({
    queryKey: ["test-data"],
    queryFn: async () => {
      queryFn();
      return { value: label };
    },
  });
  return <span data-testid="output">{data?.value ?? "loading"}</span>;
}

function renderWithClient(ui: React.ReactElement, client: QueryClient) {
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("TanStack Query cache hit on navigation", () => {
  beforeEach(() => {
    queryFn.mockClear();
  });

  it("serves from cache on remount within staleTime (simulates tab switching)", async () => {
    const client = buildQueryClient();

    // Mount first "page" — triggers fetch
    const { unmount } = renderWithClient(<TestConsumer label="dashboard" />, client);
    await waitFor(() => {
      expect(screen.getByTestId("output")).toHaveTextContent("dashboard");
    });
    expect(queryFn).toHaveBeenCalledTimes(1);

    // Unmount (simulate navigating away)
    unmount();
    cleanup();

    // Mount second "page" with same queryKey — should return from cache
    renderWithClient(<TestConsumer label="committee" />, client);
    await waitFor(() => {
      expect(screen.getByTestId("output")).toHaveTextContent("dashboard");
    });
    // Still only 1 call — data came from cache
    expect(queryFn).toHaveBeenCalledTimes(1);
  });

  it("refetches when staleTime is zero", async () => {
    const client = buildQueryClient(0); // staleTime=0 means always stale

    const { unmount } = renderWithClient(<TestConsumer label="first" />, client);
    await waitFor(() => {
      expect(screen.getByTestId("output")).toHaveTextContent("first");
    });
    expect(queryFn).toHaveBeenCalledTimes(1);
    unmount();
    cleanup();

    // Re-mount with staleTime=0 — should trigger refetch
    renderWithClient(<TestConsumer label="second" />, client);
    await waitFor(() => {
      expect(screen.getByTestId("output")).toHaveTextContent("second");
    });
    expect(queryFn).toHaveBeenCalledTimes(2);
  });
});
