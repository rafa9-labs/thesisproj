const ctx = self as unknown as Worker;

ctx.addEventListener("message", (event: MessageEvent) => {
  const { type, data } = event.data;

  if (type === "parse-results") {
    try {
      const parsed = JSON.parse(data as string);
      const equityCurve = parsed.equity_curve ?? [];
      const monthlyResults = parsed.monthly_results ?? [];
      const metrics = parsed.metrics ?? [];

      const chartData = equityCurve.map((v: number, i: number) => ({
        time: i,
        value: v,
      }));

      ctx.postMessage({
        type: "parse-results-done",
        metrics,
        equityCurve: chartData,
        monthlyResults,
        raw: parsed,
      });
    } catch (err) {
      ctx.postMessage({
        type: "parse-results-error",
        error: (err as Error).message,
      });
    }
  }
});
