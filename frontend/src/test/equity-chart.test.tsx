import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { EquityChart } from "@/pages/Monitor/EquityChart";

vi.mock("recharts", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const FakeLineChart = ({ children }: any) => <div data-testid="line-chart">{children}</div>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const FakeResponsiveContainer = ({ children }: any) => (
    <div data-testid="responsive-container">{children}</div>
  );
  const FakeLine = () => <div data-testid="line" />;
  const FakeXAxis = () => <div data-testid="x-axis" />;
  const FakeYAxis = () => <div data-testid="y-axis" />;
  const FakeTooltip = () => <div data-testid="tooltip" />;
  const FakeLegend = () => <div data-testid="legend" />;
  const FakeCartesianGrid = () => <div data-testid="grid" />;
  return {
    LineChart: FakeLineChart,
    Line: FakeLine,
    XAxis: FakeXAxis,
    YAxis: FakeYAxis,
    Tooltip: FakeTooltip,
    Legend: FakeLegend,
    CartesianGrid: FakeCartesianGrid,
    ResponsiveContainer: FakeResponsiveContainer,
  };
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const emptyOosPeriods: any[] = [];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const emptyOosEquity: any[] = [];

const sampleOosEquity = [
  { period: 1, modelName: "logistic", equity: 1.05, bh: 1.02 },
  { period: 2, modelName: "logistic", equity: 1.08, bh: 1.01 },
  { period: 1, modelName: "xgboost", equity: 1.12, bh: 1.02 },
  { period: 2, modelName: "xgboost", equity: 1.15, bh: 1.01 },
];

const sampleOosPeriods = [
  { period: 1, sharpe: 1.5, return_pct: 5.0, trades: 10, drawdown: -0.02, win_rate: 0.6 },
  { period: 2, sharpe: 1.2, return_pct: 3.0, trades: 12, drawdown: -0.03, win_rate: 0.55 },
];

describe("EquityChart", () => {
  it("renders title", () => {
    render(
      <EquityChart
        models={["logistic"]}
        oosPeriods={emptyOosPeriods}
        oosEquity={emptyOosEquity}
      />
    );
    expect(screen.getByText("Walk-Forward Equity")).toBeInTheDocument();
  });

  it("shows waiting message when no data", () => {
    render(
      <EquityChart
        models={["logistic"]}
        oosPeriods={emptyOosPeriods}
        oosEquity={emptyOosEquity}
      />
    );
    expect(screen.getByText("Waiting for simulation data...")).toBeInTheDocument();
  });

  it("renders chart when data arrives", async () => {
    render(
      <EquityChart
        models={["logistic"]}
        oosPeriods={sampleOosPeriods}
        oosEquity={sampleOosEquity}
      />
    );
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument();
    expect(screen.getByTestId("line-chart")).toBeInTheDocument();
  });

  it("renders per-month summary table when oosPeriods present", () => {
    render(
      <EquityChart
        models={["logistic"]}
        oosPeriods={sampleOosPeriods}
        oosEquity={sampleOosEquity}
      />
    );
    expect(screen.getByText("Per-Month Summary")).toBeInTheDocument();
  });

  it("does not render per-month summary when oosPeriods empty", () => {
    render(
      <EquityChart
        models={["logistic"]}
        oosPeriods={emptyOosPeriods}
        oosEquity={sampleOosEquity}
      />
    );
    expect(screen.queryByText("Per-Month Summary")).not.toBeInTheDocument();
  });

  it("toggles y-axis mode on button click", () => {
    render(
      <EquityChart
        models={["logistic"]}
        oosPeriods={sampleOosPeriods}
        oosEquity={sampleOosEquity}
      />
    );
    const btn = screen.getByRole("button", { name: "%" });
    expect(btn).toBeInTheDocument();
    fireEvent.click(btn);
    expect(screen.getByRole("button", { name: "$" })).toBeInTheDocument();
  });

  it("renders multiple model lines", () => {
    render(
      <EquityChart
        models={["logistic", "xgboost"]}
        oosPeriods={sampleOosPeriods}
        oosEquity={sampleOosEquity}
      />
    );
    const lines = screen.getAllByTestId("line");
    expect(lines.length).toBeGreaterThanOrEqual(2);
  });
});
