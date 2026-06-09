import { useState, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useStartFullCycle, useFullCycleStatus, useFullCycleResults } from "@/api/queries";
import apiClient from "@/api/client";
import { useFullCycleStore, ALL_MODELS } from "@/stores/useFullCycleStore";
import { FC_PRESETS } from "@/lib/constants";
import { ModelGrid } from "./ModelGrid";
import { FullCycleProgress } from "./FullCycleProgress";
import { FullCycleResults } from "./FullCycleResults";
import { RunHistoryTable } from "./RunHistoryTable";
import { Bug, Cpu, Network, Layers, Bot } from "lucide-react";
import type { FullCycleRequest } from "@/api/schemas";

const PRESET_ICONS: Record<string, React.ReactNode> = {
  debug: <Bug size={13} />, classical: <Cpu size={13} />,
  deep: <Network size={13} />, full: <Layers size={13} />, llm: <Bot size={13} />,
};
const PRESET_COLORS: Record<string, string> = {
  debug: "var(--color-text-muted)", classical: "var(--color-brand)",
  deep: "#a78bfa", full: "var(--color-accent-warning)", llm: "var(--color-accent-danger)",
};

const inputStyle: React.CSSProperties = {
  width: 52, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)",
  borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px",
  fontSize: 10, fontFamily: "var(--font-mono)",
};
const selectStyle: React.CSSProperties = {
  background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)",
  borderRadius: 4, color: "var(--color-text-primary)", padding: "3px 6px", fontSize: 10,
};
const rowStyle: React.CSSProperties = {
  display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap",
  padding: "10px 16px", borderBottom: "1px solid var(--color-glass-border)",
};
const labelStyle: React.CSSProperties = { fontSize: 10, color: "var(--color-text-muted)" };

function ProposerArchitecture({ proposer, ucbC, llmBackend }: { proposer: string; ucbC: number; llmBackend: string }) {
  const [expanded, setExpanded] = useState(false);

  const isLLM = proposer === "llm";
  const isHybrid = proposer === "hybrid_llm_ucb1";

  const boxBg = "var(--color-elevated)";
  const boxBorder = "1px solid var(--color-glass-border)";
  const arrow = { color: "var(--color-text-dim)", fontSize: 9, margin: "0 6px" };
  const activeColor = isHybrid ? "#A78BFA" : "#089981";
  const mutedColor = "var(--color-text-muted)";

  return (
    <div style={{ marginTop: 12, background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, overflow: "hidden" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", cursor: "pointer", userSelect: "none" }} onClick={() => setExpanded(!expanded)}>
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", color: "var(--color-text-primary)" }}>
          Phase 6 Factory Architecture
        </span>
        <span style={{ fontSize: 10, padding: "1px 8px", borderRadius: 3, background: isHybrid ? "rgba(139,92,246,0.12)" : isLLM ? "rgba(8,153,129,0.12)" : "rgba(229,160,20,0.12)", color: isHybrid ? "#A78BFA" : isLLM ? "#089981" : "#E5A014", fontFamily: "var(--font-mono)" }}>
          {isHybrid ? "LLM + UCB1 Hybrid" : isLLM ? "LLM Director" : "UCB1 Bandit"}
        </span>
        <span style={{ marginLeft: "auto", fontSize: 9, color: "var(--color-text-dim)" }}>{expanded ? "collapse" : "expand"}</span>
      </div>

      {/* Compact flow -- always visible */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "10px 14px 14px", gap: 4, flexWrap: "wrap" }}>
        {/* LLM box */}
        <div style={{ background: boxBg, border: boxBorder, borderRadius: 4, padding: "6px 12px", textAlign: "center" }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: activeColor, letterSpacing: "0.04em" }}>LLM</div>
          <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>{llmBackend}</div>
          {isHybrid && <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>shortlist builder</div>}
          {isLLM && <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>strategic director</div>}
        </div>
        <span style={arrow}>→</span>
        {/* UCB1 box */}
        <div style={{ background: boxBg, border: boxBorder, borderRadius: 4, padding: "6px 12px", textAlign: "center", opacity: isHybrid ? 1 : 0.35 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: isHybrid ? activeColor : mutedColor, letterSpacing: "0.04em" }}>UCB1</div>
          <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>c={ucbC}</div>
          {isHybrid && <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>tactical optimizer</div>}
        </div>
        <span style={arrow}>→</span>
        {/* Committee box */}
        <div style={{ background: boxBg, border: boxBorder, borderRadius: 4, padding: "6px 12px", textAlign: "center" }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: activeColor, letterSpacing: "0.04em" }}>Committee</div>
          <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>regime x model</div>
        </div>
        <span style={arrow}>→</span>
        {/* WFO box */}
        <div style={{ background: boxBg, border: boxBorder, borderRadius: 4, padding: "6px 12px", textAlign: "center" }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: activeColor, letterSpacing: "0.04em" }}>WFO</div>
          <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>proxy backtest</div>
        </div>
        {isHybrid && (
          <>
            <span style={arrow}>→</span>
            {/* Feedback loop */}
            <div style={{ background: boxBg, border: "1px dashed var(--color-accent-success)", borderRadius: 4, padding: "6px 12px", textAlign: "center" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-accent-success)", letterSpacing: "0.04em" }}>Feedback</div>
              <div style={{ fontSize: 8, color: "var(--color-text-dim)" }}>Delta Sharpe {"->"} UCB1</div>
            </div>
          </>
        )}
      </div>

      {/* Expanded details */}
      {expanded && (
        <div style={{ borderTop: "1px solid var(--color-glass-border)", padding: "12px 14px", fontSize: 10, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
          {isLLM && (
            <p style={{ margin: "0 0 8px" }}>
              The <strong style={{ color: activeColor }}>LLM Strategic Director</strong> analyzes the full
              regime x model performance matrix, current committee config, and iteration history.
              It proposes the single most logical <strong>swap / add / remove</strong> action per iteration.
              The LLM understands model family compatibility (e.g., no mean-reversion models in trending regimes).
              Each iteration is a fresh analysis -- there is no learning between iterations.
              Backends: {llmBackend}.
            </p>
          )}
          {isHybrid && (
            <>
              <p style={{ margin: "0 0 8px" }}>
                <strong style={{ color: "#A78BFA" }}>Hybrid Mode — Strategic + Tactical:</strong> The LLM acts as
                the strategic director, pruning fundamentally illogical model/regime pairings down to a shortlist
                of at most <strong>5 candidates</strong>. UCB1 then acts as the tactical manager, mathematically
                balancing exploration vs exploitation over this pruned set.
              </p>
              <p style={{ margin: "0 0 8px" }}>
                <strong style={{ color: "var(--color-text-muted)" }}>LLM refresh:</strong> every <strong>5 iterations</strong>
                or when UCB1 converges (all arms below baseline Sharpe + 0.005). This prevents excessive API calls
                while keeping the shortlist fresh as the committee evolves.
              </p>
              <p style={{ margin: 0 }}>
                <strong style={{ color: "var(--color-accent-success)" }}>Feedback loop:</strong> After each WFO
                backtest, the Sharpe delta is fed back to UCB1 via <span style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>record_result(delta)</span>.
                Arms that consistently improve Sharpe get higher running means and are selected more often.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function FullCycleTab() {
  const store = useFullCycleStore();
  const startMutation = useStartFullCycle();
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (store.selectedHistoryJobId) {
      setJobId(store.selectedHistoryJobId);
      store.setSelectedHistoryJobId(null);
    }
  }, [store.selectedHistoryJobId]);

  const { data: status } = useFullCycleStatus(jobId);
  const terminalPhases = new Set(["completed", "failed", "validation_failed", "cancelled", "orphaned"]);
  const isRunning = status && !terminalPhases.has(status.phase);
  const isDone = status && terminalPhases.has(status.phase);

  const cancelMutation = useMutation({
    mutationFn: async () => { await apiClient.post(`/committee/full-cycle/${jobId}/cancel`); },
    onSuccess: () => {
      queryClient.cancelQueries({ queryKey: ["full-cycle", "status", jobId] });
      if (jobId) queryClient.setQueryData(["full-cycle", "status", jobId], {
        phase: "cancelled", phase_number: 0, phase_progress: "", iteration: 0, total_iterations: 0,
        current_action: "Cancelled by user", best_sharpe_so_far: 0, started_at: "", error: "Cancelled by user",
        surviving_models: [], locked_features_count: 0, job_id: jobId,
      });
    },
  });

  const handleDeploy = async () => {
    const payload: FullCycleRequest = {
      models: store.selectedModels, pair: store.pair, timeframe: store.timeframe,
      sweep_n_estimators: store.sweepNEstimators, sweep_max_depth: store.sweepMaxDepth,
      skip_feature_sweep: store.skipFeatureSweep, use_boruta_shap: store.useBorutaShap,
      boruta_percentile: store.borutaPercentile, boruta_max_iter: store.borutaMaxIter,
      enable_phase3: store.enablePhase3, enable_phase4: store.enablePhase4,
      enable_phase5: store.enablePhase5, enable_phase6: store.enablePhase6,
      debug_mode: store.debugMode, committee_top_k: store.committeeTopK,
      committee_weight_method: store.committeeWeightMethod, committee_min_sharpe: store.committeeMinSharpe,
      train_months: store.trainMonths, test_months: store.testMonths,
      hpo_sampler: store.hpoSampler, cv_blocks: store.cvBlocks, cv_val_frac: 0.05, plateau_patience: 15,
      proposer: store.proposer, llm_backend: store.llmBackend, ucb_c: store.ucb1ExplorationC,
      max_iterations: store.maxIterations, patience: store.factoryPatience, stopping_tolerance: store.stoppingTolerance,
    };
    if (Object.keys(store.hpoTrials).length > 0) (payload as any).hpo_trials = store.hpoTrials;
    if (Object.keys(store.hpoStartupTrials).length > 0) (payload as any).hpo_startup_trials = store.hpoStartupTrials;
    try {
      const result = await startMutation.mutateAsync(payload);
      setErrorMsg(null); setJobId(result.job_id);
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail ?? err?.message ?? "Failed to start full cycle");
    }
  };

  const hasModels = store.selectedModels.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: "100%" }}>
      {!isRunning && !isDone && (
        <>
          {/* Presets */}
          <div style={{ marginTop: 16, display: "flex", gap: 8, flexWrap: "wrap" }}>
            {Object.entries(FC_PRESETS).map(([key, preset]) => {
              const catColor = PRESET_COLORS[key] ?? "var(--color-text-muted)";
              const isActive = store.activePreset === key;
              return (
                <button key={key} onClick={() => store.applyPreset(key)} style={{
                  display: "flex", alignItems: "center", gap: 5,
                  background: isActive ? catColor : "var(--color-glass)",
                  border: `1px solid ${isActive ? "transparent" : "var(--color-glass-border)"}`,
                  borderRadius: 4, padding: "4px 10px", fontSize: 10, fontFamily: "var(--font-mono)",
                  color: isActive ? (key === "debug" ? "var(--color-text-primary)" : "var(--color-app)") : "var(--color-text-secondary)",
                  cursor: "pointer", fontWeight: isActive ? 600 : 400,
                }} title={preset.desc}>
                  <span style={{ color: isActive ? "inherit" : catColor }}>{PRESET_ICONS[key]}</span>
                  {preset.label}
                </button>
              );
            })}
          </div>

          {/* Pair + Timeframe */}
          <div style={{ marginTop: 16, display: "flex", gap: 16, alignItems: "center" }}>
            <label style={labelStyle}>Pair: <select value={store.pair} onChange={(e) => store.setPair(e.target.value)} style={selectStyle}>
              <option value="EURUSD">EURUSD</option><option value="GBPUSD">GBPUSD</option><option value="USDJPY">USDJPY</option>
            </select></label>
            <label style={labelStyle}>TF: <select value={store.timeframe} onChange={(e) => store.setTimeframe(e.target.value)} style={selectStyle}>
              <option value="M30">M30</option><option value="H1">H1</option><option value="H4">H4</option>
            </select></label>
            <label style={{ ...labelStyle, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input type="checkbox" checked={store.debugMode} onChange={(e) => store.setDebugMode(e.target.checked)} />
              Debug
            </label>
          </div>

          {/* Models */}
          <div style={{ marginTop: 16 }}><ModelGrid selected={store.selectedModels} onToggle={store.toggleModel} /></div>

          {/* Config */}
          <div style={{ marginTop: 16, background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, overflow: "hidden" }}>
            {/* Feature Sweep */}
            <div style={rowStyle}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-primary)", letterSpacing: "0.06em", minWidth: 100 }}>Feature Sweep</span>
              <label style={labelStyle}>Trees: <input type="number" value={store.sweepNEstimators} onChange={(e) => store.setSweepNEstimators(Number(e.target.value))} min={50} max={300} step={10} style={inputStyle} /></label>
              <label style={labelStyle}>Depth: <input type="number" value={store.sweepMaxDepth} onChange={(e) => store.setSweepMaxDepth(Number(e.target.value))} min={2} max={10} step={1} style={inputStyle} /></label>
              <label style={{ ...labelStyle, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                <input type="checkbox" checked={store.useBorutaShap} onChange={(e) => store.setUseBorutaShap(e.target.checked)} />
                BorutaSHAP
              </label>
              {store.useBorutaShap && (
                <label style={labelStyle}>%: <input type="number" value={store.borutaPercentile} onChange={(e) => store.setBorutaPercentile(Number(e.target.value))} min={80} max={95} step={5} style={{ ...inputStyle, width: 36 }} /></label>
              )}
              <label style={{ ...labelStyle, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                <input type="checkbox" checked={store.skipFeatureSweep} onChange={(e) => store.setSkipFeatureSweep(e.target.checked)} />
                Skip sweep (use cached)
              </label>
            </div>

            {/* HPO */}
            <div style={rowStyle}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-primary)", letterSpacing: "0.06em", minWidth: 100 }}>HPO</span>
              <label style={labelStyle}>Sampler: <select value={store.hpoSampler} onChange={(e) => store.setHpoSampler(e.target.value)} style={selectStyle}>
                <option value="tpe">TPE</option><option value="random">Random</option><option value="grid">Grid</option>
              </select></label>
              <label style={labelStyle}>CV Blocks: <input type="number" value={store.cvBlocks} onChange={(e) => store.setCvBlocks(Number(e.target.value))} min={2} max={10} style={inputStyle} /></label>
              <span style={{ fontSize: 9, color: "var(--color-text-dim)" }}>Trial budgets auto-detected from model family</span>
            </div>

            {/* Assembly */}
            <div style={rowStyle}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-primary)", letterSpacing: "0.06em", minWidth: 100 }}>Assembly</span>
              <label style={labelStyle}>Top-K: <input type="number" value={store.committeeTopK} onChange={(e) => store.setCommitteeTopK(Number(e.target.value))} min={1} max={5} style={inputStyle} /></label>
              <label style={labelStyle}>Min SR: <input type="number" value={store.committeeMinSharpe} onChange={(e) => store.setCommitteeMinSharpe(Number(e.target.value))} min={-2} max={2} step={0.1} style={{ ...inputStyle, width: 56 }} /></label>
              <label style={labelStyle}>Weights: <select value={store.committeeWeightMethod} onChange={(e) => store.setCommitteeWeightMethod(e.target.value)} style={selectStyle}>
                <option value="equal">Equal</option><option value="sharpe_proportional">Sharpe-proportional</option>
              </select></label>
            </div>

            {/* Validation + Factory */}
            <div style={rowStyle}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-primary)", letterSpacing: "0.06em", minWidth: 100 }}>Validation & Factory</span>
              <label style={labelStyle}>Train (mo): <input type="number" value={store.trainMonths} onChange={(e) => store.setTrainMonths(Number(e.target.value))} min={12} max={120} style={inputStyle} /></label>
              <label style={labelStyle}>Test (mo): <input type="number" value={store.testMonths} onChange={(e) => store.setTestMonths(Number(e.target.value))} min={1} max={12} style={inputStyle} /></label>
              <label style={labelStyle}>Proposer: <select value={store.proposer} onChange={(e) => store.setProposer(e.target.value)} style={selectStyle}>
                <option value="llm">LLM</option><option value="hybrid_llm_ucb1">LLM + UCB1</option><option value="ucb1">UCB1</option><option value="deterministic">Greedy</option>
              </select></label>
              {(store.proposer === "llm" || store.proposer === "hybrid_llm_ucb1") && (
                <label style={labelStyle}>LLM: <select value={store.llmBackend} onChange={(e) => store.setLlmBackend(e.target.value)} style={selectStyle}>
                  <option value="deepseek">DeepSeek</option><option value="ollama">Ollama</option><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option>
                </select></label>
              )}
              {(store.proposer === "hybrid_llm_ucb1" || store.proposer === "ucb1") && (
                <label style={labelStyle}>UCB c: <input type="number" value={store.ucb1ExplorationC} onChange={(e) => store.setUcb1ExplorationC(Number(e.target.value))} min={0.5} max={5} step={0.1} style={{ ...inputStyle, width: 44 }} /></label>
              )}
              <label style={labelStyle}>Iter: <input type="number" value={store.maxIterations} onChange={(e) => store.setMaxIterations(Number(e.target.value))} min={3} max={50} style={inputStyle} /></label>
              <label style={labelStyle}>Pat: <input type="number" value={store.factoryPatience} onChange={(e) => store.setFactoryPatience(Number(e.target.value))} min={3} max={15} style={inputStyle} /></label>
            </div>

            {/* Proposer comparison */}
            <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: 9, color: "var(--color-text-dim)", lineHeight: 1.5 }}>
              {({
                llm: { label: "LLM", desc: "Fresh LLM analysis per iteration. No learning between rounds. Best for qualitative reasoning.", color: "#089981" },
                hybrid_llm_ucb1: { label: "Hybrid", desc: "LLM prunes to 5 candidates, UCB1 mathematically optimizes with Sharpe feedback. Best overall.", color: "#A78BFA" },
                ucb1: { label: "UCB1", desc: "Pure multi-armed bandit. Tests all swaps as arms, selects via confidence bounds. No LLM needed.", color: "#E5A014" },
                deterministic: { label: "Greedy", desc: "Deterministic priority-ordered testing. No AI, no learning. Fastest but least adaptive.", color: "var(--color-text-dim)" },
              })[store.proposer] && (
                <span style={{ color: ({
                  llm: "#089981", hybrid_llm_ucb1: "#A78BFA", ucb1: "#E5A014", deterministic: "var(--color-text-dim)",
                })[store.proposer] }}>
                  {({ llm: "LLM", hybrid_llm_ucb1: "Hybrid", ucb1: "UCB1", deterministic: "Greedy" })[store.proposer]}:
                </span>
              )}
              <span>
                {({
                  llm: "Fresh LLM analysis per iteration. No learning between rounds.",
                  hybrid_llm_ucb1: "LLM prunes to 5 candidates, UCB1 mathematically optimizes with Sharpe feedback.",
                  ucb1: "Pure multi-armed bandit. Tests all swaps as arms, selects via confidence bounds.",
                  deterministic: "Deterministic priority-ordered testing. No AI, no learning. Fast but least adaptive.",
                })[store.proposer]}
              </span>
            </div>
          </div>

          {(store.proposer === "llm" || store.proposer === "hybrid_llm_ucb1") && (
            <ProposerArchitecture proposer={store.proposer} ucbC={store.ucb1ExplorationC} llmBackend={store.llmBackend} />
          )}

          <div style={{ marginTop: 24 }}><RunHistoryTable activeJobId={jobId} onSelect={setJobId} /></div>

          {/* Deploy button */}
          <div style={{ marginTop: 24, display: "flex", alignItems: "center", gap: 16 }}>
            <button onClick={handleDeploy} disabled={!hasModels || startMutation.isPending} style={{
              background: hasModels ? "var(--color-accent-success)" : "var(--color-text-dim)",
              border: "none", borderRadius: 4, color: "var(--color-text-inverse)",
              padding: "12px 32px", fontSize: 12, fontWeight: 600, letterSpacing: "0.08em",
              textTransform: "uppercase", cursor: hasModels ? "pointer" : "not-allowed",
              opacity: hasModels ? 1 : 0.5,
            }}>
              {startMutation.isPending ? "Starting..." : "Run Pipeline"}
            </button>
            {!hasModels && <span style={{ fontSize: 10, color: "var(--color-text-dim)" }}>Select at least one model</span>}
          </div>
          {errorMsg && (
            <div style={{ marginTop: 8, padding: "8px 16px", background: "rgba(255,77,77,0.08)", border: "1px solid rgba(255,77,77,0.25)", borderRadius: 4, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-accent-danger)" }}>
              {errorMsg}
            </div>
          )}
        </>
      )}

      {isRunning && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 24 }}>
          <FullCycleProgress jobId={jobId!} onCancel={() => cancelMutation.mutate()} />
          <RunHistoryTable activeJobId={jobId} onSelect={setJobId} />
        </div>
      )}

      {isDone && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 24 }}>
          <FullCycleProgress jobId={jobId!} onCancel={() => {}} />
          <FullCycleResults jobId={jobId!} onRunAgain={() => setJobId(null)} />
          <RunHistoryTable activeJobId={jobId} onSelect={setJobId} />
        </div>
      )}
    </div>
  );
}
