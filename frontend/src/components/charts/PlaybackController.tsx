import { Play, Pause, SkipBack, SkipForward } from "lucide-react";

interface PlaybackControllerProps {
  currentIndex: number;
  totalBars: number;
  isPlaying: boolean;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStepForward: () => void;
  onStepBack: () => void;
  onSpeedChange: (speed: number) => void;
  onSeek: (index: number) => void;
}

const SPEEDS = [1, 2, 5, 10];

export function PlaybackController({
  currentIndex,
  totalBars,
  isPlaying,
  speed,
  onPlay,
  onPause,
  onStepForward,
  onStepBack,
  onSpeedChange,
  onSeek,
}: PlaybackControllerProps) {
  const pct = (currentIndex / Math.max(totalBars - 1, 1)) * 100;

  return (
    <div
      className="flex items-center gap-3 rounded-sm border px-3 py-2"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
    >
      <button
        onClick={onStepBack}
        disabled={currentIndex <= 0}
        className="rounded p-1 transition hover:bg-[var(--color-glass-hover)] disabled:opacity-30"
        style={{ color: "var(--color-text-secondary)" }}
        title="Step back"
      >
        <SkipBack size={14} />
      </button>

      <button
        onClick={isPlaying ? onPause : onPlay}
        className="rounded p-1.5 transition"
        style={{
          backgroundColor: isPlaying ? "rgba(239,68,68,0.15)" : "rgba(0,229,255,0.15)",
          color: isPlaying ? "var(--color-accent-danger)" : "var(--color-brand)",
        }}
        title={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
      </button>

      <button
        onClick={onStepForward}
        disabled={currentIndex >= totalBars - 1}
        className="rounded p-1 transition hover:bg-[var(--color-glass-hover)] disabled:opacity-30"
        style={{ color: "var(--color-text-secondary)" }}
        title="Step forward"
      >
        <SkipForward size={14} />
      </button>

      <div className="flex-1 mx-2">
        <input
          type="range"
          min={0}
          max={Math.max(totalBars - 1, 0)}
          value={currentIndex}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="w-full h-1 appearance-none rounded-full cursor-pointer"
          style={{
            background: `linear-gradient(to right, var(--color-brand) ${pct}%, var(--color-glass-hover) ${pct}%)`,
          }}
        />
      </div>

      <span className="text-[10px] tabular-nums" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)", minWidth: 70 }}>
        {currentIndex + 1} / {totalBars}
      </span>

      <div className="flex items-center gap-1 ml-2">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => onSpeedChange(s)}
            className="rounded px-1.5 py-0.5 text-[9px] font-medium transition"
            style={{
              backgroundColor: speed === s ? "var(--color-brand-glow)" : "transparent",
              color: speed === s ? "var(--color-brand)" : "var(--color-text-muted)",
              border: speed === s ? "1px solid var(--color-brand)" : "1px solid var(--color-glass-border)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}