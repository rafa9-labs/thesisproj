import { useState } from "react";
import { Clipboard, Check } from "lucide-react";

interface Props {
  text: string | null;
}

export function BacktestSummary({ text }: Props) {
  const [copied, setCopied] = useState(false);

  if (!text) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-end">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium uppercase tracking-wider transition-all"
          style={{
            border: "1px solid #2A2E39",
            color: copied ? "#089981" : "#787B86",
            backgroundColor: "transparent",
            cursor: "pointer",
          }}
        >
          {copied ? <Check size={10} /> : <Clipboard size={10} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p
        className="text-[12px] leading-relaxed"
        style={{ color: "#E8ECF1", maxWidth: "80ch", fontFamily: "Inter, sans-serif" }}
      >
        {text}
      </p>
    </div>
  );
}
