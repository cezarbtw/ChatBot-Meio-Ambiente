import type { NewsLabel } from "../types";

const LABEL_STYLES: Record<NewsLabel, string> = {
  Boa: "bg-emerald-50 text-emerald-800 border-emerald-200",
  Ruim: "bg-rose-50 text-rose-800 border-rose-200",
  Irrelevante: "bg-slate-100 text-slate-700 border-slate-300",
};

interface LabelBadgeProps {
  label: string;
  confidence?: number;
  isAmbiguous?: boolean;
}

export function LabelBadge({ label, confidence, isAmbiguous }: LabelBadgeProps) {
  const style =
    LABEL_STYLES[label as NewsLabel] ?? "bg-slate-100 text-slate-700 border-slate-300";

  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span
        className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${style}`}
      >
        <span>{label}</span>
        {confidence !== undefined && (
          <span className="font-mono opacity-80">{(confidence * 100).toFixed(1)}%</span>
        )}
      </span>
      {isAmbiguous && (
        <span
          title="Classificação com confiança ou margem abaixo do limiar"
          className="inline-flex items-center rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800"
        >
          Ambíguo
        </span>
      )}
    </span>
  );
}
