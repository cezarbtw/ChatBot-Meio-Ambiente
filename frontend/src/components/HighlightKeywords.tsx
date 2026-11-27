import { Fragment } from "react";

interface HighlightKeywordsProps {
  text: string;
  keywords: string[];
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function HighlightKeywords({ text, keywords }: HighlightKeywordsProps) {
  const active = keywords.map((k) => k.trim()).filter(Boolean);
  if (!text || active.length === 0) {
    return <>{text}</>;
  }

  const pattern = new RegExp(`(${active.map(escapeRegExp).join("|")})`, "gi");
  const parts = text.split(pattern);
  const lowerSet = new Set(active.map((k) => k.toLowerCase()));

  return (
    <>
      {parts.map((part, idx) =>
        lowerSet.has(part.toLowerCase()) ? (
          <mark
            key={`${idx}-${part}`}
            className="rounded bg-emerald-100 px-1 py-0.5 font-medium text-emerald-950"
          >
            {part}
          </mark>
        ) : (
          <Fragment key={idx}>{part}</Fragment>
        )
      )}
    </>
  );
}
