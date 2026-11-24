import { Fragment, type ReactNode } from 'react';

function renderInlineBold(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <strong key={idx} className="font-semibold text-[#111827]">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <Fragment key={idx}>{part}</Fragment>;
  });
}

export function ProseRenderer({ content }: { content: string }) {
  const blocks = content.split(/\n{2,}/).filter((b) => b.trim().length > 0);

  return (
    <div className="space-y-3 text-[15px] leading-[1.65] text-[#1F2937] max-w-[70ch]">
      {blocks.map((block, bIdx) => {
        const lines = block.split('\n').map((l) => l.trimEnd());
        const isBulletList = lines.every((l) => /^[-•*]\s+/.test(l.trim()));
        const isOrderedList = lines.every((l) => /^\d+[.)]\s+/.test(l.trim()));

        if (isBulletList) {
          return (
            <ul key={bIdx} className="list-disc pl-5 space-y-1.5">
              {lines.map((line, lIdx) => (
                <li key={lIdx}>{renderInlineBold(line.trim().replace(/^[-•*]\s+/, ''))}</li>
              ))}
            </ul>
          );
        }

        if (isOrderedList) {
          return (
            <ol key={bIdx} className="list-decimal pl-5 space-y-1.5">
              {lines.map((line, lIdx) => (
                <li key={lIdx}>{renderInlineBold(line.trim().replace(/^\d+[.)]\s+/, ''))}</li>
              ))}
            </ol>
          );
        }

        return (
          <p key={bIdx} className="whitespace-pre-line">
            {renderInlineBold(block)}
          </p>
        );
      })}
    </div>
  );
}
