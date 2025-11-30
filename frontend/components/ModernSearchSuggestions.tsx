import React from 'react';

interface ModernSearchSuggestionsProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export default function ModernSearchSuggestions({ suggestions, onSelect }: ModernSearchSuggestionsProps) {
  if (!suggestions || suggestions.length === 0) return null;

  return (
    <div className="bg-zinc-900 bg-opacity-98 backdrop-blur-xl border border-zinc-700 border-opacity-60 rounded-2xl shadow-[0_20px_60px_rgba(0,0,0,0.8)] overflow-hidden">
      <div className="max-h-80 overflow-y-auto">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSelect(suggestion)}
            className="w-full text-left px-5 py-3.5 text-white text-sm hover:bg-gradient-to-r hover:from-orange-500/20 hover:to-purple-500/20 transition-all duration-150 border-b border-zinc-800 border-opacity-40 last:border-b-0 flex items-center gap-3 group"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-orange-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <span className="flex-1 group-hover:translate-x-1 transition-transform duration-150">{suggestion}</span>
            <svg className="w-4 h-4 text-stone-500 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        ))}
      </div>
    </div>
  );
}
