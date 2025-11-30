interface SearchSuggestionsProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export default function SearchSuggestions({ suggestions, onSelect }: SearchSuggestionsProps) {
  return (
    <div className="bg-zinc-900 bg-opacity-95 backdrop-blur-xl border border-stone-700 border-opacity-30 rounded-lg shadow-2xl overflow-hidden max-h-60 overflow-y-auto">
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          onClick={() => onSelect(suggestion)}
          className="w-full text-left px-4 py-2.5 text-white text-sm hover:bg-orange-500 hover:bg-opacity-20 transition-colors border-b border-stone-700 border-opacity-20 last:border-b-0 no-underline cursor-pointer"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
