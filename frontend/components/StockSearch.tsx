"use client";

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Search, X } from 'lucide-react';
import { UniverseStock } from '@/types';
import { cn } from '@/lib/utils';

interface StockSearchProps {
  universe: UniverseStock[];
  onSelect: (ticker: string) => void;
}

export function StockSearch({ universe, onSelect }: StockSearchProps) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Debounced search for real-time filtering feel
  const [debouncedQuery, setDebouncedQuery] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 200);
    return () => clearTimeout(timer);
  }, [query]);

  const filteredStocks = useMemo(() => {
    if (!debouncedQuery.trim()) return [];
    const q = debouncedQuery.toLowerCase().trim();
    return universe
      .filter(s => s.ticker.toLowerCase().includes(q) || s.name.toLowerCase().includes(q))
      .slice(0, 8);
  }, [universe, debouncedQuery]);

  // Reset selected index when results change
  useEffect(() => {
    // eslint-disable-next-line
    setSelectedIndex(0);
  }, [filteredStocks]);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // 2. The Ctrl+K / ⌘K shortcut for the search bar (opens dropdown if focused)
    if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        e.stopPropagation(); // Disconnect from global CommandMenu listener
        setIsOpen(true);
        inputRef.current?.focus();
        return;
    }

    if (!isOpen) {
        if (e.key === "ArrowDown" || e.key === "Enter") {
            setIsOpen(true);
        }
        return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % Math.max(1, filteredStocks.length));
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + filteredStocks.length) % Math.max(1, filteredStocks.length));
        break;
      case "Enter":
        e.preventDefault();
        if (filteredStocks[selectedIndex]) {
          onSelect(filteredStocks[selectedIndex].ticker);
          setIsOpen(false);
          setQuery("");
        }
        break;
      case "Escape":
        e.preventDefault();
        setIsOpen(false);
        setQuery("");
        break;
    }
  };

  const highlightMatch = (text: string, q: string) => {
    if (!q) return text;
    // Escape special regex characters
    const escapedQ = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const parts = text.split(new RegExp(`(${escapedQ})`, 'gi'));
    return (
      <>
        {parts.map((part, i) => 
          part.toLowerCase() === q.toLowerCase() ? (
            <span key={i} style={{ color: '#E8650A' }}>{part}</span>
          ) : (
            part
          )
        )}
      </>
    );
  };

  return (
    <div className="relative w-64" ref={containerRef}>
      <div 
        className={cn(
          "h-8 bg-secondary/50 rounded-md border border-border flex items-center px-3 gap-2 cursor-text hover:bg-secondary transition-all",
          isOpen && "ring-1 ring-primary border-primary/50 bg-secondary"
        )}
        onClick={() => {
            setIsOpen(true);
            inputRef.current?.focus();
        }}
      >
        <Search className="w-4 h-4 text-muted-foreground shrink-0" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsOpen(true)}
          placeholder="Search stocks..."
          className="bg-transparent border-none outline-none text-xs font-mono w-full text-foreground placeholder:text-muted-foreground"
          spellCheck={false}
          autoComplete="off"
        />
        {query && (
          <button 
            onClick={(e) => {
                e.stopPropagation();
                setQuery("");
                inputRef.current?.focus();
            }} 
            className="hover:text-foreground text-muted-foreground transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      {isOpen && (debouncedQuery.trim() !== "") && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-md shadow-xl z-[100] overflow-hidden backdrop-blur-xl animate-in fade-in slide-in-from-top-1 duration-200">
          {filteredStocks.length > 0 ? (
            <ul className="py-1 max-h-[320px] overflow-y-auto">
              {filteredStocks.map((stock, index) => (
                <li 
                  key={stock.ticker}
                  className={cn(
                    "px-3 py-2 flex items-center justify-between cursor-pointer transition-colors border-l-2 border-transparent",
                    index === selectedIndex ? "bg-secondary border-orange-500" : "hover:bg-secondary/50"
                  )}
                  onClick={() => {
                    onSelect(stock.ticker);
                    setIsOpen(false);
                    setQuery("");
                  }}
                  onMouseEnter={() => setSelectedIndex(index)}
                >
                  <span className="font-mono font-bold text-sm shrink-0" style={{ color: '#E8650A' }}>
                    {highlightMatch(stock.ticker, debouncedQuery.trim())}
                  </span>
                  <div className="flex items-center gap-1.5 ml-4 flex-1 justify-end">
                    <span className={cn(
                      "text-[8px] font-black uppercase px-1.5 py-0.5 rounded border leading-none shrink-0",
                      stock.market === 'us' 
                        ? "bg-primary/10 text-primary border-primary/30 dark:bg-primary/10 dark:text-primary" 
                        : "bg-orange-500/10 text-orange-600 border-orange-500/30 dark:bg-orange-500/10 dark:text-orange-400"
                    )}>
                      {stock.market === 'us' ? '🇺🇸 USA' : stock.market === 'india' ? '🇮🇳 INDIA' : stock.market.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-muted-foreground truncate max-w-[120px] font-medium">
                      {highlightMatch(stock.name, debouncedQuery.trim())}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="px-3 py-4 text-center text-xs text-muted-foreground font-mono">
              No stocks found for &apos;<span className="text-foreground">{debouncedQuery}</span>&apos;
            </div>
          )}
        </div>
      )}
    </div>
  );
}
