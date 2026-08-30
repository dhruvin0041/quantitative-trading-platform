import React, { useRef, useEffect, useState } from 'react';
import { DebateTurn } from '@/hooks/useAgentStream';
import { TrendingUp, TrendingDown, Pause, Play } from 'lucide-react';

interface DebateFeedProps {
  history: DebateTurn[];
}

export function DebateFeed({ history }: DebateFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
      setAutoScroll(isAtBottom);
    }
  };

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, autoScroll]);

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 mb-6 shadow-lg h-80 flex flex-col relative">
      <div className="flex justify-between items-center mb-4 border-b border-slate-800 pb-2">
        <h3 className="text-slate-200 font-semibold flex items-center">
          Researcher Debate Feed
        </h3>
        <button 
          onClick={() => setAutoScroll(!autoScroll)}
          className={`text-xs px-2 py-1 rounded flex items-center space-x-1 ${autoScroll ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-800 text-slate-400'}`}
        >
          {autoScroll ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
          <span>{autoScroll ? 'Auto-scroll' : 'Paused'}</span>
        </button>
      </div>
      
      <div 
        ref={scrollRef} 
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto space-y-4 pr-2 scroll-smooth"
      >
        {history.length === 0 ? (
          <div className="text-slate-600 italic text-sm text-center mt-10">
            Debate has not started yet...
          </div>
        ) : (
          history.map((turn, index) => (
            <div key={index} className={`flex ${turn.role === 'Bullish' ? 'justify-start' : 'justify-end'}`}>
              <div className={`max-w-[80%] rounded-lg p-3 ${turn.role === 'Bullish' ? 'bg-emerald-900/30 border border-emerald-800/50' : 'bg-rose-900/30 border border-rose-800/50'}`}>
                <div className="flex items-center justify-between mb-2 border-b border-slate-700/50 pb-1">
                  <div className="flex items-center space-x-2">
                    {turn.role === 'Bullish' ? <TrendingUp className="w-4 h-4 text-emerald-400" /> : <TrendingDown className="w-4 h-4 text-rose-400" />}
                    <span className={`text-xs font-bold ${turn.role === 'Bullish' ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {turn.role} Researcher
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] font-mono text-slate-400">Conviction: {turn.conviction}%</span>
                    <div className="w-12 h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${turn.role === 'Bullish' ? 'bg-emerald-500' : 'bg-rose-500'}`}
                        style={{ width: `${turn.conviction}%` }}
                      />
                    </div>
                  </div>
                </div>
                <div className="text-slate-300 text-sm whitespace-pre-wrap leading-relaxed">
                  {turn.argument}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
