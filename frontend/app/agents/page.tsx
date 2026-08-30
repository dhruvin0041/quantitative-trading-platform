"use client";

import React, { useState } from 'react';
import { useAgentStream } from '@/hooks/useAgentStream';
import { AnalystGrid } from '@/components/agents/AnalystGrid';
import { DebateFeed } from '@/components/agents/DebateFeed';
import { RiskExecutionPanel } from '@/components/agents/RiskExecutionPanel';
import { PlayCircle, Loader2 } from 'lucide-react';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function AgentsPage() {
  const [ticker, setTicker] = useState('AAPL');
  const [isMock, setIsMock] = useState(false);
  const [isHitl, setIsHitl] = useState(true);
  
  const {
    streamStatus,
    analystReports,
    debateHistory,
    tradeProposal,
    riskAssessment,
    executionResult,
    errorMsg,
    startStream
  } = useAgentStream();

  const handleStart = () => {
    if (ticker.trim()) {
      startStream(ticker.trim().toUpperCase(), isMock);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 flex items-center justify-between border-b border-border pb-6">
          <div>
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              Multi-Agent Engine
            </h1>
            <p className="text-muted-foreground mt-1">Real-time LLM-powered quantitative consensus</p>
          </div>
          
          <div className="flex items-center space-x-3 bg-card p-2 rounded-lg border border-border">
            <label className="flex items-center space-x-2 text-sm text-muted-foreground mr-2 cursor-pointer">
              <input 
                type="checkbox" 
                checked={isHitl}
                onChange={(e) => setIsHitl(e.target.checked)}
                className="rounded border-border bg-input text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>HITL Approval</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-muted-foreground mr-2 cursor-pointer border-l border-border pl-4">
              <input 
                type="checkbox" 
                checked={isMock}
                onChange={(e) => setIsMock(e.target.checked)}
                className="rounded border-border bg-input text-blue-500 focus:ring-blue-500 focus:ring-offset-slate-900"
              />
              <span>Dry Run</span>
            </label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="Ticker (e.g. AAPL)"
              className="bg-input border border-border rounded px-3 py-2 text-white w-32 focus:outline-none focus:border-blue-500 uppercase"
              disabled={streamStatus === 'streaming'}
            />
            <button
              onClick={handleStart}
              disabled={streamStatus === 'streaming'}
              className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 text-white px-4 py-2 rounded font-medium flex items-center transition-colors"
            >
              {streamStatus === 'streaming' ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing</>
              ) : (
                <><PlayCircle className="w-4 h-4 mr-2" /> Start Pipeline</>
              )}
            </button>
            <ThemeToggle />
          </div>
        </header>

        {errorMsg && (
          <div className="bg-red-900/50 border border-red-500 text-red-200 p-4 rounded-lg mb-6">
            <strong>Error:</strong> {errorMsg}
          </div>
        )}

        {/* Phase 1: Data Gathering & Analysis */}
        <div className="mb-2">
          <h2 className="text-xl font-semibold mb-4 text-foreground">Phase 1: Analyst Briefings</h2>
          <AnalystGrid reports={analystReports} />
        </div>

        {/* Phase 2: Synthesis & Debate */}
        <div className="mb-2">
          <h2 className="text-xl font-semibold mb-4 text-foreground">Phase 2: Researcher Debate</h2>
          <DebateFeed history={debateHistory} />
        </div>

        {/* Phase 3: Execution & Risk Management */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 text-foreground">Phase 3: Risk & Execution</h2>
          <RiskExecutionPanel 
            proposal={tradeProposal} 
            risk={riskAssessment} 
            execution={executionResult} 
            hitlMode={isHitl}
          />
        </div>

      </div>
    </div>
  );
}
