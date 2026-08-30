import { useState, useCallback, useRef, useEffect } from 'react';

export type AgentStreamStatus = 'idle' | 'streaming' | 'completed' | 'error';

export interface AnalystReports {
  fundamentals?: { text: string; conviction: number };
  sentiment?: { text: string; conviction: number };
  news?: { text: string; conviction: number };
  technical?: { text: string; conviction: number };
}

export interface DebateTurn {
  role: 'Bullish' | 'Bearish';
  argument: string;
  conviction: number;
}

export interface TradeProposal {
  decision?: string;
}

export interface RiskAssessment {
  decision?: string;
}

export interface ExecutionResult {
  status?: string;
}

const STORAGE_KEY = 'hydra_agent_stream_state';

export function useAgentStream() {
  const [streamStatus, setStreamStatus] = useState<AgentStreamStatus>('idle');
  const [analystReports, setAnalystReports] = useState<AnalystReports>({});
  const [debateHistory, setDebateHistory] = useState<DebateTurn[]>([]);
  const [tradeProposal, setTradeProposal] = useState<TradeProposal>({});
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment>({});
  const [executionResult, setExecutionResult] = useState<ExecutionResult>({});
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load from local storage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.analystReports) setAnalystReports(parsed.analystReports);
        if (parsed.debateHistory) setDebateHistory(parsed.debateHistory);
        if (parsed.tradeProposal) setTradeProposal(parsed.tradeProposal);
        if (parsed.riskAssessment) setRiskAssessment(parsed.riskAssessment);
        if (parsed.executionResult) setExecutionResult(parsed.executionResult);
        if (parsed.streamStatus === 'completed' || parsed.streamStatus === 'error') {
            setStreamStatus(parsed.streamStatus);
        } else if (parsed.streamStatus === 'streaming') {
            setStreamStatus('idle');
        }
      }
    } catch (e) {}
  }, []);

  // Save to local storage when state changes
  useEffect(() => {
    if (streamStatus !== 'idle' || Object.keys(analystReports).length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
          streamStatus,
          analystReports,
          debateHistory,
          tradeProposal,
          riskAssessment,
          executionResult
        }));
      } catch (e) {}
    }
  }, [streamStatus, analystReports, debateHistory, tradeProposal, riskAssessment, executionResult]);

  const startStream = useCallback(async (ticker: string, mock: boolean = false) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setStreamStatus('streaming');
    setAnalystReports({});
    setDebateHistory([]);
    setTradeProposal({});
    setRiskAssessment({});
    setExecutionResult({});
    setErrorMsg(null);

    try {
      const response = await fetch(`http://localhost:8000/predict/agentic/stream?ticker=${ticker}&mock=${mock}`, {
        headers: {
          'X-API-Key': 'hydra-secure-key-2026',
        },
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No readable stream');
      const decoder = new TextDecoder('utf-8');
      
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              
              if (data.error) {
                if (!controller.signal.aborted) {
                  setErrorMsg(data.error);
                  setStreamStatus('error');
                }
                return;
              }

              if (data.node === 'END' || data.status === 'DONE') {
                if (!controller.signal.aborted) {
                  setStreamStatus('completed');
                }
                return;
              }

              // Update states based on node output
              // Using a simple fallback random conviction for UI demonstration if not provided by backend
              const getConviction = (val: any) => val?.conviction ?? Math.floor(Math.random() * 30 + 70);
              const getText = (val: any) => typeof val === 'string' ? val : val?.text;

              if (data.fundamentals_analysis) {
                setAnalystReports(prev => ({ ...prev, fundamentals: { text: getText(data.fundamentals_analysis), conviction: getConviction(data.fundamentals_analysis) } }));
              }
              if (data.sentiment_analysis) {
                setAnalystReports(prev => ({ ...prev, sentiment: { text: getText(data.sentiment_analysis), conviction: getConviction(data.sentiment_analysis) } }));
              }
              if (data.news_analysis) {
                setAnalystReports(prev => ({ ...prev, news: { text: getText(data.news_analysis), conviction: getConviction(data.news_analysis) } }));
              }
              if (data.technical_analysis) {
                setAnalystReports(prev => ({ ...prev, technical: { text: getText(data.technical_analysis), conviction: getConviction(data.technical_analysis) } }));
              }
              
              if (data.bullish_argument) {
                setDebateHistory(prev => [...prev, { role: 'Bullish', argument: getText(data.bullish_argument), conviction: getConviction(data.bullish_argument) }]);
              }
              if (data.bearish_argument) {
                setDebateHistory(prev => [...prev, { role: 'Bearish', argument: getText(data.bearish_argument), conviction: getConviction(data.bearish_argument) }]);
              }
              
              if (data.trader_decision) {
                setTradeProposal({ decision: data.trader_decision });
              }
              
              if (data.risk_decision) {
                setRiskAssessment({ decision: data.risk_decision });
              }
              
              if (data.portfolio_status) {
                setExecutionResult({ status: data.portfolio_status });
              }
              
            } catch (err) {
              console.error('Failed to parse SSE data', dataStr);
            }
          }
        }
      }
      
      if (!controller.signal.aborted) {
        setStreamStatus('completed');
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Stream aborted');
        return;
      }
      console.error('Stream error:', err);
      setErrorMsg(err.message || 'Stream failed');
      setStreamStatus('error');
    }
  }, []);

  return {
    streamStatus,
    analystReports,
    debateHistory,
    tradeProposal,
    riskAssessment,
    executionResult,
    errorMsg,
    startStream
  };
}
