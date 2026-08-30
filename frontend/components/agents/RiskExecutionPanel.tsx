import React, { useState } from 'react';
import { TradeProposal, RiskAssessment, ExecutionResult } from '@/hooks/useAgentStream';
import { ShieldAlert, Crosshair, CheckCircle, XCircle, Hand } from 'lucide-react';

interface RiskExecutionPanelProps {
  proposal: TradeProposal;
  risk: RiskAssessment;
  execution: ExecutionResult;
  hitlMode?: boolean;
}

export function RiskExecutionPanel({ proposal, risk, execution, hitlMode = false }: RiskExecutionPanelProps) {
  const [hitlDecision, setHitlDecision] = useState<string | null>(null);

  const needsHitlApproval = hitlMode && execution.status && !hitlDecision;
  const finalStatus = hitlMode && hitlDecision ? hitlDecision : execution.status;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      {/* Lead Trader Proposal */}
      <div className="bg-card border border-border rounded-lg p-4 shadow-lg flex flex-col">
        <h3 className="text-foreground font-semibold mb-3 flex items-center space-x-2 border-b border-border pb-2">
          <Crosshair className="w-5 h-5 text-indigo-400" />
          <span>Lead Trader</span>
        </h3>
        <div className="text-foreground text-sm flex-1 whitespace-pre-wrap">
          {proposal.decision ? proposal.decision : <span className="text-muted-foreground italic">Awaiting debate consensus...</span>}
        </div>
      </div>

      {/* Risk Manager Assessment */}
      <div className="bg-card border border-border rounded-lg p-4 shadow-lg flex flex-col">
        <h3 className="text-foreground font-semibold mb-3 flex items-center space-x-2 border-b border-border pb-2">
          <ShieldAlert className="w-5 h-5 text-amber-500" />
          <span>Risk Manager</span>
        </h3>
        <div className="text-foreground text-sm flex-1 whitespace-pre-wrap">
          {risk.decision ? risk.decision : <span className="text-muted-foreground italic">Awaiting TimeGAN stress tests...</span>}
        </div>
      </div>

      {/* Portfolio Manager Execution */}
      <div className="bg-card border border-border rounded-lg p-4 shadow-lg flex flex-col items-center justify-center text-center">
        <h3 className="text-foreground font-semibold mb-4 w-full flex justify-center items-center space-x-2 border-b border-border pb-2">
          <span>Portfolio Manager</span>
        </h3>
        
        {!finalStatus ? (
          <span className="text-muted-foreground italic text-sm">Awaiting risk approval...</span>
        ) : needsHitlApproval ? (
          <div className="flex flex-col items-center space-y-4 w-full px-2">
            <div className="flex items-center space-x-2 text-amber-400 font-semibold animate-pulse">
              <Hand className="w-5 h-5" />
              <span>Manual Approval Required</span>
            </div>
            <div className="flex space-x-3 w-full">
              <button 
                onClick={() => setHitlDecision('APPROVED')}
                className="flex-1 bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-500 border border-emerald-500/50 py-2 rounded transition-colors font-semibold"
              >
                Approve
              </button>
              <button 
                onClick={() => setHitlDecision('VETOED')}
                className="flex-1 bg-rose-600/20 hover:bg-rose-600/40 text-rose-500 border border-rose-500/50 py-2 rounded transition-colors font-semibold"
              >
                Reject
              </button>
            </div>
          </div>
        ) : finalStatus === 'APPROVED' ? (
          <div className="flex flex-col items-center space-y-2">
            <CheckCircle className="w-12 h-12 text-green-500" />
            <span className="text-green-500 font-bold text-lg tracking-wider">EXECUTED</span>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-2">
            <XCircle className="w-12 h-12 text-red-500" />
            <span className="text-red-500 font-bold text-lg tracking-wider">VETOED</span>
          </div>
        )}
      </div>
    </div>
  );
}
