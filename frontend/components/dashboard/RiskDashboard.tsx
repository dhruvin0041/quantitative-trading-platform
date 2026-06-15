import React from 'react';
import { ShieldAlert, TrendingDown, Target, Activity, BarChart } from 'lucide-react';
import { motion } from 'framer-motion';
import { ChartData } from '@/types';
import { cn } from '@/lib/utils';

interface RiskDashboardProps {
  data: ChartData | null;
  currency?: string;
}

export function RiskDashboard({ data, currency = "$" }: RiskDashboardProps) {
  if (!data || !data.risk) return (
    <div className="flex flex-col items-center justify-center h-64 border border-border rounded-xl bg-card p-6 text-center gap-4">
      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
        <ShieldAlert className="w-6 h-6 text-muted-foreground" />
      </div>
      <div className="flex flex-col gap-1">
        <h3 className="text-[14px] font-bold text-foreground uppercase tracking-widest">No Risk Telemetry Available</h3>
        <p className="text-[12px] text-muted-foreground max-w-sm">
          Risk engine requires active execution authority or simulated portfolio data to compute VaR, CVaR, and Kelly sizing.
        </p>
      </div>
    </div>
  );

  const { risk, execution_authority } = data;
  const structuralRegime = execution_authority?.structural_regime || "UNKNOWN REGIME";

  // Position Risk Metrics
  const kellyFraction = risk.kelly_fraction || 0;
  const riskClassification = kellyFraction > 0.15 ? "HIGH" : kellyFraction > 0.05 ? "MEDIUM" : "LOW";
  const riskClassColor = riskClassification === "HIGH" ? "text-negative" : riskClassification === "MEDIUM" ? "text-warning" : "text-positive";
  const posConcentration = Math.min(100, kellyFraction * 100 * 1.5); // Mock derived

  // Portfolio Risk Metrics
  const var95 = 2.4; 
  const cvar95 = 3.1;
  const portfolioExposure = 65; // Mock
  const cashUtilization = 42; // Mock
  
  // Market Risk Metrics
  const volRegime = "EXPANSION"; // Mock
  const crowdingScore = 45; // 0-100
  const correlationRisk = 0.65;

  return (
    <div className="flex flex-col gap-6">
      
      {/* POSITION RISK */}
      <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
        <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
          <Target className="w-4 h-4 text-primary" /> Position Risk
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 pt-2">
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Risk Class</span>
            <span className={cn("text-[18px] font-black uppercase", riskClassColor)}>{riskClassification}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Kelly Fraction</span>
            <span className="text-[18px] font-mono font-bold text-foreground">{(kellyFraction * 100).toFixed(1)}%</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Rec. Allocation</span>
            <span className="text-[18px] font-mono font-bold text-foreground">{currency}{(50000 * kellyFraction).toLocaleString(undefined, {maximumFractionDigits:0})}</span>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-end">
              <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Concentration</span>
              <span className="text-[14px] font-mono font-bold text-foreground">{posConcentration.toFixed(0)}%</span>
            </div>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
               <motion.div initial={{ width: 0 }} animate={{ width: `${posConcentration}%` }} className={cn("h-full", posConcentration > 70 ? "bg-negative" : "bg-primary")} />
            </div>
          </div>
        </div>
      </div>

      {/* PORTFOLIO & MARKET RISK GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Portfolio Risk */}
        <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <BarChart className="w-4 h-4 text-primary" /> Portfolio Risk
          </h3>
          <div className="grid grid-cols-2 gap-6 pt-2">
            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-center">
                <span className="text-[12px] font-bold text-muted-foreground uppercase">VaR (95%)</span>
                <span className="text-[14px] font-mono font-bold text-negative">-{var95}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[12px] font-bold text-muted-foreground uppercase">CVaR (95%)</span>
                <span className="text-[14px] font-mono font-bold text-negative">-{cvar95}%</span>
              </div>
            </div>
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between items-end">
                  <span className="text-[11px] font-bold text-muted-foreground uppercase">Exposure</span>
                  <span className="text-[12px] font-mono font-bold text-foreground">{portfolioExposure}%</span>
                </div>
                <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
                   <motion.div initial={{ width: 0 }} animate={{ width: `${portfolioExposure}%` }} className="h-full bg-primary" />
                </div>
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex justify-between items-end">
                  <span className="text-[11px] font-bold text-muted-foreground uppercase">Cash Util.</span>
                  <span className="text-[12px] font-mono font-bold text-foreground">{cashUtilization}%</span>
                </div>
                <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
                   <motion.div initial={{ width: 0 }} animate={{ width: `${cashUtilization}%` }} className="h-full bg-primary" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Market Risk */}
        <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <Activity className="w-4 h-4 text-primary" /> Market Risk
          </h3>
          <div className="grid grid-cols-2 gap-6 pt-2">
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Market State</span>
                <span className={cn("text-[14px] font-black uppercase", structuralRegime.includes('BULL') ? "text-positive" : structuralRegime.includes('BEAR') ? "text-negative" : "text-foreground")}>
                  {structuralRegime}
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Vol Regime</span>
                <span className="text-[14px] font-bold text-warning uppercase">{volRegime}</span>
              </div>
            </div>
            
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between items-end">
                  <span className="text-[11px] font-bold text-muted-foreground uppercase">Crowding</span>
                  <span className="text-[12px] font-mono font-bold text-warning">{crowdingScore}/100</span>
                </div>
                <div className="h-1 w-full bg-gradient-to-r from-positive via-warning to-negative rounded-full overflow-hidden relative">
                   <motion.div initial={{ left: 0 }} animate={{ left: `${crowdingScore}%` }} className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_5px_white]" />
                </div>
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex justify-between items-end">
                  <span className="text-[11px] font-bold text-muted-foreground uppercase">Correlation</span>
                  <span className="text-[12px] font-mono font-bold text-foreground">{correlationRisk.toFixed(2)}</span>
                </div>
                <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
                   <motion.div initial={{ width: 0 }} animate={{ width: `${correlationRisk*100}%` }} className={cn("h-full", correlationRisk > 0.7 ? "bg-negative" : "bg-warning")} />
                </div>
              </div>
            </div>

          </div>
        </div>

      </div>

      {/* DRAWDOWN RISK INDICATOR */}
      <div className="p-4 rounded-lg bg-negative/5 border border-negative/20 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <TrendingDown className="w-4 h-4 text-negative" />
          <h3 className="text-[12px] font-bold text-negative uppercase tracking-widest">Historical Max Drawdown Risk</h3>
        </div>
        <div className="grid grid-cols-3 gap-4">
           <div className="flex flex-col">
              <span className="text-[11px] text-negative/60 uppercase font-bold mb-1">Peak Equity</span>
              <span className="text-[14px] font-mono font-bold text-foreground">{currency}{risk.peak_equity.toLocaleString()}</span>
           </div>
           <div className="flex flex-col">
              <span className="text-[11px] text-negative/60 uppercase font-bold mb-1">Trough Equity</span>
              <span className="text-[14px] font-mono font-bold text-negative">{currency}{risk.trough_equity.toLocaleString()}</span>
           </div>
           <div className="flex flex-col items-end justify-center">
              <span className="text-[20px] font-mono font-black text-negative">
                {risk.max_drawdown.toFixed(1)}%
              </span>
           </div>
        </div>
      </div>

    </div>
  );
}