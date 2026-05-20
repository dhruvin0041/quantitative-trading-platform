import React from 'react';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { ChartData, UniverseStock } from '@/types';
import { ArrowUpRight, ArrowDownRight, Minus, Activity, ShieldAlert, Cpu, Settings, MoreVertical } from 'lucide-react';
import { CommandMenu } from '@/components/CommandMenu';
import { ThemeToggle } from '@/components/ThemeToggle';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuLabel, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { exportDashboardAsImage, exportDashboardAsPdf } from '@/lib/export';

interface HeroHeaderProps {
  ticker: string;
  universe: UniverseStock[];
  chartData: ChartData | null;
  loading: boolean;
  onTickerChange: (ticker: string) => void;
}

export function HeroHeader({ ticker, universe, chartData, loading, onTickerChange }: HeroHeaderProps) {
  const stockName = universe.find(u => u.ticker === ticker)?.name || ticker;
  
  const primaryAction = chartData?.ai_report?.Models?.Primary_Deep_Learning?.Suggested_Action || "HOLD";
  const confidence = chartData?.ai_report?.Models?.Primary_Deep_Learning?.Confidence || "0%";
  
  const getSignalColor = (action: string) => {
    switch (action) {
      case 'BUY': return 'text-[var(--signal-buy)] bg-[var(--signal-buy)]/10 border-[var(--signal-buy)]/20';
      case 'SELL': return 'text-[var(--signal-sell)] bg-[var(--signal-sell)]/10 border-[var(--signal-sell)]/20';
      default: return 'text-[var(--signal-hold)] bg-[var(--signal-hold)]/10 border-[var(--signal-hold)]/20';
    }
  };

  const getSignalIcon = (action: string) => {
    switch (action) {
      case 'BUY': return <ArrowUpRight className="w-5 h-5 mr-1.5" aria-hidden="true" />;
      case 'SELL': return <ArrowDownRight className="w-5 h-5 mr-1.5" aria-hidden="true" />;
      default: return <Minus className="w-5 h-5 mr-1.5" aria-hidden="true" />;
    }
  };

  return (
    <motion.div 
      data-tour="hero"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-8 pt-4 border-b border-border/50"
    >
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="px-2.5 py-0.5 rounded-sm font-mono text-xs text-muted-foreground border-border bg-secondary/50">
            HYDRA TERMINAL v4.2
          </Badge>
          {loading && (
            <Badge variant="secondary" className="px-2 py-0.5 rounded-sm flex items-center gap-1.5 bg-primary/5 text-primary border-primary/10">
              <Activity className="w-3 h-3 animate-pulse" aria-hidden="true" />
              <span className="text-[10px] tracking-wider uppercase font-semibold">Live Feed Active</span>
            </Badge>
          )}
        </div>
        <div className="flex items-baseline gap-4 mt-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground font-sans">
            {ticker}
          </h1>
          <span className="text-xl text-muted-foreground font-medium mb-1">
            {stockName}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-4 flex-wrap" data-tour="controls">
        {chartData && !loading && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`flex items-center px-4 py-2 rounded-md border ${getSignalColor(primaryAction)} backdrop-blur-sm shadow-sm`}
            aria-label={`Current signal: ${primaryAction} with ${confidence} confidence`}
          >
            {getSignalIcon(primaryAction)}
            <div className="flex flex-col mr-4">
              <span className="text-[10px] uppercase tracking-wider font-bold opacity-70 leading-none mb-1">Ensemble Signal</span>
              <span className="text-lg font-bold leading-none">{primaryAction}</span>
            </div>
            <div className="flex flex-col pl-4 border-l border-current/20">
              <span className="text-[10px] uppercase tracking-wider font-bold opacity-70 leading-none mb-1">Confidence</span>
              <span className="text-lg font-mono font-bold leading-none">{confidence}</span>
            </div>
          </motion.div>
        )}

        <div className="flex flex-col min-w-[180px]">
          <label id="asset-label" className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1.5 ml-1">Asset Universe</label>
          <div className="relative">
            <select 
              aria-labelledby="asset-label"
              className="w-full appearance-none bg-card text-foreground font-mono font-semibold py-2.5 pl-4 pr-10 rounded-md border border-border focus:outline-none focus:ring-2 focus:ring-ring focus:border-ring shadow-sm transition-all cursor-pointer text-sm"
              value={ticker}
              onChange={(e) => onTickerChange(e.target.value)}
              disabled={loading}
            >
              {universe.map((stock) => (
                <option key={stock.ticker} value={stock.ticker}>
                  {stock.ticker} — {stock.name}
                </option>
              ))}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none text-muted-foreground">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <CommandMenu />
          <ThemeToggle />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="icon" className="w-9 h-9 border-border/50" aria-label="Export and Settings">
                <MoreVertical className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48 font-sans">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => exportDashboardAsImage('dashboard-container')}>
                Export as PNG
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => exportDashboardAsPdf('dashboard-container')}>
                Export as PDF
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Settings</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </motion.div>
  );
}

