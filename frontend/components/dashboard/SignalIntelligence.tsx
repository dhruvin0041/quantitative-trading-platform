import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ChartData } from '@/types';
import { BrainCircuit, Cpu, TrendingUp, AlertTriangle, Newspaper } from 'lucide-react';
import { motion } from 'framer-motion';

interface SignalIntelligenceProps {
  data: ChartData | null;
}

export function SignalIntelligence({ data }: SignalIntelligenceProps) {
  if (!data || !data.ai_report) return null;

  const { Models, Context, Risk_Management } = data.ai_report;

  const getSignalColor = (action: string) => {
    if (action === 'BUY') return 'text-[var(--signal-buy)]';
    if (action === 'SELL') return 'text-[var(--signal-sell)]';
    return 'text-[var(--signal-hold)]';
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="grid grid-cols-1 md:grid-cols-3 gap-6"
      data-tour="intelligence"
    >
      <Card className="shadow-sm border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2 text-foreground">
            <Cpu className="w-4 h-4 text-primary" />
            Model Consensus
          </CardTitle>
          <CardDescription className="text-xs">Ensemble predictions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-md bg-secondary/50 border border-border/50">
            <span className="text-sm font-medium text-muted-foreground">LSTM Deep Learning</span>
            <div className="flex items-center gap-3">
              <span className={`text-sm font-bold ${getSignalColor(Models.Primary_Deep_Learning.Suggested_Action)}`}>
                {Models.Primary_Deep_Learning.Suggested_Action}
              </span>
              <span className="text-sm font-mono bg-background px-2 py-0.5 rounded border border-border">
                {Models.Primary_Deep_Learning.Confidence}
              </span>
            </div>
          </div>
          
          <div className="flex items-center justify-between p-3 rounded-md bg-secondary/50 border border-border/50">
            <span className="text-sm font-medium text-muted-foreground">XGBoost Agent</span>
            <div className="flex items-center gap-3">
              <span className={`text-sm font-bold ${getSignalColor(Models.Secondary_XGBoost.Suggested_Action)}`}>
                {Models.Secondary_XGBoost.Suggested_Action}
              </span>
              <span className="text-sm font-mono bg-background px-2 py-0.5 rounded border border-border">
                {Models.Secondary_XGBoost.Confidence}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="shadow-sm border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2 text-foreground">
            <AlertTriangle className="w-4 h-4 text-primary" />
            Risk & Volatility
          </CardTitle>
          <CardDescription className="text-xs">10-day forward projections</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-3 rounded-md bg-secondary/50 border border-border/50">
            <span className="text-xs text-muted-foreground uppercase tracking-widest block mb-2 font-semibold">Forecast Range</span>
            <div className="flex items-center justify-between font-mono text-sm">
              <div className="flex flex-col">
                <span className="text-muted-foreground text-xs">Low</span>
                <span className="text-[var(--signal-sell)] font-bold">${Risk_Management.Dynamic_10_Day_Range.Low.toFixed(2)}</span>
              </div>
              <div className="h-px bg-border flex-1 mx-4"></div>
              <div className="flex flex-col text-right">
                <span className="text-muted-foreground text-xs">High</span>
                <span className="text-[var(--signal-buy)] font-bold">${Risk_Management.Dynamic_10_Day_Range.High.toFixed(2)}</span>
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed italic border-l-2 border-primary/40 pl-3 py-1">
            {Risk_Management.Meta_Model_Status}
          </p>
        </CardContent>
      </Card>

      <Card className="shadow-sm border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2 text-foreground">
            <Newspaper className="w-4 h-4 text-primary" />
            Market Context
          </CardTitle>
          <CardDescription className="text-xs">NLP headline analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-foreground leading-relaxed">
            <p className="line-clamp-4">{Context.Top_Headline_Processed}</p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
