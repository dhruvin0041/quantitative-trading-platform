import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ShieldAlert, AlertOctagon, TrendingDown, Percent, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export function RiskDashboard() {
  // Mock data for Risk Dashboard since it's not explicitly in the backend API yet
  const riskMetrics = [
    { label: 'Value at Risk (VaR 95%)', value: '-$2,450', isDanger: true },
    { label: 'Expected Shortfall (CVaR)', value: '-$3,100', isDanger: true },
    { label: 'Portfolio Beta', value: '0.85', isDanger: false },
    { label: 'Kelly Fraction', value: '0.24', isDanger: false },
    { label: 'Max Drawdown', value: '-12.4%', isDanger: true },
  ];

  return (
    <Card className="shadow-sm border-border bg-card">
      <CardHeader className="pb-4">
        <div className="flex justify-between items-center">
          <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
            <ShieldAlert className="w-4 h-4 text-primary" />
            Risk Management
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--signal-buy)] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--signal-buy)]"></span>
            </span>
            <span className="text-[10px] uppercase font-bold tracking-widest text-muted-foreground">Kill-Switch Armed</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          {riskMetrics.map((metric, i) => (
            <motion.div 
              key={metric.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
              className="flex flex-col gap-1 p-3 bg-secondary/30 rounded border border-border/50"
            >
              <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">{metric.label}</span>
              <span className={`text-lg font-mono font-bold ${metric.isDanger ? 'text-[var(--signal-sell)]' : 'text-foreground'}`}>
                {metric.value}
              </span>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
