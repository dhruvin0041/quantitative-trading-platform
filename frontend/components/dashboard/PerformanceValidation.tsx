import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Target, Trophy, TrendingUp, BarChart2 } from 'lucide-react';
import { motion } from 'framer-motion';

export function PerformanceValidation() {
  const perfMetrics = [
    { label: 'Sharpe Ratio', value: '2.14', description: 'Risk-adj return' },
    { label: 'Sortino Ratio', value: '3.42', description: 'Downside risk-adj' },
    { label: 'Calmar Ratio', value: '1.85', description: 'Return vs Drawdown' },
    { label: 'Profit Factor', value: '1.76', description: 'Gross Profit / Loss' },
    { label: 'Win Rate', value: '64.2%', description: 'Profitable trades' },
  ];

  return (
    <Card className="shadow-sm border-border bg-card h-full">
      <CardHeader className="pb-4">
        <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
          <Trophy className="w-4 h-4 text-primary" />
          System Validation
        </CardTitle>
        <CardDescription className="text-xs">Out-of-sample performance</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-3">
          {perfMetrics.map((metric, i) => (
            <motion.div 
              key={metric.label}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
              className="flex items-center justify-between p-2.5 rounded hover:bg-secondary/50 border border-transparent hover:border-border/50 transition-colors"
            >
              <div className="flex flex-col">
                <span className="text-sm font-semibold text-foreground">{metric.label}</span>
                <span className="text-[10px] text-muted-foreground">{metric.description}</span>
              </div>
              <span className="text-lg font-mono font-bold text-foreground">
                {metric.value}
              </span>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
