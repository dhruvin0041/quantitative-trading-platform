"use client"

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { X, ChevronRight, ChevronLeft } from 'lucide-react';

const steps = [
  {
    target: 'body',
    title: 'Hydra Terminal v4.2',
    content: 'Welcome to your next-generation institutional AI trading command center. This terminal is configured for high-density quantitative research.',
  },
  {
    target: 'header',
    title: 'Systems & Navigation',
    content: 'The top navigation strip monitors core system health and provides access to the global command palette (⌘K).',
  },
  {
    target: '.animate-ticker',
    title: 'Live Ticker Tape',
    content: 'Real-time price feeds across your coverage universe. Market velocity is tracked with millisecond precision.',
  },
  {
    target: 'aside:first-of-type',
    title: 'Coverage Universe',
    content: 'Manage your active watchlist here. The terminal currently monitors high-liquidity S&P 500 assets and macro regime shifts.',
  },
  {
    target: '[data-tour="chart"]',
    title: 'Visual Intelligence',
    content: 'High-performance interactive charting with AI-predicted signals and Bollinger ribbon volatility clusters.',
  },
  {
    target: '[data-tour="intelligence"]',
    title: 'Ensemble Consensus',
    content: 'View real-time negotiations between specialized Alpha Agents. Decision reasoning is verified via XAI telemetry.',
  },
  {
    target: 'aside:last-of-type',
    title: 'Portfolio & Risk',
    content: 'Track paper trading performance, capital allocation, and Value-at-Risk (VaR) targets.',
  },
];

export function OnboardingTour() {
  const [open, setOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      const hasVisited = localStorage.getItem('hydra-visited-v2');
      if (!hasVisited) {
        setOpen(true);
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  const handleClose = () => {
    localStorage.setItem('hydra-visited-v2', 'true');
    setOpen(false);
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(s => s + 1);
    } else {
      handleClose();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(s => s - 1);
    }
  };

  if (!open) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/40 backdrop-blur-sm p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="max-w-md w-full"
        >
          <Card className="shadow-2xl border border-white/10 bg-black/90 backdrop-blur-2xl">
            <CardHeader className="pb-2 flex flex-row items-center justify-between border-b border-white/5">
              <CardTitle className="text-sm font-black flex items-center gap-2 uppercase tracking-widest font-mono">
                <span className="flex h-5 w-5 items-center justify-center rounded-sm bg-primary text-[10px] text-primary-foreground font-black">
                  {currentStep + 1}
                </span>
                {steps[currentStep].title}
              </CardTitle>
              <Button variant="ghost" size="icon" onClick={handleClose} className="h-6 w-6 rounded-md hover:bg-white/5">
                <X className="h-3 w-3 opacity-50" />
              </Button>
            </CardHeader>
            <CardContent className="pt-6">
              <p className="text-xs text-muted-foreground leading-relaxed font-sans">
                {steps[currentStep].content}
              </p>
              <div className="flex gap-1.5 mt-8">
                {steps.map((_, i) => (
                  <div 
                    key={i} 
                    className={`h-0.5 flex-1 transition-all duration-500 ${i === currentStep ? 'bg-primary' : 'bg-white/10'}`} 
                  />
                ))}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between pt-2 border-t border-white/5 bg-white/5 mt-4">
              <Button variant="ghost" size="sm" onClick={handleClose} className="text-[9px] font-black uppercase tracking-[0.2em] opacity-50 hover:opacity-100 transition-opacity">
                Terminate Tour
              </Button>
              <div className="flex gap-2">
                {currentStep > 0 && (
                  <Button variant="outline" size="sm" onClick={handleBack} className="h-8 text-[10px] uppercase font-bold border-white/10 bg-black/40">
                    <ChevronLeft className="h-3 w-3 mr-1" />
                    Back
                  </Button>
                )}
                <Button size="sm" onClick={handleNext} className="h-8 text-[10px] uppercase font-black bg-primary text-primary-foreground border-glow-primary">
                  {currentStep === steps.length - 1 ? 'Execute' : 'Proceed'}
                  <ChevronRight className="h-3 w-3 ml-1" />
                </Button>
              </div>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
