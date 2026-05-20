"use client"

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { X, ChevronRight, ChevronLeft } from 'lucide-react';

const steps = [
  {
    target: 'body',
    title: 'Welcome to Hydra',
    content: 'Experience institutional-grade quantitative trading. Let’s explore your new terminal.',
  },
  {
    target: '[data-tour="hero"]',
    title: 'Market Intelligence',
    content: 'Real-time ticker analysis with ensemble model signals and confidence scoring.',
  },
  {
    target: '[data-tour="chart"]',
    title: 'Advanced Visualization',
    content: 'High-performance charts with Bollinger clouds and AI-predicted entry/exit markers.',
  },
  {
    target: '[data-tour="intelligence"]',
    title: 'Model Consensus',
    content: 'Deep-dive into individual model contributions and NLP-driven market context.',
  },
  {
    target: '[data-tour="portfolio"]',
    title: 'Risk & Portfolio',
    content: 'Monitor PnL, active positions, and capital allocation in real-time.',
  },
  {
    target: '[data-tour="controls"]',
    title: 'Command Center',
    content: 'Use ⌘K to open the command palette. Switch themes, export reports, and more.',
  },
];

export function OnboardingTour() {
  const [open, setOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const hasVisited = localStorage.getItem('hydra-visited-v2');
    if (!hasVisited) {
      setOpen(true);
    }
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
          <Card className="shadow-2xl border-primary/20 bg-card/95 backdrop-blur-md">
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-lg font-bold flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-[10px] text-primary-foreground font-black">
                  {currentStep + 1}
                </span>
                {steps[currentStep].title}
              </CardTitle>
              <Button variant="ghost" size="icon" onClick={handleClose} className="h-8 w-8 rounded-full">
                <X className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {steps[currentStep].content}
              </p>
              <div className="flex gap-1 mt-6">
                {steps.map((_, i) => (
                  <div 
                    key={i} 
                    className={`h-1 flex-1 rounded-full transition-all duration-300 ${i === currentStep ? 'bg-primary' : 'bg-secondary'}`} 
                  />
                ))}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between pt-2">
              <Button variant="ghost" size="sm" onClick={handleClose} className="text-xs font-bold uppercase tracking-widest">
                Skip Tour
              </Button>
              <div className="flex gap-2">
                {currentStep > 0 && (
                  <Button variant="outline" size="sm" onClick={handleBack}>
                    <ChevronLeft className="h-4 w-4 mr-1" />
                    Back
                  </Button>
                )}
                <Button size="sm" onClick={handleNext}>
                  {currentStep === steps.length - 1 ? 'Finish' : 'Next'}
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Button>
              </div>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
