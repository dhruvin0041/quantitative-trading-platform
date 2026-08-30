import React from 'react';
import { AnalystReports } from '@/hooks/useAgentStream';
import { FileText, Activity, Globe, LineChart } from 'lucide-react';

interface AnalystGridProps {
  reports: AnalystReports;
}

export function AnalystGrid({ reports }: AnalystGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
      <Card title="Fundamentals Analyst" icon={<FileText className="w-5 h-5 text-blue-400" />} data={reports.fundamentals} />
      <Card title="Sentiment Analyst" icon={<Activity className="w-5 h-5 text-purple-400" />} data={reports.sentiment} />
      <Card title="News Analyst" icon={<Globe className="w-5 h-5 text-green-400" />} data={reports.news} />
      <Card title="Technical Analyst" icon={<LineChart className="w-5 h-5 text-orange-400" />} data={reports.technical} />
    </div>
  );
}

function Card({ title, icon, data }: { title: string, icon: React.ReactNode, data?: { text: string; conviction: number } }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col h-full shadow-lg">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          {icon}
          <h3 className="text-slate-200 font-semibold">{title}</h3>
        </div>
        {data && (
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-slate-400">{data.conviction}%</span>
            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${data.conviction > 85 ? 'bg-emerald-500' : data.conviction > 60 ? 'bg-amber-500' : 'bg-slate-500'}`}
                style={{ width: `${data.conviction}%` }}
              />
            </div>
          </div>
        )}
      </div>
      <div className="flex-1 overflow-y-auto max-h-48 text-slate-400 text-sm whitespace-pre-wrap">
        {data?.text ? data.text : <span className="text-slate-600 italic">Waiting for analysis...</span>}
      </div>
    </div>
  );
}
