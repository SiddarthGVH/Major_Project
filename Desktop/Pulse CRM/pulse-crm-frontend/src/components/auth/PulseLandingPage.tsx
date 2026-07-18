'use client';

import React, { useState } from 'react';
import { 
  Mail, 
  ArrowRight, 
  ShieldCheck, 
  Sparkles, 
  Layers, 
  Activity, 
  Loader2,
  X,
  LayoutDashboard,
  CheckCircle2,
  Lock,
  ChevronRight,
  TrendingUp,
  Award,
  Zap,
  Users
} from 'lucide-react';

interface PulseLandingPageProps {
  onLogin: () => void;
}

export default function PulseLandingPage({ onLogin }: PulseLandingPageProps) {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // Interactive Product Suite Tab State
  const [activeTab, setActiveTab] = useState(0);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      setIsModalOpen(false);
      onLogin();
    }, 1200); // Simulated loading
  };

  const handleGoogleLogin = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      setIsModalOpen(false);
      onLogin();
    }, 1200);
  };

  // Zoho-style product tabs definitions
  const productSuites = [
    {
      title: 'Sales & Pipeline',
      icon: LayoutDashboard,
      badge: 'Revenue Acceleration',
      heading: 'Manage deals and automate your sales stages',
      desc: 'Move deals through customizable funnel columns, coordinate sales reps on a live revenue leaderboard, and instantly log updates.',
      features: [
        'Interactive kanban deal pipeline boards',
        'Sales rep revenue leaderboards',
        'Win/loss analysis reason codes'
      ],
      color: 'from-violet-500 to-indigo-600',
      mockup: (
        <img src="/pulse_tab_sales.png" alt="Sales Pipeline board" className="w-full h-full rounded-xl object-cover shadow-lg border border-slate-200" />
      )
    },
    {
      title: 'Smart Emails',
      icon: Mail,
      badge: 'Unified Communications',
      heading: 'Integrated inbox syncing and automated drafts',
      desc: 'Keep client communications linked natively to deals. Sync thread timelines automatically and utilize templates to reach contacts faster.',
      features: [
        'Real-time background Gmail syncing',
        'Thread timeline logging by deal and contact',
        'Templates and rapid-fire replies'
      ],
      color: 'from-blue-500 to-sky-600',
      mockup: (
        <div className="w-full h-full p-4 flex flex-col justify-between bg-slate-900 text-white rounded-xl shadow-lg border border-slate-700">
          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
            <span className="text-[10px] text-slate-400 font-extrabold uppercase">Communication Timeline</span>
            <span className="text-[9px] text-emerald-400 font-extrabold">Active Sync</span>
          </div>
          <div className="space-y-2 mt-3 flex-1 overflow-hidden">
            {[
              { from: 'Alex Johnson', sub: 'Proposal revisions finalized', time: '10m ago', color: 'bg-violet-500' },
              { from: 'Initech Inc', sub: 'Inquiry regarding migration SLAs', time: '1h ago', color: 'bg-emerald-500' },
              { from: 'Acme Corp', sub: 'Contract signed & dispatched', time: '3h ago', color: 'bg-pink-500' }
            ].map((mail, i) => (
              <div key={i} className="bg-slate-800/50 p-2 rounded-lg border border-slate-700/30 flex justify-between items-center text-[10px]">
                <div className="flex items-center space-x-2 min-w-0 flex-1 pr-2">
                  <span className={`h-5 w-5 rounded-full flex items-center justify-center text-[8px] font-black text-white shrink-0 ${mail.color}`}>
                    {mail.from[0]}
                  </span>
                  <div className="min-w-0">
                    <span className="font-black text-white truncate block">{mail.from}</span>
                    <span className="text-[9px] text-slate-400 truncate block mt-0.5">{mail.sub}</span>
                  </div>
                </div>
                <span className="text-[8px] text-slate-500 shrink-0 font-extrabold">{mail.time}</span>
              </div>
            ))}
          </div>
        </div>
      )
    },
    {
      title: 'AI Co-pilot',
      icon: Sparkles,
      badge: 'Sales Intelligence',
      heading: 'Automated deal forecasts and priority insights',
      desc: 'Get smart suggestions, draft custom client responses, look up deal progress, and compute forecasting models with a floating Copilot.',
      features: [
        'Interactive AI chat prompt actions',
        'Automated priority rankings for leads',
        'Live summary generators for deals timeline'
      ],
      color: 'from-purple-500 to-pink-600',
      mockup: (
        <img src="/pulse_tab_copilot.png" alt="AI Co-pilot conversation" className="w-full h-full rounded-xl object-cover shadow-lg border border-slate-200" />
      )
    },
    {
      title: 'Advanced Analytics',
      icon: Activity,
      badge: 'Real-time Telemetry',
      heading: 'Git-style logs and conversion tracking',
      desc: 'Visualize team contributions with contribution activity matrices. Trace conversion metrics across your pipeline step-by-step.',
      features: [
        'Sales rep activity heatmap widget',
        'Stepped radial progress rings chart',
        'Custom report builder dashboard grids'
      ],
      color: 'from-emerald-500 to-teal-600',
      mockup: (
        <img src="/pulse_tab_analytics.png" alt="Advanced Analytics graphs" className="w-full h-full rounded-xl object-cover shadow-lg border border-slate-200" />
      )
    }
  ];

  return (
    <div className="min-h-screen w-full bg-[url('/pulse_3d_bg.png')] bg-cover bg-center bg-no-repeat bg-attachment-fixed flex flex-col overflow-x-hidden text-slate-950 font-sans relative">
      {/* Soft overlay for legibility and theme mixing */}
      <div className="absolute inset-0 bg-slate-900/5 backdrop-blur-xs z-0" />
      <div className="absolute top-10 left-[10%] w-96 h-96 bg-brand-accent/5 blur-[120px] rounded-full pointer-events-none z-0"></div>
      <div className="absolute top-[40%] right-[5%] w-80 h-80 bg-brand-blue/5 blur-[100px] rounded-full pointer-events-none z-0"></div>
      
      {/* 1. Header Navigation Bar */}
      <header className="sticky top-0 bg-white/95 backdrop-blur-md border-b border-slate-200/80 z-40 h-16 w-full flex items-center justify-between px-6 md:px-12 select-none shadow-sm/5">
        <div className="flex items-center space-x-2.5">
          <div className="h-8 w-8 rounded-lg bg-brand-accent flex items-center justify-center shadow-md">
            <svg className="h-4.5 w-4.5 text-white animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <span className="font-sans font-black text-lg text-brand-heading tracking-wider uppercase">
            PULSE
          </span>
        </div>
        
        {/* Navigation Links */}
        <nav className="hidden md:flex items-center space-x-8 text-xs font-bold text-slate-500">
          <a href="#metrics" className="hover:text-brand-accent transition-colors">Performance</a>
          <a href="#suite" className="hover:text-brand-accent transition-colors">Unified Suite</a>
          <a href="#features" className="hover:text-brand-accent transition-colors">Features</a>
        </nav>

        {/* Action Buttons */}
        <div className="flex items-center space-x-4">
          <button 
            onClick={() => setIsModalOpen(true)}
            className="text-xs font-bold text-brand-heading hover:text-brand-accent transition-colors cursor-pointer bg-transparent border-0"
          >
            Sign In
          </button>
          <button 
            onClick={() => setIsModalOpen(true)}
            className="px-4 py-2 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold shadow-sm transition-all cursor-pointer"
          >
            Start Free Trial
          </button>
        </div>
      </header>

      {/* 2. Hero Section */}
      <section className="relative w-full py-16 md:py-24 bg-transparent flex items-center justify-center px-6 md:px-12 border-b border-slate-200/5 z-10">
        <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Column: Core Value Pitch */}
          <div className="lg:col-span-7 space-y-6 text-left">
            
            
            <h1 className="text-4xl md:text-5xl font-sans font-black tracking-tight text-brand-heading leading-tight max-w-2xl">
              The Operating System for{' '}
              <span className="relative inline-block text-brand-accent">
                Sales & CRM.
                <svg className="absolute top-[90%] left-0 w-full h-2 text-brand-accent/50" viewBox="0 0 100 10" preserveAspectRatio="none" fill="none">
                  <path d="M1 5C25 8 75 2 99 5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
              </span>
            </h1>

            <p className="text-xs md:text-sm text-slate-500 font-bold leading-relaxed max-w-xl">
              Pulse brings your sales pipelines, client communications, activity grids, and real-time AI insights into a single unified workspace. Empower your team to close deals faster and automate daily workflows effortlessly.
            </p>

            {/* CTA action buttons */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 pt-2">
              <button 
                onClick={() => setIsModalOpen(true)}
                className="px-6 py-3 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-xl text-xs font-bold shadow-md hover:shadow-lg transition-all flex items-center justify-center space-x-2 cursor-pointer"
              >
                <span>Activate Free Trial</span>
                <ArrowRight className="h-4 w-4" />
              </button>
              <button 
                onClick={() => setIsModalOpen(true)}
                className="px-6 py-3 bg-white border border-slate-300 hover:border-slate-400 text-slate-600 rounded-xl text-xs font-bold shadow-sm/5 transition-all flex items-center justify-center space-x-1.5 cursor-pointer"
              >
                <span>Watch Live Demo</span>
              </button>
            </div>

            {/* Trust bullet parameters */}
            <ul className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[10.5px] font-black text-slate-400 select-none pt-2">
              <li className="flex items-center space-x-1.5">
                <CheckCircle2 className="h-4 w-4 text-brand-accent" />
                <span>Free for 14 Days</span>
              </li>
              <li className="flex items-center space-x-1.5">
                <CheckCircle2 className="h-4 w-4 text-brand-accent" />
                <span>No Credit Card Required</span>
              </li>
              <li className="flex items-center space-x-1.5">
                <CheckCircle2 className="h-4 w-4 text-brand-accent" />
                <span>Instant Set-up</span>
              </li>
            </ul>
          </div>

          {/* Right Column: Floating 3D Hero Illustration Photo */}
          <div className="lg:col-span-5 flex justify-center lg:justify-end">
            <div className="w-full max-w-md bg-white/40 border border-white/20 backdrop-blur-md p-3.5 rounded-3xl shadow-xl flex items-center justify-center select-none relative group transition-transform hover:-translate-y-1 duration-300">
              <img src="/pulse_hero_main.png" alt="Pulse 3D Sales & CRM Hub" className="w-full h-auto rounded-2xl object-cover shadow-md" />
            </div>
          </div>

        </div>
      </section>

      {/* 3. Metrics Statistics Band (Adding real data & color variety) */}
      <section id="metrics" className="py-12 bg-transparent border-b border-slate-200/5 relative overflow-hidden select-none z-10">
        <div className="w-full max-w-6xl mx-auto px-6 md:px-12 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'Active Business Seats', val: '14,820+', desc: '+18.4% this qtr', color: 'text-violet-600', bg: 'bg-violet-500/10', icon: Users },
            { label: 'Deals Closed Natively', val: '432,050+', desc: '₹124M total value', color: 'text-blue-600', bg: 'bg-blue-500/10', icon: Award },
            { label: 'AI Priority Accuracy', val: '98.4%', desc: '1.2s response latency', color: 'text-emerald-600', bg: 'bg-emerald-500/10', icon: Sparkles },
            { label: 'Pipeline Velocity Boost', val: '3.4x', desc: 'Saves 8.2 hrs / rep / wk', color: 'text-pink-600', bg: 'bg-pink-500/10', icon: Zap }
          ].map((stat, idx) => {
            const Icon = stat.icon;
            return (
              <div key={idx} className="space-y-1.5 p-5 bg-slate-50 border border-slate-200/50 rounded-2xl relative group overflow-hidden transition-all hover:border-slate-300">
                <div className={`absolute top-0 right-0 h-10 w-10 rounded-bl-3xl ${stat.bg} flex items-center justify-center text-slate-700`}>
                  <Icon className="h-4 w-4 opacity-75" />
                </div>
                <span className="text-[10px] text-slate-400 font-extrabold uppercase tracking-wide block">{stat.label}</span>
                <span className={`text-2xl md:text-3xl font-sans font-black ${stat.color} block tracking-tight`}>{stat.val}</span>
                <span className="text-[9.5px] text-slate-500 font-bold block">{stat.desc}</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* 4. Interactive Product Suite Grid (Zoho-style App Showcase) */}
      <section id="suite" className="py-20 bg-transparent flex flex-col items-center justify-center px-6 md:px-12 border-b border-slate-200/5 z-10">
        <div className="w-full max-w-6xl space-y-12">
          
          {/* Header Title */}
          <div className="text-center space-y-3">
            <h2 className="text-3xl font-sans font-black tracking-tight text-brand-heading">
              A Unified Suite to Run Your Entire Sales Cycle
            </h2>
            <p className="text-xs md:text-sm text-slate-500 font-bold max-w-xl mx-auto leading-relaxed">
              Ditch the fragmented tools. Pulse unites everything in one seamless dashboard, from pipeline triggers to real-time AI assistance.
            </p>
          </div>

          {/* Interactive tabs navigation */}
          <div className="flex flex-wrap justify-center gap-2 pb-4 border-b border-slate-200/60 select-none">
            {productSuites.map((suite, idx) => {
              const Icon = suite.icon;
              const isActive = activeTab === idx;
              return (
                <button
                  key={idx}
                  onClick={() => setActiveTab(idx)}
                  className={`flex items-center space-x-2.5 px-4.5 py-2.5 rounded-xl text-xs font-black transition-all cursor-pointer border ${
                    isActive 
                      ? 'bg-slate-900 border-slate-900 text-white shadow-md' 
                      : 'bg-slate-50 border-slate-200 hover:border-slate-300 text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <Icon className={`h-4.5 w-4.5 ${isActive ? 'text-brand-accent' : 'text-slate-400'}`} />
                  <span>{suite.title}</span>
                </button>
              );
            })}
          </div>

          {/* Active Tab Preview Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16 items-center pt-6">
            
            {/* Left Column: Selected Feature Copy */}
            <div className="lg:col-span-6 space-y-6 text-left animate-in fade-in slide-in-from-left-2 duration-300">
              <span className={`inline-block px-3 py-0.5 rounded text-[9.5px] font-black uppercase tracking-wide bg-gradient-to-r ${productSuites[activeTab].color} text-white`}>
                {productSuites[activeTab].badge}
              </span>
              <h3 className="text-2xl font-sans font-black text-brand-heading">
                {productSuites[activeTab].heading}
              </h3>
              <p className="text-xs text-slate-500 font-bold leading-relaxed">
                {productSuites[activeTab].desc}
              </p>
              
              {/* Feature bullet list */}
              <ul className="space-y-3 pt-2 text-xs font-bold text-slate-600">
                {productSuites[activeTab].features.map((feature, i) => (
                  <li key={i} className="flex items-center space-x-3">
                    <CheckCircle2 className="h-4.5 w-4.5 text-brand-accent shrink-0" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>

              {/* Activate button */}
              <button
                onClick={() => setIsModalOpen(true)}
                className="mt-4 px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold rounded-lg flex items-center space-x-1.5 cursor-pointer shadow-sm"
              >
                <span>Activate {productSuites[activeTab].title}</span>
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>

            {/* Right Column: Selected Dynamic Preview Mockup */}
            <div className="lg:col-span-6 flex justify-center lg:justify-end animate-in fade-in slide-in-from-right-2 duration-300">
              <div className="w-full max-w-md h-64 relative flex items-center justify-center p-1 bg-slate-50 border border-slate-200/80 rounded-2xl shadow-inner">
                {productSuites[activeTab].mockup}
              </div>
            </div>

          </div>

        </div>
      </section>

      {/* 5. Visual Features Showcase Grid (Adding color, graphs, and variety) */}
      <section id="features" className="py-20 bg-transparent border-b border-slate-200/5 flex flex-col items-center justify-center px-6 md:px-12 relative overflow-hidden z-10">
        
        {/* Floating gradient blur background accent */}
        <div className="absolute top-[40%] left-[5%] w-80 h-80 bg-brand-accent/5 blur-[90px] rounded-full pointer-events-none"></div>
        
        <div className="w-full max-w-6xl space-y-12 relative z-10">
          <div className="text-center space-y-3">
            <h2 className="text-3xl font-sans font-black tracking-tight text-brand-heading">
              Engineered for Hyper-Growth Sales Teams
            </h2>
            <p className="text-xs md:text-sm text-slate-500 font-bold max-w-xl mx-auto leading-relaxed">
              Unlock maximum data visibility, automated pipelines, and intelligent assistant workflows with responsive graphing dashboards.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: 'Visual Revenue Growth',
                desc: 'Live financial forecasting and deal pacing charts dynamically updated from raw client touchpoints.',
                color: 'from-violet-500/10 to-indigo-500/10 border-violet-500/20',
                iconColor: 'text-violet-600',
                badge: 'Pacing Analytics',
                chart: (
                  <svg className="w-full h-24 mt-4 select-none" viewBox="0 0 120 40">
                    <rect x="10" y="20" width="8" height="20" rx="1.5" fill="#c7d2fe" />
                    <rect x="25" y="15" width="8" height="25" rx="1.5" fill="#a5b4fc" />
                    <rect x="40" y="8" width="8" height="32" rx="1.5" fill="#818cf8" />
                    <rect x="55" y="24" width="8" height="16" rx="1.5" fill="#6366f1" />
                    <rect x="70" y="10" width="8" height="30" rx="1.5" fill="#4f46e5" />
                    <rect x="85" y="4" width="8" height="36" rx="1.5" fill="#7957fb" />
                    <rect x="100" y="2" width="8" height="38" rx="1.5" fill="#6448dc" />
                  </svg>
                )
              },
              {
                title: 'Priority Deal Probability',
                desc: 'AI algorithms score leads and deals from 0 to 100 based on buyer engagement and response speed.',
                color: 'from-emerald-500/10 to-teal-500/10 border-emerald-500/20',
                iconColor: 'text-emerald-600',
                badge: 'Predictive Scoring',
                chart: (
                  <div className="flex items-center justify-center space-x-6 mt-4 h-24 select-none">
                    <div className="relative h-20 w-20 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 40 40">
                        <circle cx="20" cy="20" r="16" fill="transparent" stroke="#e2e8f0" strokeWidth="3.5" />
                        <circle cx="20" cy="20" r="16" fill="transparent" stroke="#10B981" strokeWidth="3.5" strokeDasharray="100.5" strokeDashoffset="14" strokeLinecap="round" />
                      </svg>
                      <span className="absolute text-xs font-black text-slate-800">86%</span>
                    </div>
                    <div className="space-y-1 text-[9.5px] font-bold text-slate-500 text-left">
                      <div className="flex items-center space-x-1.5">
                        <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0"></span>
                        <span>High Conversion</span>
                      </div>
                      <div className="flex items-center space-x-1.5">
                        <span className="h-2 w-2 rounded-full bg-slate-300 shrink-0"></span>
                        <span>Industry Avg: 24%</span>
                      </div>
                    </div>
                  </div>
                )
              },
              {
                title: 'Rep Performance Tracking',
                desc: 'Leaderboards monitor reps revenue milestones and logs in real-time, encouraging friendly competition.',
                color: 'from-pink-500/10 to-rose-500/10 border-pink-500/20',
                iconColor: 'text-pink-600',
                badge: 'Milestone Tracking',
                chart: (
                  <div className="space-y-2 mt-4 h-24 flex flex-col justify-center select-none">
                    {[
                      { name: 'Alex J.', val: 'w-[90%]', rev: '₹1.2M', bg: 'bg-pink-500' },
                      { name: 'Sarah J.', val: 'w-[75%]', rev: '₹980K', bg: 'bg-rose-400' },
                      { name: 'David W.', val: 'w-[55%]', rev: '₹750K', bg: 'bg-slate-300' }
                    ].map((rep, i) => (
                      <div key={i} className="flex items-center justify-between text-[9px] font-extrabold text-slate-600">
                        <span className="w-12 truncate text-left">{rep.name}</span>
                        <div className="flex-1 mx-2.5 bg-slate-200 h-2 rounded-full overflow-hidden">
                          <div className={`h-full ${rep.val} ${rep.bg} rounded-full`}></div>
                        </div>
                        <span className="w-8 text-right font-black">{rep.rev}</span>
                      </div>
                    ))}
                  </div>
                )
              }
            ].map((feature, idx) => (
              <div key={idx} className={`bg-gradient-to-br ${feature.color} border border-slate-200/60 rounded-3xl p-6 shadow-sm hover:shadow-md transition-all flex flex-col justify-between h-[300px]`}>
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <span className={`text-[9px] font-black uppercase tracking-wider ${feature.iconColor} px-2.5 py-0.5 rounded-full bg-white border border-slate-200/50`}>
                      {feature.badge}
                    </span>
                  </div>
                  <h3 className="text-sm font-black text-brand-heading text-left">{feature.title}</h3>
                  <p className="text-[11px] text-slate-500 font-bold leading-normal mt-2 text-left">{feature.desc}</p>
                </div>
                <div className="w-full shrink-0">
                  {feature.chart}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>


      {/* 7. Footer */}
      <footer className="bg-slate-950 text-slate-500 py-10 px-6 md:px-12 select-none border-t border-slate-900 z-10">
        <div className="w-full max-w-6xl flex flex-col md:flex-row items-center justify-between gap-6 text-[10px] font-bold">
          <span>&copy; {new Date().getFullYear()} Pulse CRM Inc. All rights reserved.</span>
          <div className="flex space-x-6">
            <a href="#" className="hover:text-slate-400 transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-slate-400 transition-colors">Terms of Service</a>
            <a href="#" className="hover:text-slate-400 transition-colors">Security Standards</a>
          </div>
          <span>Powered by <span className="text-brand-heading">Kalnet</span></span>
        </div>
      </footer>

      {/* 8. Authentic Glassmorphic Login Modal Dialog */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop blur overlay */}
          <div 
            onClick={() => setIsModalOpen(false)}
            className="absolute inset-0 bg-slate-950/40 backdrop-blur-sm cursor-pointer"
          />
          
          {/* Modal Container Card */}
          <div className="w-full max-w-md bg-white border border-brand-border-purple/20 rounded-3xl p-8 shadow-2xl flex flex-col justify-between text-brand-text relative z-10 animate-in zoom-in-95 duration-200">
            {/* Close trigger */}
            <button 
              onClick={() => setIsModalOpen(false)}
              className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors cursor-pointer border-0 bg-transparent"
              aria-label="Close modal"
            >
              <X className="h-4 w-4" />
            </button>

            {/* Header titles */}
            <div className="text-left mb-6">
              <h2 className="font-sans text-2xl font-black text-brand-heading">Welcome back!</h2>
              <p className="text-[12px] text-slate-400 mt-1.5 font-bold">Login to continue to your Pulse account</p>
            </div>

            {/* Email form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-[10px] font-extrabold text-brand-heading uppercase tracking-wider mb-2">
                  Email address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-slate-400">
                    <Mail className="h-4.5 w-4.5" strokeWidth={1.75} />
                  </div>
                  <input
                    type="email"
                    required
                    disabled={isLoading}
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-brand-border-purple/35 rounded-xl text-xs text-brand-text bg-slate-50/50 placeholder-slate-400 focus:outline-none focus:border-brand-accent transition-colors shadow-sm/5"
                  />
                </div>
              </div>

              {/* Login Submit Button */}
              <button
                type="submit"
                disabled={isLoading || !email.trim()}
                className="w-full flex items-center justify-center space-x-2 py-2.5 bg-brand-accent hover:bg-brand-accent-hover disabled:opacity-50 text-white rounded-xl text-xs font-bold shadow-sm transition-all cursor-pointer"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4.5 w-4.5 animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  <>
                    <span>Login</span>
                    <ArrowRight className="h-4.5 w-4.5" />
                  </>
                )}
              </button>
            </form>

            {/* Divider line */}
            <div className="relative flex items-center my-6">
              <div className="flex-grow border-t border-brand-border-purple/15"></div>
              <span className="flex-shrink mx-4 text-[10px] font-black text-slate-400 uppercase tracking-widest">or</span>
              <div className="flex-grow border-t border-brand-border-purple/15"></div>
            </div>

            {/* Continue with Google */}
            <button
              onClick={handleGoogleLogin}
              disabled={isLoading}
              className="w-full flex items-center justify-center space-x-2.5 py-2.5 border border-brand-border-purple/25 hover:border-brand-border-purple hover:bg-slate-50 rounded-xl text-xs font-bold text-brand-text/80 transition-all cursor-pointer shadow-sm/5 bg-white"
            >
              {/* Google SVG Logo */}
              <svg className="h-4.5 w-4.5" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                />
              </svg>
              <span>Continue with Google</span>
            </button>

            {/* Security shield badge */}
            <div className="flex items-center justify-center space-x-1.5 mt-6 text-[10px] font-bold text-slate-400">
              <ShieldCheck className="h-4 w-4 text-emerald-600 shrink-0" />
              <span>Your data is safe and secure</span>
            </div>

            {/* Footer tag */}
            <div className="text-center mt-4 text-[9.5px] font-bold text-slate-400 select-none">
              <span>Powered by </span>
              <span className="text-brand-heading">Kalnet</span>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
