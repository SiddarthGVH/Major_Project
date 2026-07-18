'use client';

import React, { useState } from 'react';
import { 
  Mail, 
  ArrowRight, 
  ShieldCheck, 
  Sparkles, 
  Layers, 
  Activity, 
  Loader2 
} from 'lucide-react';

interface PulseLandingPageProps {
  onLogin: () => void;
}

export default function PulseLandingPage({ onLogin }: PulseLandingPageProps) {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      onLogin();
    }, 1200); // Simulated 1.2s loading state
  };

  const handleGoogleLogin = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      onLogin();
    }, 1200);
  };

  return (
    <div className="min-h-screen w-full bg-[url('/pulse_login_bg.png')] bg-cover bg-center flex items-center justify-center p-4 md:p-8 relative">
      {/* Soft overlay for legibility and theme mixing */}
      <div className="absolute inset-0 bg-slate-900/5 backdrop-blur-xs z-0" />

      {/* Main Container */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16 items-center relative z-10">
        
        {/* Left Column: Brand Info & App Integrations */}
        <div className="lg:col-span-7 space-y-6 text-left p-4">
          
          {/* Logo row */}
          <div className="flex items-center space-x-3">
            <div className="h-9 w-9 rounded-xl bg-brand-accent flex items-center justify-center border border-brand-border-purple/30 shadow-md">
              <svg className="h-5 w-5 text-white animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="font-sans font-black text-xl text-brand-heading tracking-wider uppercase">
              PULSE
            </span>
          </div>

          {/* Heading */}
          <h1 className="text-4xl md:text-5xl font-sans font-black tracking-tight text-brand-heading leading-tight max-w-xl">
            Pulse makes your life{' '}
            <span className="relative inline-block text-brand-accent">
              easy and fast.
              <svg className="absolute top-[90%] left-0 w-full h-2 text-brand-accent/50" viewBox="0 0 100 10" preserveAspectRatio="none" fill="none">
                <path d="M1 5C25 8 75 2 99 5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </span>
          </h1>

          {/* Description */}
          <p className="text-xs md:text-sm text-brand-text/75 leading-relaxed font-bold max-w-xl">
            PULSE is your all-in-one productivity hub that brings your tools, tasks, and insights together — so you can focus on what matters most.
          </p>

          {/* Integrations Apps Row */}
          <div className="flex items-center space-x-4 py-2">
            {[
              {
                name: 'Gmail',
                icon: (
                  <svg className="h-5.5 w-5.5" viewBox="0 0 24 24">
                    <path fill="#EA4335" d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2z" />
                    <path fill="#34A853" d="M22 6l-10 6.25L2 6v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6z" />
                    <path fill="#FBBC05" d="M22 6L12 12.25 2 6v1.5l10 6.25 10-6.25V6z" />
                    <path fill="#4285F4" d="M12 12.25L2 6v1.5l10 6.25 10-6.25V6L12 12.25z" />
                  </svg>
                )
              },
              {
                name: 'Calendar',
                icon: (
                  <svg className="h-5.5 w-5.5" viewBox="0 0 24 24" fill="none" stroke="#4285F4" strokeWidth="2.25">
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                    <line x1="16" y1="2" x2="16" y2="6" />
                    <line x1="8" y1="2" x2="8" y2="6" />
                    <line x1="3" y1="10" x2="21" y2="10" />
                  </svg>
                )
              },
              {
                name: 'Notion',
                icon: (
                  <span className="font-extrabold text-xs text-slate-800 dark:text-slate-100">N</span>
                )
              },
              {
                name: 'Meet',
                icon: (
                  <svg className="h-5.5 w-5.5" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 7a2 2 0 0 0-2.45-1.45L16 7V5a2 2 0 0 0-2-2H2a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2l4.55 1.45A2 2 0 0 0 23 17V7z" />
                  </svg>
                )
              }
            ].map((app) => (
              <div key={app.name} className="flex flex-col items-center space-y-1">
                <div className="h-10 w-10 bg-white/95 dark:bg-slate-900/95 rounded-2xl shadow-sm/5 border border-brand-border-purple/20 flex items-center justify-center shrink-0">
                  {app.icon}
                </div>
                <span className="text-[10px] text-slate-400 font-extrabold tracking-wide">{app.name}</span>
              </div>
            ))}
          </div>

          {/* Highlights glassmorphism info blocks */}
          <div className="bg-white/70 dark:bg-slate-900/60 backdrop-blur-md border border-brand-border-purple/20 rounded-2xl p-5 space-y-4 shadow-sm/5 max-w-xl">
            {[
              {
                title: 'What is PULSE?',
                desc: 'Pulse brings your essential apps into one simple workspace and helps you get more done, effortlessly.',
                icon: Layers,
                color: 'bg-brand-blue/15 text-brand-blue border-brand-blue/20'
              },
              {
                title: 'Why use PULSE?',
                desc: 'AI-powered summaries, smart prioritization, and a clean dashboard that saves you time every day.',
                icon: Sparkles,
                color: 'bg-brand-accent/15 text-brand-accent border-brand-accent/20'
              },
              {
                title: 'Purpose of PULSE',
                desc: "Pulse's purpose is to simplify your work life by bringing clarity, speed, and focus to everything you do.",
                icon: Activity,
                color: 'bg-brand-secondary-accent/15 text-brand-secondary-accent border-brand-secondary-accent/20'
              }
            ].map((item, idx) => {
              const Icon = item.icon;
              return (
                <div key={idx} className="flex items-start space-x-3.5">
                  <div className={`mt-0.5 h-8.5 w-8.5 rounded-lg flex items-center justify-center shrink-0 border ${item.color}`}>
                    <Icon className="h-4.5 w-4.5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-black text-brand-heading">{item.title}</h4>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 font-bold leading-normal">{item.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>

        </div>

        {/* Right Column: Interactive Login Container Card */}
        <div className="lg:col-span-5 flex justify-center lg:justify-end">
          <div className="w-full max-w-md bg-white/90 dark:bg-slate-950/90 backdrop-blur-md border border-brand-border-purple/25 rounded-3xl p-8 shadow-2xl flex flex-col justify-between text-brand-text">
            
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
                    className="w-full pl-10 pr-4 py-2 border border-brand-border-purple/35 rounded-xl text-xs text-brand-text bg-slate-50/50 dark:bg-slate-900/50 placeholder-slate-400 focus:outline-none focus:border-brand-accent transition-colors shadow-sm/5"
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
              className="w-full flex items-center justify-center space-x-2.5 py-2.5 border border-brand-border-purple/25 hover:border-brand-border-purple hover:bg-slate-50 dark:hover:bg-slate-900 rounded-xl text-xs font-bold text-brand-text/80 transition-all cursor-pointer shadow-sm/5 bg-transparent"
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

      </div>
    </div>
  );
}
