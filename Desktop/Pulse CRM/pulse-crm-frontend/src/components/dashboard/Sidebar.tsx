'use client';

import React from 'react';
import { 
  LayoutDashboard, 
  Users, 
  Contact, 
  Building2, 
  Briefcase, 
  Activity, 
  Layers, 
  Calendar, 
  CheckSquare, 
  Mail, 
  BarChart3, 
  Sparkles, 
  GitBranch, 
  Settings,
  ChevronDown
} from 'lucide-react';
import Link from 'next/link';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

export default function Sidebar({ activeTab, setActiveTab, collapsed }: SidebarProps) {
  const menuItems = [
    { name: 'Dashboard', icon: LayoutDashboard },
    { name: 'Leads', icon: Users },
    { name: 'Contacts', icon: Contact },
    { name: 'Companies', icon: Building2 },
    { name: 'Deals', icon: Briefcase },
    { name: 'Activities', icon: Activity },
    { name: 'Pipeline', icon: Layers },
    { name: 'Calendar', icon: Calendar },
    { name: 'Tasks', icon: CheckSquare },
    { name: 'Emails', icon: Mail },
    { name: 'Reports', icon: BarChart3 },
    { name: 'AI Insights', icon: Sparkles },
    { name: 'Workflows', icon: GitBranch },
    { name: 'Settings', icon: Settings },
  ];

  return (
    <aside 
      className={`bg-white text-brand-text min-h-screen flex flex-col justify-between transition-all duration-200 z-40 shrink-0 border-r border-slate-100 shadow-[4px_0_20px_rgba(0,0,0,0.03)] ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="h-16 flex items-center px-4 border-b border-slate-100">
          <div className="flex items-center space-x-3 overflow-hidden">
            {/* Pulse Wave Icon styled in Medium Purple */}
            <div className="h-8 w-8 rounded-lg bg-brand-accent flex items-center justify-center shrink-0 border border-brand-border-purple/30">
              <svg className="h-4.5 w-4.5 text-white animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            {!collapsed && (
              <span className="font-extrabold text-brand-text text-lg tracking-wide uppercase font-sans">
                PULSE
              </span>
            )}
          </div>
        </div>

        {/* Menu Items - Very Dark Blue Text (#00004f) on White Background */}
        <nav className="mt-4 px-2 space-y-1 overflow-y-auto max-h-[calc(100vh-14rem)] scrollbar-thin scrollbar-thumb-brand-border-purple/25">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab.toLowerCase() === item.name.toLowerCase();
            return (
              <button
                key={item.name}
                onClick={() => setActiveTab(item.name.toLowerCase())}
                className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-bold transition-all duration-200 cursor-pointer group border-l-4 relative ${
                  isActive 
                    ? 'bg-brand-secondary-accent/15 text-brand-accent border-brand-secondary-accent shadow-sm/5 font-extrabold' 
                    : 'hover:bg-slate-50 text-brand-text/80 hover:text-brand-text border-l-4 border-transparent'
                }`}
                title={collapsed ? item.name : undefined}
              >
                <Icon 
                  className={`h-4.5 w-4.5 shrink-0 transition-colors ${
                    isActive ? 'text-brand-heading' : 'text-brand-text/70 group-hover:text-brand-text'
                  }`}
                  strokeWidth={2}
                />
                {!collapsed && <span className="tracking-wide">{item.name}</span>}
                {collapsed && (
                  <div className="absolute left-full ml-3 px-2 py-1 bg-slate-900 text-white text-[10px] rounded opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-200 z-50 whitespace-nowrap">
                    {item.name}
                  </div>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* User Footer Profile - High legibility Very Dark Blue text */}
      <div className="p-4 border-t border-slate-100">
        <button 
          type="button"
          onClick={() => setActiveTab('profile')}
          className="flex items-center justify-between w-full text-left cursor-pointer hover:bg-slate-50 p-1 rounded-lg transition-colors"
        >
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className="h-8 w-8 rounded-full bg-slate-100 overflow-hidden shrink-0 border border-slate-200">
              <img 
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=80&fit=crop&q=80" 
                alt="Alex Johnson Profile" 
                className="h-full w-full object-cover"
              />
            </div>
            {!collapsed && (
              <div className="text-left overflow-hidden">
                <p className="text-xs font-extrabold text-brand-text truncate leading-tight">Alex Johnson</p>
                <p className="text-[10px] text-brand-text/75 truncate mt-0.5 font-bold">Sales Manager</p>
              </div>
            )}
          </div>
          {!collapsed && (
            <button className="text-brand-text/70 hover:text-brand-text transition-colors cursor-pointer">
              <ChevronDown className="h-4.5 w-4.5" strokeWidth={2} />
            </button>
          )}
        </button>
        {!collapsed && (
          <div className="mt-3">
            <Link 
              href="/blueprint" 
              className="inline-block text-[9px] uppercase tracking-wider font-extrabold text-center text-brand-text hover:text-white border border-brand-border-purple/30 hover:bg-brand-accent bg-brand-secondary-accent/10 px-2 py-1.5 rounded-lg transition-all w-full"
            >
              Developer Docs
            </Link>
          </div>
        )}
      </div>
    </aside>
  );
}
