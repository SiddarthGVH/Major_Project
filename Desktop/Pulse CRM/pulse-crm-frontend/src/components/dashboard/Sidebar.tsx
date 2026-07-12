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
      className={`bg-brand-sidebar text-slate-200 h-screen sticky top-0 flex flex-col justify-between transition-all duration-200 z-40 shrink-0 border-r border-brand-accent-hover/30 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="h-16 flex items-center px-4 border-b border-white/10">
          <div className="flex items-center space-x-3 overflow-hidden">
            {/* Pulse Wave Icon with consistent 1.75 stroke and dark spruce theme */}
            <div className="h-8 w-8 rounded-lg bg-white/10 flex items-center justify-center shrink-0 border border-white/10">
              <svg className="h-4.5 w-4.5 text-white animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            {!collapsed && (
              <span className="font-bold text-white text-lg tracking-wide uppercase font-sans">
                PULSE
              </span>
            )}
          </div>
        </div>

        {/* Menu Items with Light Styles */}
        <nav className="mt-4 px-2 space-y-1 overflow-y-auto max-h-[calc(100vh-14rem)] scrollbar-thin scrollbar-thumb-white/10">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab.toLowerCase() === item.name.toLowerCase();
            return (
              <button
                key={item.name}
                onClick={() => setActiveTab(item.name.toLowerCase())}
                className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-bold transition-all duration-200 cursor-pointer group border-l-2 relative ${
                  isActive 
                    ? 'bg-white/10 text-white border-white shadow-sm/5' 
                    : 'hover:bg-white/5 text-slate-350 hover:text-white border-l-2 border-transparent'
                }`}
                title={collapsed ? item.name : undefined}
              >
                <Icon 
                  className={`h-4.5 w-4.5 shrink-0 transition-colors ${
                    isActive ? 'text-white' : 'text-slate-400 group-hover:text-white'
                  }`}
                  strokeWidth={1.75}
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

      {/* User Footer Profile */}
      <div className="p-4 border-t border-white/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className="h-8 w-8 rounded-full bg-slate-200 overflow-hidden shrink-0 border border-white/10">
              <img 
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=80&fit=crop&q=80" 
                alt="Alex Johnson Profile" 
                className="h-full w-full object-cover"
              />
            </div>
            {!collapsed && (
              <div className="text-left overflow-hidden">
                <p className="text-xs font-bold text-white truncate leading-tight">Alex Johnson</p>
                <p className="text-[10px] text-slate-300 truncate mt-0.5 font-medium">Sales Manager</p>
              </div>
            )}
          </div>
          {!collapsed && (
            <button className="text-slate-400 hover:text-white transition-colors cursor-pointer">
              <ChevronDown className="h-4.5 w-4.5" strokeWidth={1.75} />
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
