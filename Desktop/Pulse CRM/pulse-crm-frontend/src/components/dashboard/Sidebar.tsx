'use client';

import React from 'react';
import { 
  LayoutDashboard, 
  Users, 
  Contact, 
  Building2, 
  Layers, 
  Package,
  Activity, 
  Mail, 
  GitBranch,
  Sparkles,
  BarChart3,
  FileText,
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
  // Nested structure matching the requested sidebar layout
  const sections = [
    {
      name: 'Sales',
      items: [
        { name: 'Leads', icon: Users, tab: 'leads' },
        { name: 'Contacts', icon: Contact, tab: 'contacts' },
        { name: 'Companies', icon: Building2, tab: 'companies' },
        { name: 'Deals', icon: Layers, tab: 'deals' },
        { name: 'Products', icon: Package, tab: 'products' },
      ]
    },
    {
      name: 'Productivity',
      items: [
        { name: 'Activities', icon: Activity, tab: 'activities' },
        { name: 'Emails', icon: Mail, tab: 'emails' },
      ]
    },
    {
      name: 'Automations & Intelligence',
      items: [
        { name: 'Workflows', icon: GitBranch, tab: 'workflows' },
        { name: 'AI Insights', icon: Sparkles, tab: 'ai insights' },
      ]
    },
    {
      name: 'Data & Analytics',
      items: [
        { name: 'Reports', icon: BarChart3, tab: 'reports' },
        { name: 'Documents', icon: FileText, tab: 'documents' },
      ]
    },
    {
      name: 'Admin',
      items: [
        { name: 'Settings', icon: Settings, tab: 'settings' },
      ]
    }
  ];

  const handleTabClick = (tabName: string) => {
    setActiveTab(tabName.toLowerCase());
  };

  const isTabActive = (tabName: string) => {
    return activeTab.toLowerCase() === tabName.toLowerCase();
  };

  return (
    <aside 
      className={`bg-white h-full flex flex-col justify-between transition-all duration-200 z-45 shrink-0 border-r border-[#e2e8f0] shadow-[4px_0_20px_rgba(0,0,0,0.03)] overflow-hidden ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      <div className="flex flex-col h-full overflow-hidden">
        {/* Brand Header */}
        <div className="h-16 flex items-center px-4 border-b border-[#e2e8f0] shrink-0">
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className="h-8 w-8 rounded-lg bg-brand-accent flex items-center justify-center shrink-0 border border-[#7e8cf1]/30">
              <svg className="h-4.5 w-4.5 text-white animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            {!collapsed && (
              <span className="font-sans font-black text-[#0f172a] text-base tracking-wider uppercase">
                PULSE
              </span>
            )}
          </div>
        </div>

        {/* Scrollable Navigation Area */}
        <div className="flex-1 overflow-y-auto py-4 px-2 space-y-4 scrollbar-thin">
          {/* Dashboard Home - Main button outside categories */}
          <div className="space-y-1">
            <button
              onClick={() => handleTabClick('dashboard')}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-bold transition-all duration-200 cursor-pointer group border-l-4 relative ${
                isTabActive('dashboard') || isTabActive('reports') && !collapsed
                  ? 'bg-[#7e71f9]/15 text-[#6448dc] border-[#7e71f9] font-extrabold shadow-sm/5' 
                  : 'hover:bg-slate-50 text-[#475569] hover:text-[#0f172a] border-l-4 border-transparent'
              }`}
              title={collapsed ? 'Dashboard (Your analytical home base)' : undefined}
            >
              <LayoutDashboard 
                className={`h-4.5 w-4.5 shrink-0 transition-colors ${
                  isTabActive('dashboard') ? 'text-[#6448dc]' : 'text-[#475569]/70 group-hover:text-[#0f172a]'
                }`}
                strokeWidth={2}
              />
              {!collapsed && (
                <div className="text-left">
                  <span className="tracking-wide block text-[#0f172a]">Dashboard</span>
                  <span className="text-[9.5px] text-[#94a3b8] font-bold block mt-0.5 leading-none">Your analytical home base</span>
                </div>
              )}
              {collapsed && (
                <div className="absolute left-full ml-3 px-2 py-1 bg-slate-900 text-white text-[10px] rounded opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-200 z-50 whitespace-nowrap">
                  Dashboard
                </div>
              )}
            </button>
          </div>

          {/* Sections Map */}
          {sections.map((section) => (
            <div key={section.name} className="space-y-1">
              {/* Category Header */}
              {!collapsed ? (
                <h4 className="text-[9.5px] font-black text-[#6448dc]/70 uppercase tracking-widest px-3 pt-2 pb-1.5 font-sans">
                  {section.name}
                </h4>
              ) : (
                <div className="h-px bg-[#f1f5f9] my-2 mx-2" />
              )}

              {/* Category Items */}
              {section.items.map((item) => {
                const Icon = item.icon;
                const active = isTabActive(item.tab);
                return (
                  <button
                    key={item.name}
                    onClick={() => handleTabClick(item.tab)}
                    className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-xs font-bold transition-all duration-200 cursor-pointer group border-l-4 relative ${
                      active 
                        ? 'bg-[#7e71f9]/15 text-[#6448dc] border-[#7e71f9] font-extrabold shadow-sm/5' 
                        : 'hover:bg-slate-50 text-[#475569] hover:text-[#0f172a] border-l-4 border-transparent'
                    }`}
                    title={collapsed ? item.name : undefined}
                  >
                    <Icon 
                      className={`h-4.5 w-4.5 shrink-0 transition-colors ${
                        active ? 'text-[#6448dc]' : 'text-[#475569]/70 group-hover:text-[#0f172a]'
                      }`}
                      strokeWidth={2}
                    />
                    {!collapsed && <span className="tracking-wide text-[#475569] group-hover:text-[#0f172a]">{item.name}</span>}
                    {collapsed && (
                      <div className="absolute left-full ml-3 px-2 py-1 bg-slate-900 text-white text-[10px] rounded opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-200 z-50 whitespace-nowrap">
                        {item.name}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* User Footer Profile */}
      <div className="p-4 border-t border-[#f1f5f9] shrink-0">
        <button 
          type="button"
          onClick={() => handleTabClick('profile')}
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
                <p className="text-xs font-black text-[#0f172a] truncate leading-tight">Alex Johnson</p>
                <p className="text-[10px] text-[#475569]/85 truncate mt-0.5 font-bold">Sales Manager</p>
              </div>
            )}
          </div>
          {!collapsed && (
            <span className="text-[#475569]/70 hover:text-[#0f172a] transition-colors">
              <ChevronDown className="h-4.5 w-4.5" strokeWidth={2} />
            </span>
          )}
        </button>
      </div>
    </aside>
  );
}
