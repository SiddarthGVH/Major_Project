'use client';

import React, { useState, useEffect } from 'react';
import Sidebar from '@/components/dashboard/Sidebar';
import Header from '@/components/dashboard/Header';
import StatCards from '@/components/dashboard/StatCards';
import Charts from '@/components/dashboard/Charts';
import Widgets from '@/components/dashboard/Widgets';
import RightPanel from '@/components/dashboard/RightPanel';
import ReportBuilderModal from '@/components/dashboard/ReportBuilderModal';
import LeadsView from '@/components/dashboard/LeadsView';
import CompaniesView from '@/components/dashboard/CompaniesView';
import ContactsView from '@/components/dashboard/ContactsView';
import PipelineView from '@/components/dashboard/PipelineView';
import ActivitiesView from '@/components/dashboard/ActivitiesView';
import EmailsView from '@/components/dashboard/EmailsView';
import AIInsightsView from '@/components/dashboard/AIInsightsView';
import CalendarView from '@/components/dashboard/CalendarView';
import TasksView from '@/components/dashboard/TasksView';
import NotificationsView from '@/components/dashboard/NotificationsView';
import ProfileView from '@/components/dashboard/ProfileView';
import SettingsView from '@/components/dashboard/SettingsView';
import { Calendar, Filter, ChevronDown, Check } from 'lucide-react';

export default function DashboardHome() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('reports');
  const [dashboardSubTab, setDashboardSubTab] = useState('overview');
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [showFiltersMenu, setShowFiltersMenu] = useState(false);
  const [selectedPipelineType, setSelectedPipelineType] = useState('All');
  
  // Simulated loading and empty states
  const [isLoading, setIsLoading] = useState(false);
  const [isEmpty, setIsEmpty] = useState(false);

  // Trigger loading skeleton on sub-tab change to demo loaders
  const handleSubTabChange = (tabKey: string) => {
    setDashboardSubTab(tabKey);
    setIsLoading(true);
    
    // Simulate empty state on Marketing tab for demo
    if (tabKey === 'marketing') {
      setIsEmpty(true);
    } else {
      setIsEmpty(false);
    }

    setTimeout(() => {
      setIsLoading(false);
    }, 450);
  };

  // Custom reports state
  const [recentReports, setRecentReports] = useState([
    { id: 1, title: "Sales Performance Overview", time: "Generated 2 hours ago" },
    { id: 2, title: "Pipeline Health Report", time: "Generated 1 day ago" },
    { id: 3, title: "Revenue Forecast Report", time: "Generated 2 days ago" },
    { id: 4, title: "Activity Summary", time: "Generated 3 days ago" }
  ]);

  const handleSaveReport = (newReport: { title: string; time: string }) => {
    setRecentReports([
      { id: Date.now(), ...newReport },
      ...recentReports
    ]);
  };

  const subTabs = [
    { name: 'Overview', key: 'overview' },
    { name: 'Sales', key: 'sales' },
    { name: 'Pipeline', key: 'pipeline' },
    { name: 'Activity', key: 'activity' },
    { name: 'Marketing', key: 'marketing' }, // will show empty state
    { name: 'Team', key: 'team' },
    { name: 'Forecasting', key: 'forecasting' },
    { name: 'Custom Reports', key: 'custom' },
  ];

  return (
    <div className="flex bg-white h-screen overflow-hidden font-sans text-brand-text antialiased">
      {/* Sidebar navigation - toned down background */}
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        collapsed={sidebarCollapsed} 
        setCollapsed={setSidebarCollapsed} 
      />

      {/* Main dashboard content container */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        
        {/* Top Navbar */}
        <Header 
          collapsed={sidebarCollapsed} 
          setCollapsed={setSidebarCollapsed} 
          onNewReportClick={() => setIsReportModalOpen(true)} 
          onTabChange={(tab) => setActiveTab(tab)}
        />

        {/* Dashboard inner scroll view with fixed size */}
        <main className="flex-1 overflow-hidden p-6 md:p-8 flex flex-col min-h-0">
          {activeTab === 'leads' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><LeadsView /></div>
          ) : activeTab === 'contacts' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><ContactsView /></div>
          ) : activeTab === 'companies' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><CompaniesView /></div>
          ) : (activeTab === 'deals' || activeTab === 'pipeline') ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><PipelineView /></div>
          ) : activeTab === 'activities' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><ActivitiesView /></div>
          ) : activeTab === 'calendar' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><CalendarView /></div>
          ) : activeTab === 'tasks' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><TasksView /></div>
          ) : activeTab === 'emails' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><EmailsView /></div>
          ) : activeTab === 'ai insights' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><AIInsightsView /></div>
          ) : activeTab === 'settings' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><SettingsView /></div>
          ) : activeTab === 'profile' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><ProfileView /></div>
          ) : activeTab === 'notifications' ? (
            <div className="flex-1 min-h-0 overflow-y-auto"><NotificationsView /></div>
          ) : (
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              {/* Header block with improved contrast & page title visual prominence */}
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 shrink-0">
                <div>
                  <h1 className="text-3xl md:text-4xl font-serif text-brand-heading tracking-tight font-normal">
                    Reports & analytics
                  </h1>
                  <p className="text-xs md:text-sm text-brand-text/75 mt-2 leading-relaxed max-w-2xl font-medium tracking-wide">
                    Track performance, analyze trends, and make data-driven decisions.
                  </p>
                </div>
              </div>

              {/* Sub Navigation Tabs (Tactile pills) */}
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 shrink-0 mt-4">
                <nav className="flex space-x-1 p-1 bg-brand-sidebar-hover/15 border border-brand-border-purple/20 rounded-xl overflow-x-auto scrollbar-none shrink-0">
                  {subTabs.map((tab) => {
                    const isActive = dashboardSubTab === tab.key;
                    return (
                      <button
                        key={tab.key}
                        onClick={() => handleSubTabChange(tab.key)}
                        className={`py-1.5 px-3.5 rounded-lg font-extrabold text-xs transition-all duration-200 whitespace-nowrap cursor-pointer ${
                          isActive 
                            ? 'bg-brand-accent text-white shadow-sm' 
                            : 'text-brand-text/75 hover:text-brand-heading hover:bg-brand-sidebar-hover/20'
                        }`}
                      >
                        {tab.name}
                      </button>
                    );
                  })}
                </nav>

                {/* Datepicker and Filters (Tactile and premium style) */}
                <div className="flex items-center space-x-2 shrink-0 self-end md:self-center">
                  <button className="inline-flex items-center space-x-1.5 bg-white border border-brand-border-purple/35 hover:border-brand-border-purple active:bg-slate-50 px-3.5 py-1.5 rounded-lg text-xs font-bold text-brand-text/80 transition-all duration-200 cursor-pointer shadow-sm/5">
                    <Calendar className="h-3.5 w-3.5 text-slate-400" strokeWidth={1.75} />
                    <span className="tabular-nums">May 12 – May 18, 2025</span>
                  </button>

                  <div className="relative">
                    <button 
                      onClick={() => setShowFiltersMenu(!showFiltersMenu)}
                      className="inline-flex items-center space-x-1.5 bg-white border border-brand-border-purple/35 hover:border-brand-border-purple active:bg-slate-50 px-3.5 py-1.5 rounded-lg text-xs font-bold text-brand-text/80 transition-all duration-200 cursor-pointer shadow-sm/5"
                    >
                      <Filter className="h-3.5 w-3.5 text-slate-400" strokeWidth={1.75} />
                      <span>Filters</span>
                      <ChevronDown className="h-3 w-3 text-slate-400" strokeWidth={1.75} />
                    </button>
                    
                    {showFiltersMenu && (
                      <div className="absolute right-0 mt-2 w-56 bg-white border border-brand-border-purple/35 rounded-xl shadow-xl overflow-hidden z-20 animate-in fade-in slide-in-from-top-2 duration-200 p-2.5 text-left">
                        <p className="text-[9px] font-bold text-brand-heading uppercase tracking-wider mb-2 px-2">Filter Pipeline</p>
                        <div className="space-y-0.5">
                          {['All', 'Enterprise Deals', 'Mid-Market Deals', 'Small Business Deals'].map((type) => (
                            <button
                              key={type}
                              onClick={() => {
                                setSelectedPipelineType(type);
                                setShowFiltersMenu(false);
                              }}
                              className="w-full flex items-center justify-between text-xs font-semibold text-brand-text/80 hover:bg-slate-50 px-2 py-1.5 rounded-lg text-left"
                            >
                              <span>{type}</span>
                              {selectedPipelineType === type && <Check className="h-3.5 w-3.5 text-brand-accent" strokeWidth={2} />}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* KPI Stat Cards (Spans full horizontal width above grid split) */}
              <div className="shrink-0 mt-6">
                <StatCards timeFilter={dashboardSubTab} loading={isLoading} />
              </div>

              {/* Dashboard Side-Scrolling Columns Layout */}
              <div className="flex-1 min-h-0 overflow-x-auto flex flex-row space-x-6 mt-6 pb-2 scrollbar-thin scrollbar-thumb-brand-border-purple/20 scrollbar-track-transparent">
                
                {/* Column 1: Charts (Revenue, stage funnel, source donuts) */}
                <div className="w-[850px] shrink-0 h-full overflow-y-auto pr-2 space-y-6 scrollbar-thin scrollbar-thumb-slate-200">
                  <Charts loading={isLoading} empty={isEmpty} />
                </div>

                {/* Column 2: Widgets (Leaderboard & Activity Logs) */}
                <div className="w-[800px] shrink-0 h-full overflow-y-auto pr-2 space-y-6 scrollbar-thin scrollbar-thumb-slate-200">
                  <Widgets loading={isLoading} />
                </div>

                {/* Column 3: Report Builder, Key Metrics, Recent Reports */}
                <div className="w-[320px] shrink-0 h-full overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-200">
                  <RightPanel 
                    onNewReportClick={() => setIsReportModalOpen(true)} 
                    recentReports={recentReports}
                    loading={isLoading}
                  />
                </div>

              </div>
            </div>
          )}
        </main>
      </div>

      {/* Report builder modal dialog */}
      <ReportBuilderModal 
        isOpen={isReportModalOpen} 
        onClose={() => setIsReportModalOpen(false)} 
        onSave={handleSaveReport}
      />
    </div>
  );
}
