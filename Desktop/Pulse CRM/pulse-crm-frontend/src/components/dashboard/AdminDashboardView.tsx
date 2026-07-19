'use client';

import React, { useState } from 'react';
import { 
  Users, 
  Database, 
  Activity, 
  Cpu, 
  Settings2, 
  HardDrive,
  CheckCircle,
  AlertTriangle,
  Play,
  TrendingUp,
  Info,
  Clock
} from 'lucide-react';

export default function AdminDashboardView() {
  const [hoveredApiIndex, setHoveredApiIndex] = useState<number | null>(null);
  const [hoveredStorageIndex, setHoveredStorageIndex] = useState<number | null>(null);

  const systemMetrics = {
    totalUsers: 45,
    activeUsers: 38,
    storageUsed: 7.2, // GB
    storageMax: 10,
    crmUsageRate: 92, // %
    apiLatency: 145 // ms
  };

  const integrations = [
    { name: "Gmail Sync API", status: "Healthy", type: "email" },
    { name: "Outlook OAuth", status: "Healthy", type: "email" },
    { name: "Calendar Daemon", status: "Healthy", type: "calendar" },
    { name: "WhatsApp Gateway", status: "Pending", type: "messenger" }
  ];

  const aiStatus = [
    { model: "Lead Scoring (XGBoost)", status: "Active", accuracy: "91.2%", latency: "85ms" },
    { model: "Recommendation Engine", status: "Active", accuracy: "86.5%", latency: "120ms" },
    { model: "Conv. Intelligence (LLM)", status: "Active", accuracy: "94.8%", latency: "245ms" }
  ];

  // Daily API Traffic Data (7 Days)
  const apiTrafficData = [
    { day: "Mon", requests: 12400, x: 50, y: 150 },
    { day: "Tue", requests: 15800, x: 120, y: 120 },
    { day: "Wed", requests: 14200, x: 190, y: 135 },
    { day: "Thu", requests: 21000, x: 260, y: 80 },
    { day: "Fri", requests: 19500, x: 330, y: 95 },
    { day: "Sat", requests: 8400,  x: 400, y: 180 },
    { day: "Sun", requests: 9200,  x: 470, y: 170 }
  ];

  // Storage Distribution Categories
  const storageBreakdown = [
    { name: "CRM Records (DB)", size: "2.88 GB", pct: 40, color: "#7957fb" },
    { name: "Uploaded Attachments", size: "2.52 GB", pct: 35, color: "#7e71f9" },
    { name: "System Audit Logs", size: "1.08 GB", pct: 15, color: "#7e8cf1" },
    { name: "Redis Caching", size: "0.72 GB", pct: 10, color: "#6ec2de" }
  ];

  // AI Inference Latency data
  const latencyModels = [
    { name: "Lead Scoring", latency: 85, color: "#6ec2de", height: 50 },
    { name: "Recommendations", latency: 120, color: "#7e8cf1", height: 80 },
    { name: "Conv. Intel.", latency: 245, color: "#7957fb", height: 160 }
  ];

  // Helper to draw donut segments for storage breakdown
  const getDonutSegments = (data: typeof storageBreakdown, radius = 50) => {
    let currentAngle = -90;
    const cx = 80;
    const cy = 80;
    
    return data.map((item) => {
      const angle = (item.pct / 100) * 360;
      const startAngleRad = (currentAngle * Math.PI) / 180;
      const endAngleRad = ((currentAngle + angle) * Math.PI) / 180;
      
      const x1 = cx + radius * Math.cos(startAngleRad);
      const y1 = cy + radius * Math.sin(startAngleRad);
      const x2 = cx + radius * Math.cos(endAngleRad);
      const y2 = cy + radius * Math.sin(endAngleRad);
      
      const largeArcFlag = angle > 180 ? 1 : 0;
      
      const pathData = `
        M ${x1} ${y1}
        A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}
      `;
      
      currentAngle += angle;
      return { path: pathData, color: item.color };
    });
  };

  const storageSegments = getDonutSegments(storageBreakdown);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-sans text-brand-heading tracking-tight font-bold">
          Admin Dashboard
        </h1>
        <p className="text-xs md:text-sm text-brand-text/75 mt-1 font-medium tracking-wide">
          Monitor system diagnostics, database allocation, background integrations, and AI pipelines.
        </p>
      </div>

      {/* KPI Core Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Users */}
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-1.5 hover:-translate-y-0.5 hover:shadow-md transition-all duration-300">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider block">Total System Users</span>
            <Users className="h-4 w-4 text-brand-accent" />
          </div>
          <h4 className="text-2xl font-bold text-brand-heading">{systemMetrics.totalUsers}</h4>
          <span className="text-[9px] text-slate-400 font-bold block mt-1">
            Across 3 authorization roles
          </span>
        </div>

        {/* Active Users */}
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-1.5 hover:-translate-y-0.5 hover:shadow-md transition-all duration-300">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider block">Active Concurrent Sessions</span>
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
          </div>
          <h4 className="text-2xl font-bold text-brand-heading">{systemMetrics.activeUsers}</h4>
          <span className="text-[9px] text-emerald-600 font-extrabold block mt-1">
            {(systemMetrics.activeUsers / systemMetrics.totalUsers * 100).toFixed(0)}% Activity Ratio
          </span>
        </div>

        {/* Storage */}
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-2 hover:-translate-y-0.5 hover:shadow-md transition-all duration-300">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider">Storage Consumption</span>
            <Database className="h-4 w-4 text-brand-blue" />
          </div>
          <h4 className="text-2xl font-bold text-brand-heading">
            {((systemMetrics.storageUsed / systemMetrics.storageMax) * 100).toFixed(0)}%
          </h4>
          <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full bg-brand-accent rounded-full" style={{ width: `${(systemMetrics.storageUsed / systemMetrics.storageMax) * 100}%` }} />
          </div>
        </div>

        {/* System Health */}
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-1.5 hover:-translate-y-0.5 hover:shadow-md transition-all duration-300">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-wider block">Core System Health</span>
            <Activity className="h-4 w-4 text-emerald-500" />
          </div>
          <div className="flex items-center space-x-1.5">
            <h4 className="text-2xl font-bold text-brand-heading">Online</h4>
          </div>
          <span className="text-[9px] text-slate-400 font-bold block mt-1">
            API latency average: {systemMetrics.apiLatency}ms
          </span>
        </div>
      </div>

      {/* CHARTS GRID (2 Columns: API load area chart & storage allocation donut) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Daily API Request Load Area Chart (2 Cols) */}
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 lg:col-span-2 flex flex-col justify-between hover:border-brand-border-purple/45 hover:shadow-md transition-all duration-300">
          <div>
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center space-x-2">
                <h3 className="font-bold text-brand-heading text-sm">CRM Request Traffic</h3>
                <span title="Operational query traffic logged across all API endpoints in the last 7 days.">
                  <Info className="h-3.5 w-3.5 text-slate-400 cursor-help" />
                </span>
              </div>
              <span className="text-[10px] font-extrabold bg-brand-sidebar-hover/60 text-brand-text px-2 py-1 rounded">
                Daily API Queries
              </span>
            </div>

            {/* SVG Area Chart */}
            <div className="relative h-56 w-full mt-4">
              <svg className="w-full h-full" viewBox="0 0 520 200" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="admin-api-gradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7957fb" stopOpacity="0.15" />
                    <stop offset="100%" stopColor="#7957fb" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Horizontal Gridlines */}
                <line x1="40" y1="30" x2="500" y2="30" stroke="#7e8cf1" strokeOpacity="0.1" strokeWidth="1" />
                <line x1="40" y1="80" x2="500" y2="80" stroke="#7e8cf1" strokeOpacity="0.1" strokeWidth="1" />
                <line x1="40" y1="130" x2="500" y2="130" stroke="#7e8cf1" strokeOpacity="0.1" strokeWidth="1" />
                <line x1="40" y1="180" x2="500" y2="180" stroke="#7e8cf1" strokeOpacity="0.15" strokeWidth="1.5" />

                {/* Axis Labels */}
                <text x="10" y="34" className="text-[9px] font-bold fill-slate-400 tabular-nums">25K</text>
                <text x="10" y="84" className="text-[9px] font-bold fill-slate-400 tabular-nums">15K</text>
                <text x="10" y="134" className="text-[9px] font-bold fill-slate-400 tabular-nums">5K</text>

                {/* Shaded Area */}
                <path 
                  d={`M 50 180 L 50 150 Q 85 135 120 120 T 190 135 T 260 80 T 330 95 T 400 180 T 470 170 L 470 180 Z`} 
                  fill="url(#admin-api-gradient)"
                />

                {/* Chart Line Path */}
                <path 
                  d="M 50 150 Q 85 135 120 120 T 190 135 T 260 80 T 330 95 T 400 180 T 470 170" 
                  fill="none" 
                  stroke="#7957fb" 
                  strokeWidth="2.5" 
                  strokeLinecap="round"
                />

                {/* Interactive Points */}
                {apiTrafficData.map((pt, idx) => {
                  const isHovered = hoveredApiIndex === idx;
                  return (
                    <g key={idx}>
                      <circle 
                        cx={pt.x} 
                        cy={pt.y} 
                        r={isHovered ? 6 : 3.5} 
                        fill={isHovered ? "#fff" : "#7957fb"} 
                        stroke="#7957fb" 
                        strokeWidth={isHovered ? 3.5 : 0} 
                        className="cursor-pointer transition-all duration-150"
                        onMouseEnter={() => setHoveredApiIndex(idx)}
                        onMouseLeave={() => setHoveredApiIndex(null)}
                      />
                      <text x={pt.x - 8} y="195" className="text-[9px] font-bold fill-slate-450">{pt.day}</text>
                    </g>
                  );
                })}
              </svg>

              {/* Hover Tooltip */}
              {hoveredApiIndex !== null && (
                <div 
                  className="absolute bg-slate-900 text-white rounded-lg p-2 text-[10px] font-bold shadow-xl border border-slate-700 pointer-events-none transition-all duration-150 animate-in fade-in"
                  style={{ 
                    left: `${(apiTrafficData[hoveredApiIndex].x / 520) * 100}%`, 
                    top: `${(apiTrafficData[hoveredApiIndex].y / 200) * 100 - 25}%`,
                    transform: 'translateX(-50%)'
                  }}
                >
                  <p className="whitespace-nowrap">{apiTrafficData[hoveredApiIndex].requests.toLocaleString()} queries</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Database Storage Allocation breakdown Donut Chart (1 Col) */}
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 flex flex-col justify-between hover:border-brand-border-purple/45 hover:shadow-md transition-all duration-300">
          <div>
            <h3 className="font-bold text-brand-heading text-sm mb-4">Storage Allocation</h3>
            
            {/* Donut Canvas */}
            <div className="relative h-36 w-36 mx-auto mt-4">
              <svg className="w-full h-full" viewBox="0 0 160 160">
                {storageSegments.map((seg, idx) => {
                  const isHovered = hoveredStorageIndex === idx;
                  return (
                    <path
                      key={idx}
                      d={seg.path}
                      fill="none"
                      stroke={seg.color}
                      strokeWidth={isHovered ? "18" : "14"}
                      className="cursor-pointer transition-all duration-150"
                      onMouseEnter={() => setHoveredStorageIndex(idx)}
                      onMouseLeave={() => setHoveredStorageIndex(null)}
                    />
                  );
                })}
              </svg>
              {/* Central text displaying total */}
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none pointer-events-none">
                <span className="text-base font-extrabold text-brand-heading tabular-nums">7.2 GB</span>
                <span className="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Allocated</span>
              </div>
            </div>

            {/* Legends list */}
            <div className="mt-4 space-y-1.5">
              {storageBreakdown.map((item, idx) => (
                <div 
                  key={idx}
                  className={`flex justify-between items-center p-1 rounded transition-colors ${
                    hoveredStorageIndex === idx ? 'bg-slate-50' : ''
                  }`}
                  onMouseEnter={() => setHoveredStorageIndex(idx)}
                  onMouseLeave={() => setHoveredStorageIndex(null)}
                >
                  <div className="flex items-center space-x-2">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-[10px] font-bold text-brand-text/80">{item.name}</span>
                  </div>
                  <span className="text-[10px] font-extrabold text-brand-heading tabular-nums">{item.size} ({item.pct}%)</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Lower Row: AI Models Latencies (Chart) & Micro-Services List */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* AI Inference Latency Chart (Col 6) */}
        <div className="col-span-12 lg:col-span-6 bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-4 hover:border-brand-border-purple/45 hover:shadow-md transition-all duration-300">
          <div className="flex justify-between items-center">
            <h3 className="font-extrabold text-brand-heading text-sm flex items-center">
              <Clock className="h-4.5 w-4.5 mr-2 text-brand-accent" />
              <span>AI Models Latency (Inference Speed)</span>
            </h3>
            <span className="text-[9px] font-extrabold bg-blue-50 text-blue-700 px-2 py-0.5 rounded">
              Lower is Better
            </span>
          </div>

          {/* Vertical Bar Chart (SVG) */}
          <div className="relative h-48 w-full mt-4">
            <svg className="w-full h-full" viewBox="0 0 350 180">
              {/* Horizontal Gridlines */}
              <line x1="10" y1="40" x2="330" y2="40" stroke="#7e8cf1" strokeOpacity="0.1" strokeWidth="1" />
              <line x1="10" y1="90" x2="330" y2="90" stroke="#7e8cf1" strokeOpacity="0.1" strokeWidth="1" />
              <line x1="10" y1="140" x2="330" y2="140" stroke="#7e8cf1" strokeOpacity="0.1" strokeWidth="1" />
              <line x1="10" y1="170" x2="330" y2="170" stroke="#7e8cf1" strokeOpacity="0.2" strokeWidth="1.5" />

              {/* Grid Latency Markers */}
              <text x="332" y="44" className="text-[8px] font-bold fill-slate-400 tabular-nums">250ms</text>
              <text x="332" y="94" className="text-[8px] font-bold fill-slate-400 tabular-nums">120ms</text>
              <text x="332" y="144" className="text-[8px] font-bold fill-slate-400 tabular-nums">50ms</text>

              {/* Draw Model Columns */}
              {latencyModels.map((m, idx) => {
                const colWidth = 40;
                const colSpacing = 90;
                const xPos = 40 + idx * colSpacing;
                const yPos = 170 - m.height;
                return (
                  <g key={idx}>
                    {/* Bar columns */}
                    <rect 
                      x={xPos} 
                      y={yPos} 
                      width={colWidth} 
                      height={m.height} 
                      rx="4" 
                      fill={m.color} 
                      className="hover:opacity-90 transition-opacity cursor-pointer"
                    />
                    {/* Latency Number Value Label */}
                    <text 
                      x={xPos + colWidth / 2} 
                      y={yPos - 6} 
                      textAnchor="middle" 
                      className="text-[9px] font-black fill-brand-heading tabular-nums"
                    >
                      {m.latency}ms
                    </text>
                    {/* Model Name Axis Label */}
                    <text 
                      x={xPos + colWidth / 2} 
                      y="180" 
                      textAnchor="middle" 
                      className="text-[8.5px] font-extrabold fill-slate-450"
                    >
                      {m.name}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Integrations Monitor (Col 6) */}
        <div className="col-span-12 lg:col-span-6 bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-4 hover:border-brand-border-purple/45 hover:shadow-md transition-all duration-300">
          <h3 className="font-extrabold text-brand-heading text-sm flex items-center">
            <Activity className="h-4.5 w-4.5 mr-2 text-brand-accent" />
            <span>Integration Micro-Services Status</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {integrations.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-3.5 border border-brand-border-purple/15 rounded-lg bg-slate-50/50 hover:bg-slate-50 transition-colors">
                <div>
                  <p className="text-xs font-extrabold text-brand-text">{item.name}</p>
                  <p className="text-[9px] text-slate-400 font-bold uppercase mt-0.5">{item.type}</p>
                </div>
                <span className={`px-2 py-0.5 rounded font-extrabold uppercase tracking-wide text-[8px] ${
                  item.status === 'Healthy' 
                    ? 'bg-emerald-50 text-emerald-700' 
                    : 'bg-amber-50 text-amber-700'
                }`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
