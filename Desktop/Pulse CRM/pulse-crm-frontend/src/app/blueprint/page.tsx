'use client';

import React from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Cpu, 
  Database, 
  Calendar, 
  Code, 
  CheckCircle2, 
  Server
} from 'lucide-react';

export default function BlueprintPage() {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-brand-text antialiased selection:bg-brand-accent/20">
      {/* Top Banner Header */}
      <header className="bg-white border-b border-slate-100 shadow-sm/5 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link 
              href="/"
              className="inline-flex items-center space-x-2 text-brand-text/70 hover:text-brand-accent font-bold text-xs transition-colors py-1.5 px-3 rounded-lg hover:bg-slate-50 border border-slate-100 shadow-sm/5"
            >
              <ArrowLeft className="h-4 w-4" strokeWidth={2.25} />
              <span>Back to Dashboard</span>
            </Link>
            <div className="h-6 w-px bg-slate-200" />
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 rounded-lg bg-brand-accent flex items-center justify-center border border-brand-border-purple/35 shadow-sm/5">
                <svg className="h-4.5 w-4.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.25}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="font-extrabold text-brand-heading text-sm tracking-wider uppercase">
                Pulse CRM Engineering
              </span>
            </div>
          </div>
          <span className="text-[10px] font-extrabold text-white bg-brand-accent hover:bg-brand-accent-hover px-2.5 py-1 rounded-full uppercase tracking-wider">
            Week 1 Blueprint
          </span>
        </div>
      </header>

      {/* Hero section */}
      <div className="bg-gradient-to-r from-brand-heading/10 via-brand-accent/5 to-transparent border-b border-brand-border-purple/10 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl md:text-5xl font-serif text-brand-heading font-normal tracking-tight">
            Week 1 Planning & Handoff Blueprint
          </h1>
          <p className="text-sm md:text-base text-brand-text/75 mt-3 max-w-3xl leading-relaxed font-medium">
            This document outlines the technical architecture, Entity-Relationship database schemas, 
            REST API route planner, and task checklist implemented for the **Pulse CRM** platform.
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Component cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Client App Card */}
          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.02)] flex flex-col justify-between">
            <div>
              <div className="h-10 w-10 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center mb-4">
                <Code className="h-5 w-5 text-indigo-600" strokeWidth={2} />
              </div>
              <h3 className="text-base font-extrabold text-brand-text">Frontend App Router</h3>
              <p className="text-xs text-brand-text/70 mt-2 leading-relaxed font-medium">
                Next.js client interface styled with Tailwind CSS, supporting responsive views, 
                rich widgets, pipeline dashboards, and client-side simulated data loaders.
              </p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-50 flex items-center justify-between text-[11px] font-bold text-indigo-600">
              <span>Next.js 16.2 & React 19</span>
              <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded">Active</span>
            </div>
          </div>

          {/* Backend App Card */}
          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.02)] flex flex-col justify-between">
            <div>
              <div className="h-10 w-10 rounded-lg bg-violet-50 border border-violet-100 flex items-center justify-center mb-4">
                <Server className="h-5 w-5 text-violet-600" strokeWidth={2} />
              </div>
              <h3 className="text-base font-extrabold text-brand-text">REST API Server</h3>
              <p className="text-xs text-brand-text/70 mt-2 leading-relaxed font-medium">
                FastAPI web service written in Python 3.12, providing modular routing controllers, 
                JWT authentication, auto-generated OpenAPI Swagger docs, and database models.
              </p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-50 flex items-center justify-between text-[11px] font-bold text-violet-600">
              <span>FastAPI & Python 3.12</span>
              <span className="bg-violet-50 text-violet-700 px-2 py-0.5 rounded">Active</span>
            </div>
          </div>

          {/* Database Card */}
          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.02)] flex flex-col justify-between">
            <div>
              <div className="h-10 w-10 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center mb-4">
                <Database className="h-5 w-5 text-emerald-600" strokeWidth={2} />
              </div>
              <h3 className="text-base font-extrabold text-brand-text">Relational Database</h3>
              <p className="text-xs text-brand-text/70 mt-2 leading-relaxed font-medium">
                PostgreSQL schema utilizing UUID primary keys, relational mapping tables, indexes, 
                and structured cascade deletions for robust pipeline stage management.
              </p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-50 flex items-center justify-between text-[11px] font-bold text-emerald-600">
              <span>PostgreSQL 15</span>
              <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded">Awaiting Local Setup</span>
            </div>
          </div>
        </div>

        {/* Section: Entity Relationship Database Design */}
        <section className="bg-white rounded-xl border border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.02)] p-6 md:p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="h-9 w-9 rounded-lg bg-brand-accent/10 flex items-center justify-center border border-brand-accent/20">
              <Database className="h-4.5 w-4.5 text-brand-accent" strokeWidth={2} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-brand-heading">Entity-Relationship Database Schema</h2>
              <p className="text-xs text-brand-text/60 mt-0.5 font-bold">10 core tables defined in the PostgreSQL database configuration</p>
            </div>
          </div>

          <div className="overflow-x-auto border border-brand-border-purple/15 rounded-xl">
            <table className="w-full text-left border-collapse text-xs font-semibold">
              <thead>
                <tr className="bg-brand-secondary-accent/5 border-b border-brand-border-purple/20 text-brand-heading uppercase tracking-wider text-[10px]">
                  <th className="p-3.5 pl-5">Table Name</th>
                  <th className="p-3.5">Primary Key</th>
                  <th className="p-3.5">Key Relationships</th>
                  <th className="p-3.5">Key Fields & Types</th>
                  <th className="p-3.5 pr-5">Purpose</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-brand-text/90">
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">users</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5">Many-to-Many via <span className="font-mono text-[10px]">users_roles</span></td>
                  <td className="p-3.5 font-medium">email (unique), password_hash, full_name, timestamps</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Authenticable system users/sales representatives.</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">roles</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5">Many-to-Many via <span className="font-mono text-[10px]">roles_permissions</span></td>
                  <td className="p-3.5 font-medium">name (unique, e.g., 'Administrator'), description</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Role-Based Access Control configurations.</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">companies</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5"><span className="font-mono text-[10px]">owner_id</span> references <span className="font-mono text-[10px]">users</span></td>
                  <td className="p-3.5 font-medium">name, domain, industry, owner_id, created_at</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Organization profiles linked to contacts/deals.</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">contacts</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5"><span className="font-mono text-[10px]">company_id</span> references <span className="font-mono text-[10px]">companies</span></td>
                  <td className="p-3.5 font-medium">first_name, last_name, email, phone, job_title</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Personnel contact roster at profile organizations.</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">leads</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5"><span className="font-mono text-[10px]">contact_id</span> references <span className="font-mono text-[10px]">contacts</span></td>
                  <td className="p-3.5 font-medium">title, description, value (numeric), status, source</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Intake pipeline entries before conversions.</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">pipeline_stages</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5">Referenced by <span className="font-mono text-[10px]">deals</span></td>
                  <td className="p-3.5 font-medium">name (unique), probability (0-100), display_order</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Board columns (Qualified, Proposal, Under Review, Won).</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">deals</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5">
                    <span className="font-mono text-[10px]">company_id</span>, <span className="font-mono text-[10px]">contact_id</span>, <span className="font-mono text-[10px]">stage_id</span>, <span className="font-mono text-[10px]">owner_id</span>
                  </td>
                  <td className="p-3.5 font-medium">name, value, status, closed_at, created_at</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Active sales pipeline opportunities.</td>
                </tr>
                <tr className="hover:bg-slate-50/50">
                  <td className="p-3.5 pl-5 font-bold text-brand-heading">emails</td>
                  <td className="p-3.5"><span className="font-mono bg-slate-100 text-[10px] px-1.5 py-0.5 rounded">id (UUID)</span></td>
                  <td className="p-3.5">
                    <span className="font-mono text-[10px]">contact_id</span>, <span className="font-mono text-[10px]">deal_id</span>
                  </td>
                  <td className="p-3.5 font-medium">message_id (unique), subject, body, sender, sent_at</td>
                  <td className="p-3.5 pr-5 font-medium text-brand-text/75">Synchronized Gmail activity histories.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section: OpenAPI Route Planner */}
        <section className="bg-white rounded-xl border border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.02)] p-6 md:p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="h-9 w-9 rounded-lg bg-indigo-50 flex items-center justify-center border border-indigo-100">
              <Cpu className="h-4.5 w-4.5 text-indigo-600" strokeWidth={2} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-brand-heading">API Endpoint & Routing Map</h2>
              <p className="text-xs text-brand-text/60 mt-0.5 font-bold">Comprehensive FastAPI backend router mappings</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Core Routers */}
            <div className="border border-slate-100 rounded-xl p-4 bg-slate-50/30">
              <h3 className="text-xs font-extrabold text-brand-heading uppercase tracking-wider mb-3">Authentication & Operators</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 bg-white border border-slate-100 rounded-lg text-[11px] font-bold">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 bg-emerald-50 text-emerald-700 rounded uppercase text-[9px]">POST</span>
                    <span className="font-mono font-medium">/auth/register</span>
                  </div>
                  <span className="text-brand-text/60">Register new user</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-white border border-slate-100 rounded-lg text-[11px] font-bold">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 bg-emerald-50 text-emerald-700 rounded uppercase text-[9px]">POST</span>
                    <span className="font-mono font-medium">/auth/login</span>
                  </div>
                  <span className="text-brand-text/60">Generate JWT Access Token</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-white border border-slate-100 rounded-lg text-[11px] font-bold">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded uppercase text-[9px]">GET</span>
                    <span className="font-mono font-medium">/users/me</span>
                  </div>
                  <span className="text-brand-text/60">Retrieve active operator profile</span>
                </div>
              </div>
            </div>

            {/* Pipeline & Sales Routers */}
            <div className="border border-slate-100 rounded-xl p-4 bg-slate-50/30">
              <h3 className="text-xs font-extrabold text-brand-heading uppercase tracking-wider mb-3">CRM Entities & Board Pipelines</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 bg-white border border-slate-100 rounded-lg text-[11px] font-bold">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded uppercase text-[9px]">GET</span>
                    <span className="font-mono font-medium">/deals</span>
                  </div>
                  <span className="text-brand-text/60">List all pipeline deals</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-white border border-slate-100 rounded-lg text-[11px] font-bold">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 bg-amber-50 text-amber-700 rounded uppercase text-[9px]">PUT</span>
                    <span className="font-mono font-medium">/deals/{"{id}"}/stage</span>
                  </div>
                  <span className="text-brand-text/60">Update pipeline stages (drag-drop)</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-white border border-slate-100 rounded-lg text-[11px] font-bold">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded uppercase text-[9px]">GET</span>
                    <span className="font-mono font-medium">/gmail/oauth/link</span>
                  </div>
                  <span className="text-brand-text/60">Retrieve Google OAuth login URL</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section: Sprint Tasks Milestone */}
        <section className="bg-white rounded-xl border border-slate-100 shadow-[0_4px_20px_rgba(0,0,0,0.02)] p-6 md:p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="h-9 w-9 rounded-lg bg-emerald-50 flex items-center justify-center border border-emerald-100">
              <Calendar className="h-4.5 w-4.5 text-emerald-600" strokeWidth={2} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-brand-heading">Development & Setup Guide</h2>
              <p className="text-xs text-brand-text/60 mt-0.5 font-bold">Recommended next steps for manual application setup</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-start space-x-3 p-3 rounded-xl border border-slate-100 hover:bg-slate-50/30 transition-colors">
              <CheckCircle2 className="h-4.5 w-4.5 text-brand-accent mt-0.5 shrink-0" strokeWidth={2.25} />
              <div>
                <p className="text-xs font-extrabold text-brand-text">1. Start the Frontend Client</p>
                <p className="text-[11px] text-brand-text/75 mt-1 leading-relaxed font-medium">
                  The client dev server is currently running at <a href="http://localhost:3000" className="text-brand-accent hover:underline font-bold">http://localhost:3000</a>. Next.js supports full-fidelity UI views with local mock data out-of-the-box.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-3 p-3 rounded-xl border border-slate-100 hover:bg-slate-50/30 transition-colors">
              <div className="h-4.5 w-4.5 rounded-full border-2 border-slate-350 shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-extrabold text-brand-text">2. Configure the Database Instance</p>
                <p className="text-[11px] text-brand-text/75 mt-1 leading-relaxed font-medium">
                  You need a running PostgreSQL server on port <code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px]">5432</code> with a database named <code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px]">pulse-crm</code>. Once up, execute the SQL schema defined in <code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px]">pulse-crm-backend/db/init.sql</code> to create all schemas and default seeds.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-3 p-3 rounded-xl border border-slate-100 hover:bg-slate-50/30 transition-colors">
              <div className="h-4.5 w-4.5 rounded-full border-2 border-slate-350 shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-extrabold text-brand-text">3. Boot the FastAPI Backend Web Server</p>
                <p className="text-[11px] text-brand-text/75 mt-1 leading-relaxed font-medium font-sans">
                  Navigate to <code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px]">pulse-crm-backend</code>, activate the virtual environment (<code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px]">.\venv\Scripts\activate</code>), install the required dependencies (<code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px]">pip install -r requirements.txt</code>), and start the server with:
                </p>
                <pre className="bg-slate-900 text-slate-100 text-[10px] font-mono p-3 rounded-lg mt-2 overflow-x-auto shadow-inner">
uvicorn main:app --reload --port 8000</pre>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 border-t border-slate-100 text-center text-[10px] text-brand-text/50 font-bold">
        © {new Date().getFullYear()} Pulse CRM Engineering Board. Built using React 19 & FastAPI.
      </footer>
    </div>
  );
}
