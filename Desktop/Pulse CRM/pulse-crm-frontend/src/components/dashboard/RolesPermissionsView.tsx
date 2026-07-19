'use client';

import React, { useState } from 'react';
import { 
  Shield, 
  Lock, 
  Check, 
  Info,
  Save
} from 'lucide-react';

interface PermissionRow {
  key: string;
  category: string;
  name: string;
  description: string;
  admin: boolean;
  manager: boolean;
  representative: boolean;
}

export default function RolesPermissionsView() {
  const [matrix, setMatrix] = useState<PermissionRow[]>([
    { key: "view_leads", category: "CRM Data", name: "View Leads/Contacts", description: "Read base client records and timelines", admin: true, manager: true, representative: true },
    { key: "edit_leads", category: "CRM Data", name: "Write/Modify Leads", description: "Create or modify leads details and deals values", admin: true, manager: true, representative: true },
    { key: "delete_leads", category: "CRM Data", name: "Delete Lead Records", description: "Permanently delete lead records", admin: true, manager: false, representative: false },
    
    { key: "view_reports", category: "Analytics", name: "Access Standard Reports", description: "View pipeline and revenue reports", admin: true, manager: true, representative: true },
    { key: "view_forecasts", category: "Analytics", name: "Access Team Forecasts", description: "Read forecasted projections and confidence indexes", admin: true, manager: true, representative: false },
    
    { key: "manage_users", category: "Administration", name: "Manage System Users", description: "Provision accounts, toggle status, reset credentials", admin: true, manager: false, representative: false },
    { key: "edit_roles", category: "Administration", name: "Configure Roles Permissions", description: "Modify matrix mapping access bounds", admin: true, manager: false, representative: false },
    { key: "system_settings", category: "Administration", name: "Modify System Settings", description: "Email configurations, API connection links, security keys", admin: true, manager: false, representative: false }
  ]);

  const [toast, setToast] = useState<string | null>(null);

  const togglePermission = (idx: number, role: 'admin' | 'manager' | 'representative') => {
    // Admin permissions are locked to true for safety in this demo
    if (role === 'admin') return;
    
    const updated = [...matrix];
    updated[idx][role] = !updated[idx][role];
    setMatrix(updated);
  };

  const handleSave = () => {
    setToast("Authorization matrix configurations written successfully!");
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Toast Alert */}
      {toast && (
        <div className="fixed bottom-5 right-5 z-55 bg-slate-900 dark:bg-brand-accent text-white px-4 py-2.5 rounded-xl shadow-xl flex items-center space-x-2 text-xs font-bold animate-in fade-in slide-in-from-bottom-2 duration-300">
          <Check className="h-4 w-4" />
          <span>{toast}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-sans text-brand-heading tracking-tight font-bold">
            Roles & Permissions
          </h1>
          <p className="text-xs md:text-sm text-brand-text/75 mt-1 font-medium tracking-wide">
            Configure system authorization profiles and manage the permission access matrix.
          </p>
        </div>

        <button 
          onClick={handleSave}
          className="inline-flex items-center space-x-1.5 px-3.5 py-2 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold transition-colors cursor-pointer shadow-sm self-start sm:self-center"
        >
          <Save className="h-4 w-4" />
          <span>Save Changes</span>
        </button>
      </div>

      {/* Roles Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { name: "Admin Role", usersCount: 2, desc: "Full root access permissions. Manage system configurations, users profiles, integrations, and core DB settings." },
          { name: "Sales Manager", usersCount: 3, desc: "Manage team performance, review forecasted revenue metrics, sign off on deals stages, and generate audits." },
          { name: "Sales Representative", usersCount: 40, desc: "Standard workspace profile. Ingest leads, log activity calls, edit deals pipeline stages, and sync email accounts." }
        ].map((item, idx) => (
          <div key={idx} className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-xs font-extrabold text-brand-heading">{item.name}</span>
              <span className="text-[9px] font-extrabold bg-slate-100 text-slate-800 px-2 py-0.5 rounded">
                {item.usersCount} Users
              </span>
            </div>
            <p className="text-[11px] text-brand-text/75 leading-relaxed font-semibold">{item.desc}</p>
          </div>
        ))}
      </div>

      {/* Permission Matrix */}
      <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 space-y-4">
        <h3 className="font-extrabold text-brand-heading text-sm flex items-center">
          <Shield className="h-4.5 w-4.5 mr-2 text-brand-accent" />
          <span>Permission matrix Grid</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-100 text-[10px] uppercase font-extrabold text-slate-400">
                <th className="py-2.5">Category</th>
                <th className="py-2.5">Permission Name</th>
                <th className="py-2.5">Description</th>
                <th className="py-2.5 text-center w-24">Administrator</th>
                <th className="py-2.5 text-center w-24">Sales Manager</th>
                <th className="py-2.5 text-center w-24">Sales Rep</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs font-semibold text-brand-text">
              {matrix.map((row, idx) => (
                <tr key={row.key} className="hover:bg-slate-50/50">
                  <td className="py-3 font-extrabold text-slate-450 uppercase text-[9px] tracking-wide">{row.category}</td>
                  <td className="py-3 font-extrabold">{row.name}</td>
                  <td className="py-3 text-slate-450 text-[11px] font-medium leading-relaxed">{row.description}</td>
                  
                  {/* Admin Checkbox */}
                  <td className="py-3 text-center">
                    <button 
                      type="button" 
                      disabled
                      className="h-4.5 w-4.5 rounded border border-brand-border-purple bg-brand-accent/15 text-brand-accent flex items-center justify-center mx-auto cursor-not-allowed opacity-60"
                    >
                      <Check className="h-3 w-3" strokeWidth={3} />
                    </button>
                  </td>

                  {/* Manager Checkbox */}
                  <td className="py-3 text-center">
                    <button 
                      type="button"
                      onClick={() => togglePermission(idx, 'manager')}
                      className={`h-4.5 w-4.5 rounded border transition-all flex items-center justify-center mx-auto cursor-pointer ${
                        row.manager 
                          ? 'border-brand-accent bg-brand-accent text-white' 
                          : 'border-slate-300 hover:border-brand-accent'
                      }`}
                    >
                      {row.manager && <Check className="h-3 w-3" strokeWidth={3} />}
                    </button>
                  </td>

                  {/* Representative Checkbox */}
                  <td className="py-3 text-center">
                    <button 
                      type="button"
                      onClick={() => togglePermission(idx, 'representative')}
                      className={`h-4.5 w-4.5 rounded border transition-all flex items-center justify-center mx-auto cursor-pointer ${
                        row.representative 
                          ? 'border-brand-accent bg-brand-accent text-white' 
                          : 'border-slate-300 hover:border-brand-accent'
                      }`}
                    >
                      {row.representative && <Check className="h-3 w-3" strokeWidth={3} />}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
