'use client';

import React, { useState } from 'react';
import { Plus, Search, Trash2, Edit, X, Package, DollarSign, Tag, Info } from 'lucide-react';

interface Product {
  id: number;
  name: string;
  sku: string;
  category: string;
  price: number;
  status: 'Active' | 'Archived';
  dealsCount: number;
  description: string;
}

export default function ProductsView() {
  const [products, setProducts] = useState<Product[]>([
    { id: 1, name: "Enterprise Database Cloud License", sku: "DB-CLD-ENT", category: "Software Licensing", price: 15000, status: "Active", dealsCount: 14, description: "Full relational database cloud hosting license with auto-scale." },
    { id: 2, name: "HIPAA Security Compliance SLA Add-on", sku: "SEC-HIPAA-SLA", category: "Compliance & Security", price: 4500, status: "Active", dealsCount: 8, description: "End-to-end encryption audit pipeline log sync for health enterprise." },
    { id: 3, name: "Real-time AI Co-pilot Seat (Annual)", sku: "AI-COP-SEAT", category: "SaaS Subscription", price: 1200, status: "Active", dealsCount: 22, description: "Access key to real-time sync suggestions and leads scorer pipeline." },
    { id: 4, name: "Professional Services Migration (Day Rate)", sku: "MIG-PROF-SRV", category: "Professional Services", price: 2500, status: "Active", dealsCount: 5, description: "Dedicated database architecture integration specialist consultancy." },
    { id: 5, name: "SSO Identity Integration Gateway", sku: "GW-SSO-OAUTH", category: "Infrastructure", price: 6000, status: "Archived", dealsCount: 0, description: "Legacy SAML integration engine gateway module." }
  ]);

  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('All');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [form, setForm] = useState({ name: '', sku: '', category: 'Software Licensing', price: 0, status: 'Active' as Product['status'], description: '' });

  const categories = ['Software Licensing', 'Compliance & Security', 'SaaS Subscription', 'Professional Services', 'Infrastructure'];

  const filteredProducts = products.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase()) || p.sku.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = categoryFilter === 'All' || p.category === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  const handleAddProduct = (e: React.FormEvent) => {
    e.preventDefault();
    const newProduct: Product = {
      id: Date.now(),
      name: form.name,
      sku: form.sku,
      category: form.category,
      price: Number(form.price),
      status: form.status,
      dealsCount: 0,
      description: form.description
    };
    setProducts([newProduct, ...products]);
    setIsAddModalOpen(false);
    setForm({ name: '', sku: '', category: 'Software Licensing', price: 0, status: 'Active', description: '' });
  };

  const handleDelete = (id: number) => {
    setProducts(products.filter(p => p.id !== id));
  };

  return (
    <div className="space-y-6">
      <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
          <div>
            <h2 className="font-sans text-2xl text-brand-heading font-bold">Products Catalog</h2>
            <p className="text-[11px] text-brand-text/60 mt-0.5 font-bold">Configure CRM software keys, professional services templates, and pipeline license pricing.</p>
          </div>
          <button 
            onClick={() => setIsAddModalOpen(true)}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold shadow-sm/10 transition-colors cursor-pointer"
          >
            <Plus className="h-3.5 w-3.5" strokeWidth={2.25} />
            <span>Add Product</span>
          </button>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5">
          <div className="relative">
            <span className="absolute inset-y-0 left-2.5 flex items-center pointer-events-none text-slate-400">
              <Search className="h-3.5 w-3.5" />
            </span>
            <input 
              type="text" 
              placeholder="Search by name, SKU..." 
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text bg-slate-50/50 focus:bg-white placeholder-slate-400 focus:outline-none"
            />
          </div>

          <div>
            <select 
              value={categoryFilter}
              onChange={e => setCategoryFilter(e.target.value)}
              className="w-full px-3 py-1.5 border border-brand-border-purple/35 bg-white text-brand-text/80 rounded-lg text-xs focus:outline-none cursor-pointer"
            >
              <option value="All">All Categories</option>
              {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-brand-border-purple/20 text-[9px] uppercase font-extrabold tracking-wider text-brand-heading pb-2">
                <th className="pb-2">Product Info</th>
                <th className="pb-2">SKU</th>
                <th className="pb-2">Category</th>
                <th className="pb-2">Unit Price</th>
                <th className="pb-2 text-center">Active Deals</th>
                <th className="pb-2">Status</th>
                <th className="pb-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-brand-border-purple/15 text-xs text-brand-text font-semibold">
              {filteredProducts.length > 0 ? (
                filteredProducts.map((product) => (
                  <tr key={product.id} className="hover:bg-slate-50/30 transition-colors">
                    <td className="py-3 pr-4 max-w-[200px]">
                      <div className="font-extrabold text-brand-heading flex items-center gap-1.5">
                        <Package className="h-3.5 w-3.5 text-indigo-500 shrink-0" />
                        <span className="truncate">{product.name}</span>
                      </div>
                      <div className="text-[10px] text-brand-text/60 mt-0.5 font-medium truncate" title={product.description}>
                        {product.description}
                      </div>
                    </td>
                    <td className="py-3 font-mono text-[10px] text-slate-500">{product.sku}</td>
                    <td className="py-3 font-medium text-brand-text/80">{product.category}</td>
                    <td className="py-3 font-bold text-brand-heading tabular-nums">${product.price.toLocaleString()}</td>
                    <td className="py-3 text-center tabular-nums text-slate-500">{product.dealsCount}</td>
                    <td className="py-3">
                      <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        product.status === 'Active' ? 'text-emerald-700 bg-emerald-50 border border-emerald-100' : 'text-slate-650 bg-slate-50 border border-slate-100'
                      }`}>
                        {product.status}
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <button 
                        onClick={() => handleDelete(product.id)}
                        className="p-1 hover:text-rose-600 text-slate-400 rounded transition-colors"
                        title="Delete Product"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="text-center py-8 text-slate-400 font-medium">
                    No products found matching your filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Product Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
          <div className="bg-white border border-brand-border-purple/25 rounded-xl shadow-xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
            <div className="px-5 py-3.5 border-b border-brand-border-purple/15 flex justify-between items-center bg-slate-50">
              <h3 className="font-bold text-brand-heading text-sm">Add New Catalog Product</h3>
              <button onClick={() => setIsAddModalOpen(false)} className="text-slate-400 hover:text-brand-text p-1 cursor-pointer"><X className="h-4.5 w-4.5" /></button>
            </div>
            <form onSubmit={handleAddProduct} className="p-5 space-y-4">
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Product Name</label>
                <input type="text" required value={form.name} onChange={e => setForm({...form, name: e.target.value})} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none focus:ring-1 focus:ring-brand-accent/20" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">SKU Code</label>
                  <input type="text" required value={form.sku} onChange={e => setForm({...form, sku: e.target.value})} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none focus:ring-1 focus:ring-brand-accent/20" />
                </div>
                <div>
                  <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Unit Price ($)</label>
                  <input type="number" required value={form.price} onChange={e => setForm({...form, price: Number(e.target.value)})} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none focus:ring-1 focus:ring-brand-accent/20" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Category</label>
                  <select value={form.category} onChange={e => setForm({...form, category: e.target.value})} className="w-full px-2 py-1.5 border border-brand-border-purple/35 bg-white text-brand-text rounded-lg text-xs focus:outline-none cursor-pointer">
                    {categories.map(cat => <option key={cat}>{cat}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Status</label>
                  <select value={form.status} onChange={e => setForm({...form, status: e.target.value as any})} className="w-full px-2 py-1.5 border border-brand-border-purple/35 bg-white text-brand-text rounded-lg text-xs focus:outline-none cursor-pointer">
                    <option>Active</option>
                    <option>Archived</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Description</label>
                <textarea rows={3} value={form.description} onChange={e => setForm({...form, description: e.target.value})} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none focus:ring-1 focus:ring-brand-accent/20 resize-none" />
              </div>
              <div className="pt-3 border-t border-brand-border-purple/15 flex justify-end space-x-2.5">
                <button type="button" onClick={() => setIsAddModalOpen(false)} className="px-4 py-1.5 border border-brand-border-purple/30 rounded-lg text-xs font-bold text-brand-text/75 hover:bg-slate-50 cursor-pointer">Cancel</button>
                <button type="submit" className="px-4 py-1.5 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold shadow-sm/10 cursor-pointer">Add Product</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
