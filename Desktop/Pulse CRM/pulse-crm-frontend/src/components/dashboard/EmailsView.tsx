'use client';

import React, { useState } from 'react';
import { 
  Inbox, 
  Send, 
  FileText, 
  Mail, 
  Search, 
  Plus, 
  CornerUpLeft, 
  CornerUpRight, 
  Paperclip, 
  Sparkles, 
  X, 
  Check,
  ChevronRight,
  User
} from 'lucide-react';

interface EmailThread {
  id: number;
  sender: string;
  senderEmail: string;
  subject: string;
  body: string;
  time: string;
  folder: 'inbox' | 'sent' | 'drafts';
  aiSummary: string;
  unread: boolean;
  attachments: { name: string; size: string }[];
}

export default function EmailsView() {
  const [threads, setThreads] = useState<EmailThread[]>([
    {
      id: 1,
      sender: "Alex Rivera",
      senderEmail: "alex.rivera@techcorp.com",
      subject: "SSO Config Approved & Security Review",
      body: "Hi Sarah,\n\nWe reviewed the SSO guidelines you sent. Our compliance team approved the SAML setup but they have some questions regarding custom SLAs and liability limits. Can we schedule a quick call tomorrow to clarify these items?\n\nBest,\nAlex",
      time: "10:15 AM",
      folder: "inbox",
      aiSummary: "Prospect approved SAML config specs but has questions about liability limits and SLAs. Recommends arranging clarification call.",
      unread: true,
      attachments: [{ name: "SAML_Approval.docx", size: "110 KB" }]
    },
    {
      id: 2,
      sender: "Helena Troy",
      senderEmail: "helena.t@spartacreative.io",
      subject: "Pricing Inquiry - Custom Enterprise Tier",
      body: "Hello Sarah,\n\nI was looking over the custom analytics dashboard tiers on your pricing page. We have around 40 designers who need priority SLA support. Do you support volumetric discounts for design agencies? Let me know.\n\nThanks,\nHelena",
      time: "Yesterday",
      folder: "inbox",
      aiSummary: "Helena inquires about agency volume pricing tiers for 40 seats. Needs response regarding custom priority SLA.",
      unread: false,
      attachments: []
    },
    {
      id: 3,
      sender: "Sarah Johnson",
      senderEmail: "sarah.j@pulse.crm",
      subject: "Re: Security compliance review",
      body: "Hi Marcus,\n\nI have attached our HIPAA and SOC2 compliance audit folders. Let me know if you need our lead architect on the sandbox review call.\n\nSarah Johnson",
      time: "2 days ago",
      folder: "sent",
      aiSummary: "Sent compliance audit folders (HIPAA/SOC2) to Marcus to coordinate sandbox reviews.",
      unread: false,
      attachments: [{ name: "Pulse_SOC2_HIPAA.zip", size: "4.5 MB" }]
    }
  ]);

  const [activeFolder, setActiveFolder] = useState<'inbox' | 'sent' | 'drafts'>('inbox');
  const [selectedThreadId, setSelectedThreadId] = useState(1);
  const [search, setSearch] = useState('');
  
  // Modals / forms state
  const [isComposeOpen, setIsComposeOpen] = useState(false);
  const [isReplyOpen, setIsReplyOpen] = useState(false);
  const [isForwardOpen, setIsForwardOpen] = useState(false);

  const [composeForm, setComposeForm] = useState({ to: '', subject: '', body: '' });
  const [replyBody, setReplyBody] = useState('');
  const [forwardTo, setForwardTo] = useState('');

  const activeThread = threads.find(t => t.id === selectedThreadId) || threads[0];

  const folderThreads = threads.filter(t => t.folder === activeFolder);
  const filteredThreads = folderThreads.filter(t => 
    t.sender.toLowerCase().includes(search.toLowerCase()) || 
    t.subject.toLowerCase().includes(search.toLowerCase()) || 
    t.body.toLowerCase().includes(search.toLowerCase())
  );

  const handleCompose = (e: React.FormEvent) => {
    e.preventDefault();
    const newMail: EmailThread = {
      id: Date.now(),
      sender: "Sarah Johnson",
      senderEmail: "sarah.j@pulse.crm",
      subject: composeForm.subject,
      body: composeForm.body,
      time: "Just now",
      folder: "sent",
      aiSummary: `Sent message to ${composeForm.to}. Subject: ${composeForm.subject}`,
      unread: false,
      attachments: []
    };
    setThreads([newMail, ...threads]);
    setSelectedThreadId(newMail.id);
    setIsComposeOpen(false);
    setComposeForm({ to: '', subject: '', body: '' });
  };

  const handleReply = (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyBody.trim()) return;
    const newReply: EmailThread = {
      id: Date.now(),
      sender: "Sarah Johnson",
      senderEmail: "sarah.j@pulse.crm",
      subject: `Re: ${activeThread.subject}`,
      body: replyBody,
      time: "Just now",
      folder: "sent",
      aiSummary: `Replied to ${activeThread.sender}. Subject: Re: ${activeThread.subject}`,
      unread: false,
      attachments: []
    };
    setThreads([newReply, ...threads]);
    setSelectedThreadId(newReply.id);
    setIsReplyOpen(false);
    setReplyBody('');
  };

  const handleForward = (e: React.FormEvent) => {
    e.preventDefault();
    if (!forwardTo.trim()) return;
    const newForward: EmailThread = {
      id: Date.now(),
      sender: "Sarah Johnson",
      senderEmail: "sarah.j@pulse.crm",
      subject: `Fwd: ${activeThread.subject}`,
      body: `---------- Forwarded message ----------\nFrom: ${activeThread.sender} <${activeThread.senderEmail}>\nSubject: ${activeThread.subject}\n\n${activeThread.body}`,
      time: "Just now",
      folder: "sent",
      aiSummary: `Forwarded thread: Fwd: ${activeThread.subject} to ${forwardTo}`,
      unread: false,
      attachments: activeThread.attachments
    };
    setThreads([newForward, ...threads]);
    setSelectedThreadId(newForward.id);
    setIsForwardOpen(false);
    setForwardTo('');
  };

  return (
    <div className="grid grid-cols-12 gap-6 items-start h-[650px]">
      
      {/* Email folders list (Col 2) */}
      <div className="col-span-12 md:col-span-3 lg:col-span-2 space-y-4">
        <button 
          onClick={() => setIsComposeOpen(true)}
          className="w-full flex items-center justify-center space-x-1.5 bg-brand-accent hover:bg-brand-accent-hover text-white py-2 px-4 rounded-lg text-xs font-bold shadow-sm cursor-pointer"
        >
          <Plus className="h-4 w-4" />
          <span>Compose</span>
        </button>

        <nav className="space-y-1 bg-white border border-brand-border-purple/20 rounded-xl p-2.5">
          {[
            { id: 'inbox', label: 'Inbox', icon: Inbox, count: threads.filter(t => t.folder === 'inbox' && t.unread).length },
            { id: 'sent', label: 'Sent', icon: Send, count: 0 },
            { id: 'drafts', label: 'Drafts', icon: FileText, count: threads.filter(t => t.folder === 'drafts').length }
          ].map((fol) => {
            const Icon = fol.icon;
            const isSelected = activeFolder === fol.id;
            return (
              <button
                key={fol.id}
                onClick={() => {
                  setActiveFolder(fol.id as any);
                  const first = threads.find(t => t.folder === fol.id);
                  if (first) setSelectedThreadId(first.id);
                }}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  isSelected 
                    ? 'bg-brand-secondary-accent/15 text-brand-accent' 
                    : 'hover:bg-slate-50 text-brand-text/75 hover:text-brand-text'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <Icon className="h-4 w-4 shrink-0" />
                  <span>{fol.label}</span>
                </div>
                {fol.count > 0 && (
                  <span className="text-[9px] font-extrabold bg-brand-accent text-white px-1.5 py-0.25 rounded-full tabular-nums">
                    {fol.count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Email List Threads (Col 4) */}
      <div className="col-span-12 md:col-span-4 lg:col-span-4 space-y-4 h-full flex flex-col">
        <div className="bg-white border border-brand-border-purple/20 rounded-xl p-4 flex-1 overflow-y-auto flex flex-col h-full">
          <div className="relative mb-3 shrink-0">
            <span className="absolute inset-y-0 left-2.5 flex items-center pointer-events-none text-slate-400">
              <Search className="h-3.5 w-3.5" />
            </span>
            <input 
              type="text" 
              placeholder="Search mail..." 
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none"
            />
          </div>

          <div className="flex-1 space-y-2.5 overflow-y-auto pr-1">
            {filteredThreads.length > 0 ? (
              filteredThreads.map((thread) => {
                const isSelected = thread.id === selectedThreadId;
                return (
                  <div
                    key={thread.id}
                    onClick={() => {
                      setSelectedThreadId(thread.id);
                      if (thread.unread) {
                        setThreads(threads.map(t => t.id === thread.id ? { ...t, unread: false } : t));
                      }
                    }}
                    className={`p-3 border rounded-xl cursor-pointer hover:border-brand-border-purple/40 transition-all ${
                      isSelected 
                        ? 'border-brand-border-purple bg-brand-secondary-accent/10' 
                        : 'border-brand-border-purple/20 bg-white'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <span className={`text-[10px] font-extrabold truncate max-w-[120px] ${thread.unread ? 'text-brand-heading' : 'text-brand-text/80'}`}>
                        {thread.sender}
                      </span>
                      <span className="text-[9px] text-slate-450 font-bold tabular-nums shrink-0">{thread.time}</span>
                    </div>
                    <h4 className={`text-xs mt-1 truncate ${thread.unread ? 'font-extrabold text-brand-heading' : 'font-bold text-brand-text/90'}`}>
                      {thread.subject}
                    </h4>
                    <p className="text-[10px] text-brand-text/60 mt-1 line-clamp-1 leading-normal font-semibold">
                      {thread.body}
                    </p>
                    
                    {/* Tiny AI tag */}
                    <div className="mt-2.5 pt-2 border-t border-brand-border-purple/10 flex items-center text-[8px] font-bold text-brand-accent">
                      <Sparkles className="h-2.5 w-2.5 mr-1" />
                      <span>AI Summary Available</span>
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="text-slate-400 text-center py-8 text-[11px] font-bold">No mail threads here.</p>
            )}
          </div>
        </div>
      </div>

      {/* Email Reader Pane (Col 6) */}
      <div className="col-span-12 md:col-span-5 lg:col-span-6 h-full">
        {activeThread ? (
          <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm/5 h-full flex flex-col overflow-y-auto">
            {/* Subject */}
            <div className="border-b border-brand-border-purple/15 pb-3 shrink-0">
              <h3 className="text-sm font-extrabold text-brand-heading leading-snug">{activeThread.subject}</h3>
              <div className="flex justify-between items-center mt-2 text-[10px] font-semibold text-brand-text/60">
                <div>
                  <span className="font-extrabold text-brand-heading">{activeThread.sender}</span>
                  <span className="ml-1 font-medium">&lt;{activeThread.senderEmail}&gt;</span>
                </div>
                <span className="tabular-nums font-bold">{activeThread.time}</span>
              </div>
            </div>

            {/* AI Summary Banner */}
            <div className="mt-3.5 bg-brand-sidebar-hover/15 border border-brand-border-purple/25 rounded-xl p-3.5 flex items-start space-x-2 shrink-0">
              <Sparkles className="h-4.5 w-4.5 text-brand-accent shrink-0 mt-0.5" />
              <div>
                <h4 className="text-[9px] font-extrabold text-brand-heading uppercase tracking-wider">AI Email Summary</h4>
                <p className="text-[10px] text-brand-text/80 mt-1 leading-relaxed font-bold">{activeThread.aiSummary}</p>
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 py-5 text-xs text-brand-text font-semibold leading-relaxed whitespace-pre-line overflow-y-auto min-h-[120px]">
              {activeThread.body}
            </div>

            {/* Attachments */}
            {activeThread.attachments && activeThread.attachments.length > 0 && (
              <div className="border-t border-brand-border-purple/10 py-3 shrink-0">
                <h4 className="text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-2">Attachments</h4>
                <div className="flex flex-wrap gap-2">
                  {activeThread.attachments.map((file, idx) => (
                    <div key={idx} className="p-2 border border-brand-border-purple/15 rounded bg-slate-50/50 flex items-center text-[10px] font-semibold">
                      <Paperclip className="h-3.5 w-3.5 mr-1.5 text-slate-400" />
                      <span className="font-bold text-brand-heading mr-2">{file.name}</span>
                      <span className="text-slate-400">({file.size})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reader Action Triggers */}
            <div className="border-t border-brand-border-purple/15 pt-3.5 flex space-x-2.5 shrink-0">
              <button 
                onClick={() => setIsReplyOpen(true)}
                className="inline-flex items-center space-x-1 px-3 py-1.5 border border-brand-border-purple/35 hover:border-brand-border-purple text-brand-text/80 text-xs font-bold rounded-lg cursor-pointer"
              >
                <CornerUpLeft className="h-4 w-4" />
                <span>Reply</span>
              </button>
              <button 
                onClick={() => setIsForwardOpen(true)}
                className="inline-flex items-center space-x-1 px-3 py-1.5 border border-brand-border-purple/35 hover:border-brand-border-purple text-brand-text/80 text-xs font-bold rounded-lg cursor-pointer"
              >
                <CornerUpRight className="h-4 w-4" />
                <span>Forward</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-brand-border-purple/20 rounded-xl p-5 shadow-sm h-full flex items-center justify-center text-slate-400 font-bold text-xs">
            Select a mail thread to view.
          </div>
        )}
      </div>

      {/* Compose Modal */}
      {isComposeOpen && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
          <div className="bg-white border border-brand-border-purple/25 rounded-xl shadow-xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
            <div className="px-5 py-3.5 border-b border-brand-border-purple/15 flex justify-between items-center bg-slate-50">
              <h3 className="font-bold text-brand-heading text-sm">New Message</h3>
              <button onClick={() => setIsComposeOpen(false)} className="text-slate-400 hover:text-brand-text p-1 cursor-pointer"><X className="h-4.5 w-4.5" /></button>
            </div>
            <form onSubmit={handleCompose} className="p-5 space-y-4">
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">To</label>
                <input type="email" required placeholder="recipient@company.com" value={composeForm.to} onChange={e => setComposeForm({...composeForm, to: e.target.value})} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none" />
              </div>
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Subject</label>
                <input type="text" required placeholder="Subject line" value={composeForm.subject} onChange={e => setComposeForm({...composeForm, subject: e.target.value})} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none" />
              </div>
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Message Body</label>
                <textarea required placeholder="Write email..." value={composeForm.body} onChange={e => setComposeForm({...composeForm, body: e.target.value})} className="w-full p-2.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none min-h-[120px]" />
              </div>
              <div className="pt-3 border-t border-brand-border-purple/15 flex justify-end space-x-2.5">
                <button type="button" onClick={() => setIsComposeOpen(false)} className="px-4 py-1.5 border border-brand-border-purple/30 rounded-lg text-xs font-bold text-brand-text/75 hover:bg-slate-50 cursor-pointer">Cancel</button>
                <button type="submit" className="px-4 py-1.5 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold shadow-sm/10 cursor-pointer">Send</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Reply Modal */}
      {isReplyOpen && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
          <div className="bg-white border border-brand-border-purple/25 rounded-xl shadow-xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
            <div className="px-5 py-3.5 border-b border-brand-border-purple/15 flex justify-between items-center bg-slate-50">
              <h3 className="font-bold text-brand-heading text-sm">Reply to {activeThread.sender}</h3>
              <button onClick={() => setIsReplyOpen(false)} className="text-slate-400 hover:text-brand-text p-1 cursor-pointer"><X className="h-4.5 w-4.5" /></button>
            </div>
            <form onSubmit={handleReply} className="p-5 space-y-4">
              <div>
                <label className="block text-[9px] font-extrabold text-slate-450 uppercase tracking-wider mb-1">Original Subject</label>
                <div className="text-xs text-brand-text font-bold bg-slate-50 p-2 rounded-lg border border-brand-border-purple/10">{activeThread.subject}</div>
              </div>
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Reply Message</label>
                <textarea required placeholder="Write reply message..." value={replyBody} onChange={e => setReplyBody(e.target.value)} className="w-full p-2.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none min-h-[120px]" />
              </div>
              <div className="pt-3 border-t border-brand-border-purple/15 flex justify-end space-x-2.5">
                <button type="button" onClick={() => setIsReplyOpen(false)} className="px-4 py-1.5 border border-brand-border-purple/30 rounded-lg text-xs font-bold text-brand-text/75 hover:bg-slate-50 cursor-pointer">Cancel</button>
                <button type="submit" className="px-4 py-1.5 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold shadow-sm/10 cursor-pointer">Send Reply</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Forward Modal */}
      {isForwardOpen && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
          <div className="bg-white border border-brand-border-purple/25 rounded-xl shadow-xl w-full max-w-md overflow-hidden animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
            <div className="px-5 py-3.5 border-b border-brand-border-purple/15 flex justify-between items-center bg-slate-50">
              <h3 className="font-bold text-brand-heading text-sm">Forward Thread</h3>
              <button onClick={() => setIsForwardOpen(false)} className="text-slate-400 hover:text-brand-text p-1 cursor-pointer"><X className="h-4.5 w-4.5" /></button>
            </div>
            <form onSubmit={handleForward} className="p-5 space-y-4">
              <div>
                <label className="block text-[9px] font-extrabold text-brand-heading uppercase tracking-wider mb-1">Forward To</label>
                <input type="email" required placeholder="recipient@company.com" value={forwardTo} onChange={e => setForwardTo(e.target.value)} className="w-full px-3 py-1.5 border border-brand-border-purple/35 rounded-lg text-xs text-brand-text focus:outline-none" />
              </div>
              <div>
                <label className="block text-[9px] font-extrabold text-slate-450 uppercase tracking-wider mb-1">Forwarding content preview</label>
                <div className="text-[10px] text-brand-text/75 font-semibold bg-slate-50 p-2 rounded-lg border border-brand-border-purple/15 max-h-36 overflow-y-auto whitespace-pre-line leading-relaxed">
                  {activeThread.body}
                </div>
              </div>
              <div className="pt-3 border-t border-brand-border-purple/15 flex justify-end space-x-2.5">
                <button type="button" onClick={() => setIsForwardOpen(false)} className="px-4 py-1.5 border border-brand-border-purple/30 rounded-lg text-xs font-bold text-brand-text/75 hover:bg-slate-50 cursor-pointer">Cancel</button>
                <button type="submit" className="px-4 py-1.5 bg-brand-accent hover:bg-brand-accent-hover text-white rounded-lg text-xs font-bold shadow-sm/10 cursor-pointer">Forward Mail</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
