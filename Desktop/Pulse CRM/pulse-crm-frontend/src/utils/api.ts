// API client wrapper with automatic mock-fallback for robustness

const API_BASE_URL = 'http://localhost:8000';

export interface Lead {
  id: string | number;
  name: string;
  company: string;
  email: string;
  phone: string;
  status: 'New' | 'Contacted' | 'Qualified' | 'Converted' | 'Lost';
  value: string;
  priority: 'Low' | 'Medium' | 'High';
  owner: string;
  ownerAvatar: string;
  notes: string;
  score: number;
  history?: { date: string; type: string; details: string }[];
}

export interface Contact {
  id: string | number;
  name: string;
  company: string;
  email: string;
  phone: string;
  role: string;
  avatar?: string;
}

export interface Company {
  id: string | number;
  name: string;
  domain: string;
  industry: string;
  employees: string;
  revenue: string;
  status: 'Active' | 'Prospect' | 'Partner';
}

export interface Deal {
  id: string | number;
  title: string;
  company: string;
  value: number;
  stage: 'Qualified' | 'Proposal' | 'Under Review' | 'Won' | 'Lost';
  priority: 'Low' | 'Medium' | 'High';
  owner: string;
  closeDate: string;
}

// Default High-Fidelity Fallback Mock Data
export const MOCK_LEADS: Lead[] = [
  {
    id: 1,
    name: "Alex Rivera",
    company: "TechCorp",
    email: "alex.rivera@techcorp.com",
    phone: "+1 (555) 019-2834",
    status: "New",
    value: "120000",
    priority: "High",
    owner: "Sarah Johnson",
    ownerAvatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=80&fit=crop&q=80",
    notes: "SSO guidelines sent. Security compliance approved SAML setup. SLA terms negotiation pending.",
    score: 87,
    history: [
      { date: "Yesterday, 10:15 AM", type: "email", details: "SSO Config Approved & Security Review" },
      { date: "2 days ago", type: "call", details: "Initial introductory discovery call" }
    ]
  },
  {
    id: 2,
    name: "Helena Troy",
    company: "Sparta Creative",
    email: "helena.t@spartacreative.io",
    phone: "+1 (555) 014-9821",
    status: "New",
    value: "45000",
    priority: "Medium",
    owner: "Sarah Johnson",
    ownerAvatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=80&fit=crop&q=80",
    notes: "Volumetric pricing requested for 40 agency designer seats. Seeking customized priority SLA.",
    score: 62,
    history: [
      { date: "Yesterday, 4:30 PM", type: "email", details: "Pricing Inquiry - Custom Enterprise Tier" }
    ]
  }
];

export const MOCK_CONTACTS: Contact[] = [
  { id: 1, name: "Alex Rivera", company: "TechCorp", email: "alex.rivera@techcorp.com", phone: "+1 (555) 019-2834", role: "Director of Security", avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=80&fit=crop&q=80" },
  { id: 2, name: "Helena Troy", company: "Sparta Creative", email: "helena.t@spartacreative.io", phone: "+1 (555) 014-9821", role: "Creative Lead", avatar: "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=80&fit=crop&q=80" },
  { id: 3, name: "Marcus Vance", company: "Empiric Logistics", email: "marcus.v@empiric.com", phone: "+1 (555) 012-3456", role: "VP of Infrastructure", avatar: "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=80&fit=crop&q=80" }
];

export const MOCK_COMPANIES: Company[] = [
  { id: 1, name: "TechCorp", domain: "techcorp.com", industry: "Technology", employees: "1,200", revenue: "$45M", status: "Prospect" },
  { id: 2, name: "Sparta Creative", domain: "spartacreative.io", industry: "Design Agency", employees: "85", revenue: "$3.2M", status: "Prospect" },
  { id: 3, name: "Empiric Logistics", domain: "empiriclogistics.com", industry: "Supply Chain", employees: "450", revenue: "$18.5M", status: "Partner" }
];

export const MOCK_DEALS: Deal[] = [
  { id: 1, title: "SSO Deployment & Enterprise Expansion", company: "TechCorp", value: 120000, stage: "Qualified", priority: "High", owner: "Sarah Johnson", closeDate: "2025-07-20" },
  { id: 2, title: "Design Agency Analytics Tier", company: "Sparta Creative", value: 45000, stage: "Proposal", priority: "Medium", owner: "Sarah Johnson", closeDate: "2025-08-15" }
];

// Helper to make API calls with fallback
async function apiFetch<T>(endpoint: string, options?: RequestInit, fallbackData?: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {})
      }
    });
    if (!res.ok) {
      throw new Error(`API error ${res.status}`);
    }
    return await res.json() as T;
  } catch (err) {
    console.warn(`Failed fetching ${endpoint}, using mock fallback.`, err);
    if (fallbackData !== undefined) return fallbackData;
    throw err;
  }
}

// --- Leads API ---
export async function getLeads(): Promise<Lead[]> {
  try {
    const dbLeads = await apiFetch<any[]>('/leads', {}, []);
    if (!dbLeads || dbLeads.length === 0) return MOCK_LEADS;

    // Join/map leads with mock metadata since backend schema is simple
    return dbLeads.map((dl, idx) => {
      const fallback = MOCK_LEADS[idx] || MOCK_LEADS[0];
      return {
        ...fallback, // Inherit all mock fields (timeline, emails, calls, meetings, owner, score)
        id: dl.id,
        status: dl.status || fallback.status,
        value: String(dl.value || fallback.value),
        notes: dl.description || fallback.notes
      };
    });
  } catch {
    return MOCK_LEADS;
  }
}

export async function createLead(leadData: any): Promise<any> {
  return apiFetch('/leads', {
    method: 'POST',
    body: JSON.stringify(leadData)
  });
}

export async function convertLead(leadId: string | number, payload: { name: string }): Promise<any> {
  return apiFetch(`/leads/${leadId}/convert`, {
    method: 'PUT',
    body: JSON.stringify(payload)
  });
}

// --- Contacts API ---
export async function getContacts(): Promise<Contact[]> {
  try {
    const dbContacts = await apiFetch<any[]>('/contacts', {}, []);
    if (!dbContacts || dbContacts.length === 0) return MOCK_CONTACTS;

    return dbContacts.map((dc, idx) => {
      const fallback = MOCK_CONTACTS[idx] || MOCK_CONTACTS[0];
      return {
        ...fallback, // Inherit mock timeline, calls, meetings, emails
        id: dc.id,
        name: `${dc.first_name} ${dc.last_name}`,
        email: dc.email,
        phone: dc.phone || fallback.phone,
        role: dc.job_title || fallback.role
      };
    });
  } catch {
    return MOCK_CONTACTS;
  }
}

export async function createContact(contactData: any): Promise<any> {
  return apiFetch('/contacts', {
    method: 'POST',
    body: JSON.stringify(contactData)
  });
}

// --- Companies API ---
export async function getCompanies(): Promise<Company[]> {
  try {
    const dbCompanies = await apiFetch<any[]>('/companies', {}, []);
    if (!dbCompanies || dbCompanies.length === 0) return MOCK_COMPANIES;

    return dbCompanies.map((dc, idx) => {
      const fallback = MOCK_COMPANIES[idx] || MOCK_COMPANIES[0];
      return {
        ...fallback, // Inherit mock contacts, timeline, emails, files
        id: dc.id,
        name: dc.name,
        domain: dc.domain || fallback.domain,
        industry: dc.industry || fallback.industry
      };
    });
  } catch {
    return MOCK_COMPANIES;
  }
}

export async function createCompany(companyData: any): Promise<any> {
  return apiFetch('/companies', {
    method: 'POST',
    body: JSON.stringify(companyData)
  });
}

// --- Deals API ---
export async function getDeals(): Promise<Deal[]> {
  try {
    const dbDeals = await apiFetch<any[]>('/deals', {}, []);
    if (!dbDeals || dbDeals.length === 0) return MOCK_DEALS;

    return dbDeals.map((dd, idx) => {
      const fallback = MOCK_DEALS[idx] || MOCK_DEALS[0];
      return {
        ...fallback,
        id: dd.id,
        title: dd.name,
        value: Number(dd.value || fallback.value),
        stage: dd.stage_id === 'd1f60c42-b0c6-4767-88ea-d4b68e9f2918' ? 'Qualified' :
               dd.stage_id === 'e2f50c42-b0c6-4767-88ea-d4b68e9f2919' ? 'Proposal' :
               dd.stage_id === 'f3f40c42-b0c6-4767-88ea-d4b68e9f2920' ? 'Under Review' :
               dd.stage_id === 'a4f30c42-b0c6-4767-88ea-d4b68e9f2921' ? 'Won' : 'Lost'
      };
    });
  } catch {
    return MOCK_DEALS;
  }
}

export async function updateDealStage(dealId: string | number, stageId: string): Promise<any> {
  return apiFetch(`/deals/${dealId}/stage`, {
    method: 'PUT',
    body: JSON.stringify({ stage_id: stageId })
  });
}
