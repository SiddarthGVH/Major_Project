"use client";
import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

type CommentResult = {
  text: string;
  label: string;
  score: number;
  isToxic: boolean;
};

interface Props {
  comments: CommentResult[];
}

export default function ToxicityCharts({ comments }: Props) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  if (!mounted) return <div style={{ height: "300px" }} />; // Placeholder to avoid layout shift

  // 1. Prepare Distribution Data (Histogram)
  const ranges = [
    { name: "0-20%", min: 0, max: 0.2, count: 0, color: "#10b981" },
    { name: "20-40%", min: 0.2, max: 0.4, count: 0, color: "#34d399" },
    { name: "40-60%", min: 0.4, max: 0.6, count: 0, color: "#fbbf24" },
    { name: "60-80%", min: 0.6, max: 0.8, count: 0, color: "#f87171" },
    { name: "80-100%", min: 0.8, max: 1.0, count: 0, color: "#ef4444" },
  ];

  comments.forEach((c) => {
    const range = ranges.find((r) => c.score >= r.min && c.score < r.max) || ranges[ranges.length - 1];
    range.count++;
  });

  // 2. Prepare Trend Data (Chronological)
  const trendData = comments.map((c, i) => ({
    index: i + 1,
    score: Math.round(c.score * 100),
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: "var(--bg-secondary)", border: "1px solid var(--border-subtle)", padding: "10px", borderRadius: "8px", boxShadow: "var(--shadow-md)" }}>
          <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>{`${payload[0].value}% Toxicity`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem", marginBottom: "3rem" }}>
      {/* Distribution Bar Chart */}
      <div className="enterprise-card" style={{ padding: "2rem" }}>
        <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-muted)", marginBottom: "1.5rem" }}>Toxicity Distribution</h3>
        <div style={{ width: "100%", height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ranges}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ fill: "var(--bg-elevated)", opacity: 0.4 }} content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {ranges.map((entry, index) => (
                  <Bar key={`cell-${index}`} dataKey="count" fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Trend Area Chart */}
      <div className="enterprise-card" style={{ padding: "2rem" }}>
        <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-muted)", marginBottom: "1.5rem" }}>Analysis Trendline</h3>
        <div style={{ width: "100%", height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--text-primary)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="var(--text-primary)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="index" stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} hide />
              <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="score" stroke="var(--text-primary)" strokeWidth={2} fillOpacity={1} fill="url(#colorScore)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
