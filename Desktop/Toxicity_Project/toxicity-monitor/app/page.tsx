"use client";

import { useState } from "react";

type CommentResult = {
  text: string;
  label: string;
  score: number;
  isToxic: boolean;
};

type AnalysisData = {
  overallToxicity: number;
  totalComments: number;
  toxicComments: number;
  comments: CommentResult[];
};

export default function Home() {
  const [hasStarted, setHasStarted] = useState(false);
  const [url, setUrl] = useState("");
  const [commentCount, setCommentCount] = useState(20);
  const [mode, setMode] = useState<"video" | "audio">("video");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioResult, setAudioResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<AnalysisData | null>(null);

  const handleAudioScan = async () => {
    if (!audioFile) return;
    setLoading(true);
    setError("");
    setAudioResult(null);

    const formData = new FormData();
    formData.append("file", audioFile);

    try {
      const res = await fetch("/api/audio", {
        method: "POST",
        body: formData,
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.error || "Audio analysis failed");
      
      setAudioResult(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleScan = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError("");
    setData(null);

    try {
      const res = await fetch("/api/toxicity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videoUrl: url, commentCount }),
      });

      const result = await res.json();

      if (!res.ok) {
        throw new Error(result.error || "Analysis failed");
      }

      setData(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container animate-fade-in" style={{ paddingTop: "4rem", paddingBottom: "4rem" }}>
      <header style={{ textAlign: "center", marginBottom: "3rem" }}>
        <h1 style={{ fontSize: "3.5rem", fontWeight: "700", marginBottom: "1rem", letterSpacing: "-1px" }}>
          Toxicity <span className="text-gradient-toxic">Monitor</span>
        </h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px", margin: "0 auto", fontSize: "1.2rem" }}>
          Instantly detect and analyze toxic behavior, hate speech, and sentiment in digital interactions. Powered by on-device ML.
        </p>
      </header>

      {!hasStarted ? (
        <section style={{ textAlign: "center", marginTop: "4rem" }} className="animate-fade-in">
          <button 
            className="btn-primary" 
            style={{ fontSize: "1.5rem", padding: "1.2rem 4rem", borderRadius: "50px", boxShadow: "0 10px 25px rgba(16, 185, 129, 0.4)", cursor: "pointer", transition: "all 0.3s ease" }}
            onClick={() => setHasStarted(true)}
            onMouseOver={(e: any) => e.currentTarget.style.transform = "scale(1.05)"}
            onMouseOut={(e: any) => e.currentTarget.style.transform = "scale(1)"}
          >
            Get Started
          </button>
        </section>
      ) : (
        <>
          {!data && !audioResult && (
            <section style={{ maxWidth: "600px", margin: "0 auto" }}>
          <div className={`glass-card scanner-box ${loading ? "scanning" : ""}`} style={{ textAlign: "center" }}>
            <div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginBottom: "1.5rem" }}>
              <button onClick={() => setMode("video")} style={{ flex: 1, padding: "0.75rem", borderRadius: "8px", background: mode === "video" ? "var(--accent-safe)" : "transparent", color: mode === "video" ? "#fff" : "var(--text-secondary)", border: "1px solid var(--accent-safe)", cursor: "pointer", fontWeight: "600" }}>🎥 Video URL</button>
              <button onClick={() => setMode("audio")} style={{ flex: 1, padding: "0.75rem", borderRadius: "8px", background: mode === "audio" ? "var(--accent-safe)" : "transparent", color: mode === "audio" ? "#fff" : "var(--text-secondary)", border: "1px solid var(--accent-safe)", cursor: "pointer", fontWeight: "600" }}>🎤 Upload Audio</button>
            </div>
            <h2 style={{ fontSize: "1.5rem", marginBottom: "1.5rem" }}>{mode === "video" ? "Scan a Conversation Thread" : "Analyze Audio Recording"}</h2>
            {mode === "video" ? (
            <div key="video-mode" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <input
                key="url-input"
                type="text"
                placeholder="Enter YouTube Video URL..."
                className="input-field"
                value={url || ""}
                onChange={(e) => setUrl(e.target.value)}
                disabled={loading}
              />
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", textAlign: "left", padding: "0 0.5rem" }}>
                <label style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                  Comments to Analyze: <span style={{ color: "var(--text-primary)", fontWeight: "600" }}>{commentCount}</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={commentCount || 20}
                  onChange={(e) => setCommentCount(Number(e.target.value))}
                  disabled={loading}
                  style={{ accentColor: "var(--accent-safe)", cursor: "pointer" }}
                />
              </div>
              <button 
                className="btn-primary" 
                onClick={handleScan}
                disabled={loading || !url.trim()}
              >
                {loading ? "Initializing AI Core..." : "Start Deep Scan"}
              </button>
            </div>
            ) : (
            <div key="audio-mode" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <input 
                key="file-input" 
                type="file" 
                accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac" 
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file && !file.type.startsWith("audio/") && !file.name.match(/\.(mp3|wav|m4a|ogg|flac)$/i)) {
                    setError("Invalid format! Please select a valid audio file (e.g., MP3, WAV, M4A).");
                    setAudioFile(null);
                    e.target.value = "";
                  } else {
                    setError("");
                    setAudioFile(file || null);
                  }
                }} 
                disabled={loading} 
                className="input-field" 
                style={{ padding: "1rem", cursor: "pointer" }} 
              />
              <button className="btn-primary" onClick={handleAudioScan} disabled={loading || !audioFile}>
                {loading ? "Processing Audio Model..." : "Analyze Audio"}
              </button>
            </div>
            )}
            
            {loading && (
              <p style={{ marginTop: "1.5rem", fontSize: "0.95rem", color: "var(--accent-neutral)", opacity: 0.8 }}>
                Downloading/Loading models and analyzing text...<br/>This may take a minute on the very first run.
              </p>
            )}

            {error && (
              <div style={{ marginTop: "1.5rem", padding: "1rem", borderRadius: "0.5rem", background: "rgba(239, 68, 68, 0.1)", color: "var(--accent-toxic)", border: "1px solid rgba(239, 68, 68, 0.3)" }}>
                Error: {error}
              </div>
            )}
          </div>
        </section>
      )}

      {data && (
        <section className="animate-fade-in">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
            <h2 style={{ fontSize: "2rem", fontWeight: "600" }}>Analysis Report</h2>
            <button className="btn-primary" onClick={() => setData(null)}>Scan Another Thread</button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "2rem" }}>
            {/* Overview Card */}
            <div className={`glass-card ${data.overallToxicity > 20 ? 'toxic-glow' : 'safe-glow'}`} style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyItems: "center", textAlign: "center", height: "fit-content" }}>
              <h3 style={{ color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "2px", fontSize: "0.85rem", marginBottom: "2rem" }}>System Toxicity Score</h3>
              
              <div style={{ position: "relative", width: "160px", height: "160px", display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "50%", background: "rgba(0,0,0,0.5)", border: `4px solid ${data.overallToxicity > 20 ? 'var(--accent-toxic)' : 'var(--accent-safe)'}`, boxShadow: `0 0 30px ${data.overallToxicity > 20 ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}` }}>
                <span style={{ fontSize: "3.5rem", fontWeight: "700", color: data.overallToxicity > 20 ? 'var(--accent-toxic)' : 'var(--accent-safe)' }}>
                  {data.overallToxicity}%
                </span>
              </div>

              <div style={{ marginTop: "3rem", display: "flex", width: "100%", justifyContent: "space-around", borderTop: "1px solid var(--glass-border)", paddingTop: "1.5rem" }}>
                <div>
                  <div style={{ fontSize: "2rem", fontWeight: "600" }}>{data.totalComments}</div>
                  <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>Analyzed</div>
                </div>
                <div>
                  <div style={{ fontSize: "2rem", fontWeight: "600", color: "var(--accent-toxic)" }}>{data.toxicComments}</div>
                  <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>Flags</div>
                </div>
              </div>
            </div>

            {/* Comments Stream */}
            <div className="glass-card" style={{ padding: "0" }}>
              <h3 style={{ fontSize: "1.2rem", padding: "1.5rem 1.5rem 0.5rem 1.5rem", borderBottom: "1px solid var(--glass-border)" }}>Conversation Stream</h3>
              <div style={{ maxHeight: "600px", overflowY: "auto", padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
                {data.comments.map((c, i) => (
                  <div 
                    key={i} 
                    style={{ 
                      padding: "1.25rem", 
                      borderRadius: "0.75rem",
                      background: c.isToxic ? "rgba(239, 68, 68, 0.05)" : "var(--glass-bg)",
                      border: "1px solid", 
                      borderColor: c.isToxic ? "rgba(239, 68, 68, 0.2)" : "var(--glass-border)",
                      animationDelay: `${i * 0.1}s`, 
                      opacity: 0, 
                      animation: "fadeIn 0.5s forwards" 
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                      <span style={{ 
                        fontSize: "0.75rem", 
                        padding: "0.25rem 0.75rem", 
                        borderRadius: "1rem", 
                        background: c.isToxic ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.1)",
                        color: c.isToxic ? "var(--accent-toxic)" : "var(--accent-safe)",
                        fontWeight: "600",
                        letterSpacing: "0.5px"
                      }}>
                        {c.isToxic ? "TOXIC" : "SAFE"}
                      </span>
                      <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                        Confidence: {(c.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p style={{ color: c.isToxic ? "var(--text-primary)" : "var(--text-secondary)", fontSize: "1rem", lineHeight: "1.5" }}>
                      "{c.text}"
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {audioResult && (
        <section className="animate-fade-in">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
             <h2 style={{ fontSize: "2rem", fontWeight: "600" }}>Audio Analysis Report</h2>
             <button className="btn-primary" onClick={() => setAudioResult(null)}>Analyze Another</button>
          </div>

          <div className={`glass-card ${audioResult.prediction === 'TOXIC' ? 'toxic-glow' : 'safe-glow'}`} style={{ textAlign: "center", padding: "3rem" }}>
             <h3 style={{ marginBottom: "1.5rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "2px" }}>Transcription</h3>
             <p style={{ fontSize: "1.2rem", lineHeight: "1.6", marginBottom: "3rem", fontStyle: "italic", background: "var(--glass-bg)", padding: "1.5rem", borderRadius: "10px" }}>
               "{audioResult.transcription}"
             </p>

             <div style={{ display: "flex", justifyContent: "space-around" }}>
                <div>
                  <div style={{ fontSize: "2rem", fontWeight: "700", color: audioResult.prediction === 'TOXIC' ? "var(--accent-toxic)" : "var(--accent-safe)" }}>
                     {audioResult.prediction}
                  </div>
                  <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "0.5rem" }}>Verdict</div>
                </div>
                <div>
                  <div style={{ fontSize: "2rem", fontWeight: "700" }}>
                     {(audioResult.toxicity_score * 100).toFixed(1)}%
                  </div>
                  <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "0.5rem" }}>Toxicity Confidence</div>
                </div>
             </div>
          </div>
        </section>
      )}
        </>
      )}
    </main>
  );
}
