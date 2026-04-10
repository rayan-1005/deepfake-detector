import { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import "./index.css";

const API = "http://localhost:8000";

function ConfidenceBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: "12px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "6px",
        }}
      >
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: "12px",
            color: "var(--muted)",
          }}
        >
          {label}
        </span>
        <span style={{ fontFamily: "var(--mono)", fontSize: "12px", color }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div
        style={{
          height: "4px",
          background: "var(--border)",
          borderRadius: "2px",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${value * 100}%`,
            background: color,
            borderRadius: "2px",
            transition: "width 1s cubic-bezier(0.16,1,0.3,1)",
            boxShadow: `0 0 12px ${color}88`,
          }}
        />
      </div>
    </div>
  );
}

function FrameTimeline({ preds }) {
  return (
    <div style={{ marginTop: "24px" }}>
      <p
        style={{
          fontFamily: "var(--mono)",
          fontSize: "11px",
          color: "var(--muted)",
          marginBottom: "10px",
        }}
      >
        FRAME ANALYSIS — {preds.length} SAMPLES
      </p>
      <div style={{ display: "flex", gap: "3px", flexWrap: "wrap" }}>
        {preds.map((p, i) => (
          <div
            key={i}
            style={{
              width: "14px",
              height: "14px",
              borderRadius: "2px",
              background: p === 0 ? "var(--danger)" : "var(--accent)",
              opacity: 0.85,
              boxShadow:
                p === 0 ? "0 0 6px var(--danger)" : "0 0 6px var(--accent)",
            }}
            title={`Frame ${i + 1}: ${p === 0 ? "FAKE" : "REAL"}`}
          />
        ))}
      </div>
      <div style={{ display: "flex", gap: "16px", marginTop: "10px" }}>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: "10px",
            color: "var(--accent)",
          }}
        >
          ■ REAL
        </span>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: "10px",
            color: "var(--danger)",
          }}
        >
          ■ FAKE
        </span>
      </div>
    </div>
  );
}

// --- NEW COMPONENT: Analysis Dashboard ---
function AnalysisDashboard() {
  const [step, setStep] = useState(0);

  // Cycle through analysis steps to make the UI feel alive
  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => (s + 1) % 4);
    }, 4500);
    return () => clearInterval(timer);
  }, []);

  const steps = [
    {
      title: "FRAME SAMPLING & LANDMARKING",
      desc: "Extracting frames and utilizing MediaPipe to map 468 3D facial landmarks. Locating topographical boundaries.",
    },
    {
      title: "MICRO-ARTIFACT ISOLATION",
      desc: "Cleaving the face into 3 distinct zones. Deepfakes frequently leave micro-artifacts in ocular and nasal blending boundaries.",
    },
    {
      title: "TRI-HEAD CNN ENSEMBLE",
      desc: "EfficientNetB0 scans the full 224x224 face crop. Twin MobileNetV2 networks hunt for sub-pixel anomalies in the eyes and nose.",
    },
    {
      title: "PROBABILISTIC AGGREGATION",
      desc: "Averaging model confidence across all regions and frames. Calculating final neural verdict against the threshold.",
    },
  ];

  return (
    <div
      style={{
        marginTop: "32px",
        width: "100%",
        maxWidth: "560px",
        border: "1px solid var(--border)",
        borderRadius: "4px",
        padding: "32px",
        background: "var(--surface)",
        animation: "fadeIn 0.4s ease",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "32px",
          borderBottom: "1px solid var(--border)",
          paddingBottom: "16px",
        }}
      >
        <p
          style={{
            fontFamily: "var(--mono)",
            fontSize: "12px",
            color: "var(--accent)",
            animation: "pulse 1.5s infinite",
          }}
        >
          ● RUNNING NEURAL INFERENCE
        </p>
        <div
          style={{
            width: "24px",
            height: "24px",
            border: "1px solid var(--border)",
            borderTop: "1px solid var(--accent)",
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
          }}
        />
      </div>

      {/* Animated Architecture Diagram */}
      <div style={{ position: "relative", marginBottom: "40px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "12px",
            marginBottom: "16px",
          }}
        >
          <div className="arch-box">
            FULL FACE
            <br />
            <span>224×224</span>
          </div>
          <div className="arch-box">
            OCULAR
            <br />
            <span>128×128</span>
          </div>
          <div className="arch-box">
            NASAL
            <br />
            <span>128×128</span>
          </div>
        </div>

        {/* Animated flow lines */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "60px",
            marginBottom: "16px",
          }}
        >
          <div className="flow-line" />
          <div className="flow-line" />
          <div className="flow-line" />
        </div>

        <div style={{ display: "flex", justifyContent: "center", gap: "12px" }}>
          <div
            className="arch-model"
            style={{
              borderColor: step === 2 ? "var(--accent)" : "var(--border)",
            }}
          >
            EfficientNetB0
          </div>
          <div
            className="arch-model"
            style={{
              borderColor: step === 2 ? "var(--accent)" : "var(--border)",
            }}
          >
            MobileNetV2
          </div>
          <div
            className="arch-model"
            style={{
              borderColor: step === 2 ? "var(--accent)" : "var(--border)",
            }}
          >
            MobileNetV2
          </div>
        </div>
      </div>

      {/* Explainer Text Box */}
      <div
        style={{
          background: "#0a0a0a",
          padding: "16px",
          borderRadius: "2px",
          borderLeft: "2px solid var(--accent)",
        }}
      >
        <p
          style={{
            fontFamily: "var(--mono)",
            fontSize: "10px",
            color: "var(--muted)",
            letterSpacing: "2px",
            marginBottom: "8px",
          }}
        >
          ACTIVE PROCESS: {step + 1}/4
        </p>
        <p
          key={step} // Forces re-render animation on step change
          style={{
            fontFamily: "var(--sans)",
            fontSize: "14px",
            color: "var(--text)",
            lineHeight: 1.5,
            animation: "fadeInRight 0.5s ease forwards",
          }}
        >
          <strong
            style={{
              color: "var(--accent)",
              display: "block",
              marginBottom: "4px",
            }}
          >
            {steps[step].title}
          </strong>
          {steps[step].desc}
        </p>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
        @keyframes fadeInRight { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: none; } }
        @keyframes flowDown { 0% { background-position: 0 -20px; } 100% { background-position: 0 20px; } }
        
        .arch-box {
          flex: 1;
          text-align: center;
          border: 1px dashed var(--muted);
          padding: 8px 4px;
          font-family: var(--mono);
          font-size: 10px;
          color: var(--muted);
          border-radius: 2px;
        }
        .arch-box span { color: var(--accent); font-size: 9px; }
        
        .arch-model {
          flex: 1;
          text-align: center;
          border: 1px solid var(--border);
          padding: 10px 4px;
          font-family: var(--mono);
          font-size: 11px;
          color: var(--text);
          background: #0a0a0a;
          transition: border-color 0.3s;
        }

        .flow-line {
          width: 2px;
          height: 24px;
          background: linear-gradient(to bottom, transparent, var(--accent), transparent);
          background-size: 100% 200%;
          animation: flowDown 1s linear infinite;
        }
      `}</style>
    </div>
  );
}

export default function App() {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("video/")) return;
    setFile(f);
    setResult(null);
    setError(null);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API}/predict`, form);
      setResult(data);
    } catch (e) {
      setError("Server error. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  const isFake = result?.prediction === "fake";

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "40px 20px",
      }}
    >
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "48px" }}>
        <p
          style={{
            fontFamily: "var(--mono)",
            fontSize: "11px",
            color: "var(--accent)",
            letterSpacing: "4px",
            marginBottom: "12px",
          }}
        >
          NEURAL FORENSICS v1.0
        </p>
        <h1
          style={{
            fontFamily: "var(--sans)",
            fontWeight: 700,
            fontSize: "clamp(36px, 6vw, 64px)",
            letterSpacing: "-1px",
            lineHeight: 1,
          }}
        >
          DEEPFAKE
          <br />
          <span style={{ color: "var(--accent)" }}>DETECTOR</span>
        </h1>
        <p
          style={{
            fontFamily: "var(--mono)",
            fontSize: "12px",
            color: "var(--muted)",
            marginTop: "16px",
          }}
        >
          EfficientNetB0 + MobileNetV2 ensemble · Face / Eyes / Nose analysis
        </p>
      </div>

      {/* Upload zone */}
      <div
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          width: "100%",
          maxWidth: "560px",
          border: `1px solid ${dragging ? "var(--accent)" : file ? "#2a2a2a" : "var(--border)"}`,
          borderRadius: "4px",
          padding: "48px 32px",
          textAlign: "center",
          cursor: "pointer",
          transition: "all 0.2s",
          background: dragging ? "rgba(0,255,157,0.03)" : "var(--surface)",
          boxShadow: dragging ? "0 0 30px rgba(0,255,157,0.1)" : "none",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* corner accents */}
        {["topLeft", "topRight", "bottomLeft", "bottomRight"].map((pos) => (
          <div
            key={pos}
            style={{
              position: "absolute",
              width: "12px",
              height: "12px",
              borderColor: "var(--accent)",
              borderStyle: "solid",
              borderWidth: pos.includes("top") ? "1px 0 0 1px" : "0 1px 1px 0",
              ...(pos.includes("top") ? { top: 0 } : { bottom: 0 }),
              ...(pos.includes("Left") ? { left: 0 } : { right: 0 }),
            }}
          />
        ))}

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {file ? (
          <>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--accent)",
                letterSpacing: "2px",
              }}
            >
              FILE LOADED
            </p>
            <p
              style={{
                fontFamily: "var(--sans)",
                fontSize: "18px",
                fontWeight: 600,
                marginTop: "8px",
              }}
            >
              {file.name}
            </p>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginTop: "4px",
              }}
            >
              {(file.size / 1e6).toFixed(1)} MB
            </p>
          </>
        ) : (
          <>
            <div
              style={{ fontSize: "32px", marginBottom: "16px", opacity: 0.3 }}
            >
              ▲
            </div>
            <p
              style={{
                fontFamily: "var(--sans)",
                fontSize: "18px",
                fontWeight: 600,
              }}
            >
              Drop video here
            </p>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginTop: "8px",
              }}
            >
              or click to browse · MP4 recommended
            </p>
          </>
        )}
      </div>

      {/* Analyze button */}
      {file && !loading && !result && (
        <button
          onClick={analyze}
          style={{
            marginTop: "20px",
            padding: "14px 48px",
            background: "transparent",
            border: "1px solid var(--accent)",
            color: "var(--accent)",
            fontFamily: "var(--mono)",
            fontSize: "13px",
            letterSpacing: "3px",
            cursor: "pointer",
            transition: "all 0.2s",
            borderRadius: "2px",
          }}
          onMouseEnter={(e) => {
            e.target.style.background = "var(--accent)";
            e.target.style.color = "#000";
          }}
          onMouseLeave={(e) => {
            e.target.style.background = "transparent";
            e.target.style.color = "var(--accent)";
          }}
        >
          ANALYZE
        </button>
      )}

      {/* Loading Overlay */}
      {loading && <AnalysisDashboard />}

      {/* Error */}
      {error && (
        <p
          style={{
            marginTop: "20px",
            fontFamily: "var(--mono)",
            fontSize: "12px",
            color: "var(--danger)",
          }}
        >
          {error}
        </p>
      )}

      {/* Result */}
      {result && (
        <div
          style={{
            marginTop: "32px",
            width: "100%",
            maxWidth: "560px",
            border: `1px solid ${isFake ? "var(--danger)" : "var(--accent)"}`,
            borderRadius: "4px",
            padding: "32px",
            background: "var(--surface)",
            boxShadow: isFake
              ? "0 0 40px rgba(255,45,85,0.08)"
              : "0 0 40px rgba(0,255,157,0.08)",
            animation: "fadeIn 0.4s ease",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "28px",
            }}
          >
            <div>
              <p
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "10px",
                  color: "var(--muted)",
                  letterSpacing: "3px",
                }}
              >
                VERDICT
              </p>
              <p
                style={{
                  fontFamily: "var(--sans)",
                  fontWeight: 700,
                  fontSize: "48px",
                  lineHeight: 1,
                  color: isFake ? "var(--danger)" : "var(--accent)",
                  textShadow: isFake
                    ? "0 0 30px var(--danger)"
                    : "0 0 30px var(--accent)",
                }}
              >
                {isFake ? "FAKE" : "REAL"}
              </p>
            </div>
            <div style={{ textAlign: "right" }}>
              <p
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "10px",
                  color: "var(--muted)",
                  letterSpacing: "2px",
                }}
              >
                THRESHOLD
              </p>
              <p
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "20px",
                  color: "var(--text)",
                }}
              >
                {result.threshold}
              </p>
            </div>
          </div>

          <ConfidenceBar
            label="FAKE PROBABILITY"
            value={result.fake_probability}
            color="var(--danger)"
          />
          <ConfidenceBar
            label="REAL PROBABILITY"
            value={result.real_probability}
            color="var(--accent)"
          />

          <div
            style={{
              display: "flex",
              gap: "24px",
              marginTop: "24px",
              paddingTop: "24px",
              borderTop: "1px solid var(--border)",
            }}
          >
            {[
              ["TRIED", result.frames.tried],
              ["FACES", result.frames.faces_found],
              ["USED", result.frames.used],
            ].map(([label, val]) => (
              <div key={label}>
                <p
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: "10px",
                    color: "var(--muted)",
                    letterSpacing: "2px",
                  }}
                >
                  {label}
                </p>
                <p
                  style={{
                    fontFamily: "var(--sans)",
                    fontWeight: 700,
                    fontSize: "28px",
                  }}
                >
                  {val}
                </p>
              </div>
            ))}
          </div>

          {result.per_frame_preds && (
            <FrameTimeline preds={result.per_frame_preds} />
          )}

          <button
            onClick={() => {
              setFile(null);
              setResult(null);
            }}
            style={{
              marginTop: "24px",
              width: "100%",
              padding: "12px",
              background: "transparent",
              border: "1px solid var(--border)",
              color: "var(--muted)",
              fontFamily: "var(--mono)",
              fontSize: "11px",
              letterSpacing: "2px",
              cursor: "pointer",
              borderRadius: "2px",
            }}
          >
            ANALYZE ANOTHER
          </button>
        </div>
      )}
    </div>
  );
}
