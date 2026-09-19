/* eslint-disable no-unused-vars */
import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, Upload, Image as ImageIcon, AlertTriangle, CheckCircle,
  Zap, ChevronRight, Info, Eye, Cpu, FlaskConical, HeartPulse,
  BarChart3, ShieldAlert, X, Moon, Sun, Clock, Download,
  History, Atom, TrendingUp, Phone, Layers, Code, ScanEye, Globe, CircuitBoard
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import BatchUploadTab from './BatchUploadTab.jsx';

// ─── Constants ───────────────────────────────────────────────────────────────

const CLASS_LABELS = {
  Normal: 'Normal',
  Arrhythmia: 'Arrhythmia',
  Myocardial_Infarction: 'Myocardial Infarction',
  History_of_MI: 'History of MI',
};

const SEVERITY_STYLES = {
  normal:   { badge: 'severity-normal',   icon: CheckCircle,   color: '#15803d', bg: '#f0fdf4' },
  warning:  { badge: 'severity-warning',  icon: AlertTriangle, color: '#b45309', bg: '#fffbeb' },
  critical: { badge: 'severity-critical', icon: ShieldAlert,   color: '#be123c', bg: '#fff1f2' },
};

const CONFIDENCE_BAR_COLOR = { high: '#10b981', medium: '#f59e0b', low: '#ef4444' };

// Real results from our trained models (QSVC tuned: minmax_0pi, reps=2, circular, C=5.0)
const MODEL_PERF = [
  { name: 'Classical SVM',   accuracy: 84.95, precision: 84.17, recall: 84.41, f1: 84.10, color: '#3b82f6' },
  { name: 'QSVC',            accuracy: 94.62, precision: 95.00, recall: 94.62, f1: 94.50, color: '#8b5cf6' },
  { name: 'Pegasos QSVC',    accuracy: 91.94, precision: 91.80, recall: 91.94, f1: 91.50, color: '#a78bfa' },
];

const PIPELINE_STEPS = [
  ['OTSU Adaptive Preprocessing', '340×340 grayscale, grid removal'],
  ['ResNet50 Feature Extraction',  'pool1_pool → 462,400-D vector'],
  ['Truncated SVD Reduction',      '462K → 9 latent dimensions (SVD)'],
  ['MinMax Normalization',          '[0, π] scaling'],
  ['RBF / Quantum SVM',             'C=5, ZZFeatureMap kernel (QSVC)'],
];

// ─── Animated ECG SVG (empty state) ──────────────────────────────────────────
function EcgAnimation() {
  return (
    <svg viewBox="0 0 200 60" className="w-48 h-16 mb-4" fill="none">
      <path
        className="ecg-path"
        d="M0 30 L30 30 L38 30 L42 10 L46 50 L50 30 L58 30 L62 20 L66 40 L70 30 L80 30 L84 28 L88 8 L92 52 L96 30 L110 30 L114 26 L118 15 L122 45 L126 30 L140 30 L200 30"
        stroke="#3b82f6"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ─── MI Urgency Widget ────────────────────────────────────────────────────────
function MIUrgencyWidget() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="mi-alert rounded-2xl bg-rose-50 p-4 flex gap-3 items-start"
    >
      <div className="flex-shrink-0">
        <Phone className="w-6 h-6 text-rose-600 animate-pulse" />
      </div>
      <div>
        <p className="font-extrabold text-rose-700 text-sm uppercase tracking-wide mb-1">
          ⚡ Critical — Myocardial Infarction Detected
        </p>
        <p className="text-xs text-rose-600 leading-relaxed">
          Every minute without treatment causes irreversible cardiac tissue loss.
          Call <strong>112 / 911</strong> immediately. Do not drive yourself to hospital.
        </p>
        <div className="mt-2 flex items-center gap-1.5 text-[10px] text-rose-500 font-medium">
          <div className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping" />
          SEEK EMERGENCY CARE NOW
        </div>
      </div>
    </motion.div>
  );
}

// ─── Alert Banner ───────────────────────────────────────────────────────────────────────
function AlertBanner({ result }) {
  if (!result) return null;
  const pred = result.prediction;
  const conf = result.confidence;
  const isLow = conf < 0.75;

  let cfg;
  if (isLow) {
    cfg = { bg: 'bg-blue-50 border-blue-300', text: 'text-blue-800', icon: '🔵', pulse: false,
      title: 'LOW CONFIDENCE — Manual Review Required',
      body: `Model confidence is ${(conf*100).toFixed(1)}%. Do not use this prediction alone. Refer to a cardiologist.` };
  } else if (pred === 'Myocardial_Infarction') {
    cfg = { bg: 'bg-rose-100 border-rose-500', text: 'text-rose-900', icon: '🚨', pulse: true,
      title: 'CRITICAL — Possible Myocardial Infarction',
      body: 'Call emergency services (112 / 911) immediately. Do not leave patient unattended.' };
  } else if (pred === 'Arrhythmia') {
    cfg = { bg: 'bg-amber-50 border-amber-400', text: 'text-amber-900', icon: '⚠️', pulse: false,
      title: 'ABNORMAL RHYTHM — Arrhythmia Detected',
      body: 'Refer patient to a cardiologist promptly. Holter monitoring may be warranted.' };
  } else if (pred === 'History_of_MI') {
    cfg = { bg: 'bg-yellow-50 border-yellow-400', text: 'text-yellow-900', icon: '📈', pulse: false,
      title: 'CARDIAC HISTORY FLAG — History of MI',
      body: 'Recommend follow-up ECG and cardiology review. Monitor for new ischaemic changes.' };
  } else {
    cfg = { bg: 'bg-emerald-50 border-emerald-400', text: 'text-emerald-900', icon: '✅', pulse: false,
      title: 'NORMAL SINUS RHYTHM',
      body: 'No acute cardiac findings detected. Routine follow-up as clinically indicated.' };
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border-2 p-3 flex items-start gap-3 ${cfg.bg}`}>
      <span className={`text-xl flex-shrink-0 ${cfg.pulse ? 'animate-pulse' : ''}`}>{cfg.icon}</span>
      <div>
        <p className={`font-extrabold text-xs uppercase tracking-wide ${cfg.text}`}>{cfg.title}</p>
        <p className={`text-xs mt-0.5 ${cfg.text} opacity-80`}>{cfg.body}</p>
      </div>
    </motion.div>
  );
}

// ─── Comparison Panel ───────────────────────────────────────────────────────────────────
const MODEL_META = {
  classical: { label: 'Classical SVM',  acc: 84.95, color: '#3b82f6' },
  pegasos:   { label: 'Pegasos QSVC',   acc: 91.94, color: '#a78bfa' },
  quantum:   { label: 'QSVC ⭐',         acc: 94.62, color: '#8b5cf6' },
};

function ComparisonPanel({ file, currentModel }) {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const runComparison = async () => {
    if (!file || !(file instanceof Blob)) return;
    setLoading(true); setError(null); setResults(null);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/predict/all', { method: 'POST', body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Server error'); }
      const data = await res.json();
      setResults(data.all_results);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const allSame = results && new Set(results.map(r => r.data.prediction)).size === 1;

  return (
    <div className="glass-card p-4 mt-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
          <BarChart3 className="w-3.5 h-3.5" /> Multi-Model Consensus
        </p>
        {!results && !loading && (
          <button onClick={runComparison}
            className="text-xs px-3 py-1.5 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors font-semibold">
            Compare All Models
          </button>
        )}
        {results && (
          <button onClick={runComparison}
            className="text-xs px-2 py-1 text-slate-500 hover:text-slate-700 underline">Refresh</button>
        )}
      </div>

      {loading && (
        <div className="flex flex-col gap-2">
          {['Classical SVM (~1s)', 'Pegasos QSVC (~12s)', 'QSVC (~15s)'].map((m, i) => (
            <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-slate-50 animate-pulse">
              <div className="w-2 h-2 rounded-full bg-slate-300" />
              <span className="text-xs text-slate-400">{m} — running…</span>
            </div>
          ))}
          <p className="text-[10px] text-slate-400 mt-1">Total estimated time: ~28s for all 3 models</p>
        </div>
      )}

      {error && <p className="text-xs text-rose-500">{error}</p>}

      {results && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead><tr className="border-b border-slate-100">
                <th className="pb-2 text-slate-400 font-medium">Model</th>
                <th className="pb-2 text-slate-400 font-medium">Prediction</th>
                <th className="pb-2 text-slate-400 font-medium">Confidence</th>
                <th className="pb-2 text-slate-400 font-medium">Time</th>
              </tr></thead>
              <tbody>
                {results.map(({ model, data }) => {
                  const meta = MODEL_META[model] || {};
                  const isBest = model === 'quantum';
                  return (
                    <tr key={model} className={`border-b border-slate-50 ${isBest ? 'bg-violet-50/50' : ''}`}>
                      <td className="py-2 pr-2 font-semibold" style={{ color: meta.color }}>
                        {meta.label}
                      </td>
                      <td className="py-2 pr-2">
                        {CLASS_LABELS[data.prediction] || data.prediction}
                      </td>
                      <td className="py-2 pr-2">
                        <div className="flex items-center gap-1">
                          <div className="h-1.5 rounded-full bg-slate-100 w-16 overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${data.confidence*100}%`, background: meta.color }} />
                          </div>
                          <span className="font-bold" style={{ color: meta.color }}>{(data.confidence*100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="py-2 text-slate-400">{data.processing_time?.total_ms?.toFixed(0)}ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className={`mt-3 p-2.5 rounded-lg text-xs font-semibold flex items-center gap-2 ${
            allSame ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-800'}`}>
            {allSame
              ? '✅ All 3 models agree — High clinical consensus. Prediction is robust.'
              : '⚠️ Models disagree — Treat with caution. Refer to a cardiologist for review.'}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Stats Bar ─────────────────────────────────────────────────────────────────────
function StatsBar({ history }) {
  if (history.length === 0) return null;
  const avgConf = history.reduce((s, h) => s + (h.result?.confidence || 0), 0) / history.length;
  const avgTime = history.reduce((s, h) => s + (h.result?.processing_time?.total_ms || 0), 0) / history.length;
  const cacheHits = history.filter(h => h.result?.cache_hit).length;
  const flagged   = history.filter(h => (h.result?.confidence || 1) < 0.75).length;
  return (
    <div className="glass-card p-3 text-[10px] text-slate-500">
      <p className="font-semibold uppercase tracking-widest text-slate-400 mb-2 flex items-center gap-1">
        <TrendingUp className="w-3 h-3" /> Session Stats
      </p>
      <div className="grid grid-cols-2 gap-y-1.5 gap-x-2">
        <span>📊 Analyses: <strong className="text-slate-700">{history.length}</strong></span>
        <span>🎯 Avg Conf: <strong className="text-slate-700">{(avgConf*100).toFixed(1)}%</strong></span>
        <span>⏱ Avg Time: <strong className="text-slate-700">{avgTime < 2000 ? `${avgTime.toFixed(0)}ms` : `${(avgTime/1000).toFixed(1)}s`}</strong></span>
        <span>⚡ Cache Hits: <strong className="text-slate-700">{cacheHits}</strong></span>
        {flagged > 0 && <span className="col-span-2 text-amber-600">⚠️ Flagged: <strong>{flagged}</strong></span>}
      </div>
    </div>
  );
}

// ─── Drop Zone ────────────────────────────────────────────────────────────────
function DropZone({ onFileSelect, isDragging, setIsDragging }) {
  const ref = useRef(null);
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) onFileSelect(f);
  };
  return (
    <div
      className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center gap-3 py-10 px-6 cursor-pointer transition-all duration-200
        ${isDragging ? 'dropzone-active border-blue-400' : 'border-slate-200 hover:border-blue-300 hover:bg-blue-50/40'}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => ref.current?.click()}
    >
      <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${isDragging ? 'bg-blue-100' : 'bg-slate-100'}`}>
        <Upload className={`w-6 h-6 ${isDragging ? 'text-blue-500' : 'text-slate-400'}`} />
      </div>
      <div className="text-center">
        <p className="font-semibold text-slate-700 text-sm">Drop ECG image here</p>
        <p className="text-xs text-slate-400 mt-0.5">PNG, JPG · Max 10 MB</p>
      </div>
      <button className="px-4 py-1.5 text-xs font-semibold text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors">
        Browse Files
      </button>
      <input ref={ref} type="file" accept="image/*" className="hidden"
        onChange={e => e.target.files?.[0] && onFileSelect(e.target.files[0])} />
    </div>
  );
}

// ─── Preset Buttons ────────────────────────────────────────────────────────────
function PresetButtons({ onSelect, onOodSelect }) {
  const loadSample = async (filename, isOod = false) => {
    try {
      const res = await fetch(`/samples/${filename}`);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      if (isOod && onOodSelect) {
        onOodSelect(file);   // sets skipGatekeeper=true then loads file
      } else {
        onSelect(file);
      }
    } catch (e) {
      console.error('Failed to load sample', e);
    }
  };
  return (
    <div className="flex gap-2 flex-wrap mb-4">
      <button onClick={() => loadSample('normal.jpg')} className="text-xs px-2 py-1 bg-green-50 text-green-700 rounded hover:bg-green-100 font-medium">🫀 Normal</button>
      <button onClick={() => loadSample('arrhythmia.jpg')} className="text-xs px-2 py-1 bg-amber-50 text-amber-700 rounded hover:bg-amber-100 font-medium">⚡ Arrhythmia</button>
      <button onClick={() => loadSample('mi.jpg')} className="text-xs px-2 py-1 bg-rose-50 text-rose-700 rounded hover:bg-rose-100 font-medium">🔴 MI</button>
      <button onClick={() => loadSample('history_mi.jpg')} className="text-xs px-2 py-1 bg-yellow-50 text-yellow-700 rounded hover:bg-yellow-100 font-medium">📈 History MI</button>
      <button onClick={() => loadSample('ood_test.jpg', true)} className="text-xs px-2 py-1 bg-slate-100 text-slate-700 rounded hover:bg-slate-200 font-medium">🚫 OOD Test</button>
    </div>
  );
}

// ─── Patient Form ─────────────────────────────────────────────────────────────
function PatientForm({ patient, setPatient }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="mb-4">
      <button onClick={() => setIsOpen(!isOpen)} className="text-xs font-semibold text-blue-600 hover:text-blue-700 flex items-center gap-1 w-full justify-between bg-blue-50/50 p-2 rounded-lg transition-colors">
        <span className="flex items-center gap-1">👤 Patient Details (Optional)</span>
        <span>{isOpen ? '▼' : '▶'}</span>
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="p-3 bg-white border border-blue-100 rounded-lg mt-2 grid grid-cols-2 gap-3 text-xs">
              <input type="text" placeholder="Patient Name" value={patient.name} onChange={e => setPatient({...patient, name: e.target.value})} className="px-2 py-1.5 border border-slate-200 rounded" />
              <input type="text" placeholder="Patient ID" value={patient.id} onChange={e => setPatient({...patient, id: e.target.value})} className="px-2 py-1.5 border border-slate-200 rounded" />
              <input type="number" placeholder="Age" value={patient.age} onChange={e => setPatient({...patient, age: e.target.value})} className="px-2 py-1.5 border border-slate-200 rounded" />
              <select value={patient.sex} onChange={e => setPatient({...patient, sex: e.target.value})} className="px-2 py-1.5 border border-slate-200 rounded text-slate-600">
                <option value="">Sex</option><option value="M">M</option><option value="F">F</option><option value="Other">Other</option>
              </select>
              <input type="text" placeholder="Attending Doctor" value={patient.doctor} onChange={e => setPatient({...patient, doctor: e.target.value})} className="col-span-2 px-2 py-1.5 border border-slate-200 rounded" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Image Tab Viewer ─────────────────────────────────────────────────────────
function ImageTabs({ original, preprocessed, file, selectedModel }) {
  const [tab, setTab] = useState('original');
  const [gradcam, setGradcam]       = useState(null);
  const [gcLoading, setGcLoading]   = useState(false);
  const [gcError, setGcError]       = useState(null);

  const fetchGradCam = async () => {
    if (!file || !(file instanceof Blob)) {
      setGcError('Upload an image and run analysis first.');
      return;
    }
    setGcLoading(true); setGcError(null);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(`/predict/gradcam?model=${encodeURIComponent(selectedModel || 'classical')}`, {
        method: 'POST', body: fd,
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Server error'); }
      const data = await res.json();
      setGradcam(data.gradcam_overlay);
    } catch (e) {
      setGcError(e.message);
    } finally {
      setGcLoading(false);
    }
  };

  // Auto-fetch when tab switches to gradcam and we don't have it yet
  useEffect(() => {
    if (tab === 'gradcam' && !gradcam && !gcLoading) setTimeout(fetchGradCam, 0);
  }, [tab]); // eslint-disable-line react-hooks/exhaustive-deps

  const TABS = [
    { id: 'original',     label: 'Original',         Icon: ImageIcon },
    { id: 'preprocessed', label: 'Preprocessed',      Icon: Eye       },
    { id: 'gradcam',      label: 'Grad-CAM',          Icon: ScanEye   },
  ];

  return (
    <div className="rounded-xl overflow-hidden border border-slate-200">
      <div className="flex border-b border-slate-200 bg-slate-50">
        {TABS.map(({ id, label, Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium flex-1 justify-center transition-colors
              ${tab === id ? 'tab-active bg-white' : 'tab-inactive'}`}>
            <Icon className="w-3 h-3" /> {label}
          </button>
        ))}
      </div>

      <div className="bg-slate-900 flex items-center justify-center min-h-[200px] relative">
        <AnimatePresence mode="wait">
          {tab === 'original' && (
            <motion.img key="orig" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              src={original} alt="Original ECG" className="max-h-[220px] w-full object-contain" />
          )}
          {tab === 'preprocessed' && (
            preprocessed
              ? <motion.img key="pre" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  src={preprocessed} alt="Preprocessed ECG" className="max-h-[220px] w-full object-contain" />
              : <motion.div key="wait" className="text-slate-500 text-xs flex flex-col items-center gap-2 py-8">
                  <Activity className="w-7 h-7 opacity-40 animate-pulse" />
                  Run analysis to see preprocessed output
                </motion.div>
          )}
          {tab === 'gradcam' && (
            gcLoading
              ? <motion.div key="gcload" className="flex flex-col items-center gap-3 py-10 text-slate-400">
                  <div className="w-8 h-8 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs">Computing activation map…</span>
                </motion.div>
              : gcError
                ? <motion.div key="gcerr" className="flex flex-col items-center gap-2 py-8 text-slate-400 px-4 text-center">
                    <ScanEye className="w-7 h-7 opacity-40" />
                    <span className="text-xs text-rose-400">{gcError}</span>
                    <button onClick={fetchGradCam}
                      className="text-xs text-blue-400 hover:text-blue-300 underline">Retry</button>
                  </motion.div>
                : gradcam
                  ? <motion.div key="gcimg" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative w-full">
                      <img src={gradcam} alt="Grad-CAM" className="max-h-[220px] w-full object-contain" />
                      {/* Colour legend */}
                      <div className="absolute bottom-2 right-2 bg-slate-900/80 rounded-lg px-2 py-1 flex items-center gap-1.5">
                        <div className="w-16 h-2 rounded-full" style={{ background: 'linear-gradient(to right, #00f, #0ff, #0f0, #ff0, #f00)' }} />
                        <span className="text-[8px] text-slate-300 font-mono">low → high</span>
                      </div>
                    </motion.div>
                  : <motion.div key="gcwait" className="flex flex-col items-center gap-3 py-10 text-slate-400">
                      <ScanEye className="w-7 h-7 opacity-40 animate-pulse" />
                      <span className="text-xs">Switch here after analysis to load heatmap</span>
                    </motion.div>
          )}
        </AnimatePresence>
      </div>

      {tab === 'preprocessed' && preprocessed && (
        <div className="bg-blue-50 px-3 py-1.5 flex items-center gap-1.5 text-xs text-blue-700 border-t border-blue-100">
          <Info className="w-3 h-3 flex-shrink-0" />
          OTSU thresholding + grid removal applied — exact ResNet50 input
        </div>
      )}
      {tab === 'gradcam' && gradcam && (
        <div className="bg-violet-50 px-3 py-1.5 flex items-center gap-1.5 text-xs text-violet-700 border-t border-violet-100">
          <ScanEye className="w-3 h-3 flex-shrink-0" />
          ResNet50 conv5_block3_out activations — red = most attended, blue = least
        </div>
      )}
    </div>
  );
}

// ─── Processing Time Breakdown ────────────────────────────────────────────────
function TimingBreakdown({ timing }) {
  if (!timing) return null;
  const steps = [
    { label: 'Preprocessing',      ms: timing.preprocessing_ms,       color: '#3b82f6' },
    { label: 'ResNet50 Extraction', ms: timing.feature_extraction_ms,  color: '#8b5cf6' },
    { label: 'SVD Reduction',       ms: timing.svd_reduction_ms,        color: '#06b6d4' },
    { label: 'SVM Classification',  ms: timing.classification_ms,       color: '#10b981' },
  ];
  const max = Math.max(...steps.map(s => s.ms), 1);
  return (
    <div className="glass-card p-4 mt-4">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
        <Clock className="w-3.5 h-3.5" /> Pipeline Timing
      </p>
      <div className="flex flex-col gap-2">
        {steps.map(({ label, ms, color }) => (
          <div key={label}>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-slate-600">{label}</span>
              <span className="font-semibold text-slate-700">{ms.toFixed(1)} ms</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
              <motion.div initial={{ width: 0 }} animate={{ width: `${(ms / max) * 100}%` }}
                transition={{ duration: 0.8, delay: 0.1 }}
                className="h-full rounded-full" style={{ background: color }} />
            </div>
          </div>
        ))}
        <div className="pt-1.5 border-t border-slate-100 flex justify-between text-xs mt-1">
          <span className="text-slate-500 font-medium">Total</span>
          <span className="font-bold text-slate-800">{timing.total_ms.toFixed(1)} ms</span>
        </div>
      </div>
    </div>
  );
}

// ─── Probability Bars ─────────────────────────────────────────────────────────
function ProbabilityBar({ label, value, isTop }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className={isTop ? 'font-semibold text-slate-800' : 'text-slate-500'}>
          {CLASS_LABELS[label] || label}
        </span>
        <span className={isTop ? 'font-bold text-slate-800' : 'text-slate-400'}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.9, delay: 0.1 }}
          className="h-full rounded-full"
          style={{ background: isTop ? '#3b82f6' : '#cbd5e1' }}
        />
      </div>
    </div>
  );
}

// ─── Result Panel ─────────────────────────────────────────────────────────────
function ResultPanel({ result, file, selectedModel, resultRef, patient }) {
  const [isPdfLoading, setIsPdfLoading] = useState(false);
  const sev       = SEVERITY_STYLES[result.class_info?.severity || 'normal'];
  const SevIcon   = sev.icon;
  const isLowConf = result.confidence_level === 'low';
  const isMI      = result.prediction === 'Myocardial_Infarction';

  const handleDownloadPDF = async () => {
    try {
      if (file && file instanceof Blob) {
        setIsPdfLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        // Patient info optional fields
        if (patient?.name)   formData.append('patient_name', patient.name);
        if (patient?.id)     formData.append('patient_id',   patient.id);
        if (patient?.age)    formData.append('patient_age',  patient.age);
        if (patient?.sex)    formData.append('patient_sex',  patient.sex);
        if (patient?.doctor) formData.append('doctor_name',  patient.doctor);
        const res = await fetch(`/predict/pdf?model=${encodeURIComponent(selectedModel || 'classical')}`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) throw new Error('Backend PDF generation failed');
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `QuCardio_Report_${file.name}.pdf`;
        a.click();
        window.URL.revokeObjectURL(url);
      } else if (resultRef?.current) {
        // Fallback for history items where we only have the DOM
        setIsPdfLoading(true);
        const canvas = await html2canvas(resultRef.current, { scale: 2, useCORS: true, backgroundColor: '#ffffff' });
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
        const pdfW = pdf.internal.pageSize.getWidth();
        const pdfH = (canvas.height * pdfW) / canvas.width;
        pdf.setFontSize(16);
        pdf.setTextColor(15, 23, 42);
        pdf.text('QuCardio — ECG Diagnostic Report', 14, 14);
        pdf.setFontSize(8);
        pdf.setTextColor(100, 116, 139);
        pdf.text(`Generated: ${new Date().toLocaleString()}  |  For research purposes only. Not a medical device.`, 14, 20);
        pdf.addImage(imgData, 'PNG', 0, 26, pdfW, pdfH);
        pdf.save(`QuCardio_Report_${result.prediction}_${Date.now()}.pdf`);
      }
    } catch (err) {
      console.error('PDF generation failed:', err);
      alert('Failed to generate PDF. Check console for details.');
    } finally {
      setIsPdfLoading(false);
    }
  };

  return (
    <motion.div ref={resultRef} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-4">

      {/* MI Urgency widget */}
      {isMI && <MIUrgencyWidget />}

      {/* Diagnosis card */}
      <div className="glass-card p-5">
        <div className="flex items-start justify-between mb-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Diagnosis</p>
          <span className={`flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full ${sev.badge}`}>
            <SevIcon className="w-3 h-3" />
            {result.class_info?.severity?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>
        <h2 className="text-2xl font-bold mb-1" style={{ color: sev.color }}>
          {CLASS_LABELS[result.prediction] || result.prediction}
        </h2>
        <p className="text-sm text-slate-500 mb-3">{result.class_info?.description}</p>
        {result.class_info?.action && (
          <div className={`flex items-start gap-2 p-3 rounded-xl text-sm ${sev.badge}`}>
            <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span><strong>Recommended Action:</strong> {result.class_info.action}</span>
          </div>
        )}
        <button onClick={handleDownloadPDF} disabled={isPdfLoading}
          className="mt-3 w-full flex items-center justify-center gap-2 py-2 text-xs font-semibold text-slate-600 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-wait">
          <Download className="w-3.5 h-3.5" /> {isPdfLoading ? 'Generating...' : 'Download PDF Report'}
        </button>
      </div>

      {/* Grid wrapper for compactness */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Confidence */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Model Confidence</p>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full
              ${result.confidence_level === 'high' ? 'bg-emerald-50 text-emerald-700' :
                result.confidence_level === 'medium' ? 'bg-amber-50 text-amber-700' :
                'bg-rose-50 text-rose-700'}`}>
              {result.confidence_level?.toUpperCase()}
            </span>
          </div>
          <div className="flex items-end gap-1.5 mb-3">
            <span className="text-4xl font-extrabold text-slate-900">{(result.confidence * 100).toFixed(1)}</span>
            <span className="text-lg font-medium text-slate-400 mb-1">%</span>
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden mb-1">
            <motion.div initial={{ width: 0 }} animate={{ width: `${result.confidence * 100}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
              className="h-full rounded-full"
              style={{ background: CONFIDENCE_BAR_COLOR[result.confidence_level] }} />
          </div>
          {isLowConf && (
            <div className="mt-2 flex items-start gap-2 p-2.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              Low confidence — atypical ECG style or image quality issue. Manual clinical review strongly recommended.
            </div>
          )}
          <p className="text-[11px] text-slate-400 mt-2 flex items-center gap-1">
            <Cpu className="w-3 h-3" /> {result.model_used}
          </p>
        </div>

        {/* Class probabilities */}
        <div className="glass-card p-5">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5" /> Class Probabilities
          </p>
          <div className="flex flex-col gap-2.5">
            {Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([label, value]) => (
                <ProbabilityBar key={label} label={label} value={value} isTop={label === result.prediction} />
              ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Session History Sidebar ──────────────────────────────────────────────────
function HistorySidebar({ history, activeId, onSelect, onClear }) {
  const SEV_ICON = { normal: '🟢', warning: '🟡', critical: '🔴' };
  if (history.length === 0) return (
    <div className="text-center py-8 text-slate-400 text-xs">
      <History className="w-8 h-8 mx-auto mb-2 opacity-30" />
      No analyses yet
    </div>
  );
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
          {history.length} Analysis{history.length !== 1 ? 'es' : ''}
        </span>
        <button onClick={onClear} className="text-[10px] text-slate-400 hover:text-rose-500 transition-colors">Clear</button>
      </div>
      {history.map(item => (
        <div key={item.id} onClick={() => onSelect(item)}
          className={`history-item ${activeId === item.id ? 'active' : ''}`}>
          <div className="flex items-center justify-between mb-0.5">
            <span className="text-xs font-semibold text-slate-700 truncate max-w-[140px]">
              {SEV_ICON[item.result.class_info?.severity || 'normal']} {CLASS_LABELS[item.result.prediction]}
            </span>
            <span className="text-[10px] font-bold text-slate-500">{(item.result.confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="text-[10px] text-slate-400 truncate">{item.filename}</div>
          <div className="text-[10px] text-slate-300 mt-0.5">{item.timestamp}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Model Performance Tab ────────────────────────────────────────────────────
function ModelPerformanceTab() {
  const [metric, setMetric] = useState('accuracy');
  const metricOptions = ['accuracy', 'precision', 'recall', 'f1'];

  return (
    <div className="flex flex-col gap-6">
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <p className="font-semibold text-slate-700">Model Comparison — Your Trained Results</p>
          <div className="flex gap-1">
            {metricOptions.map(m => (
              <button key={m} onClick={() => setMetric(m)}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-colors capitalize
                  ${metric === m ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>
                {m === 'f1' ? 'F1' : m.charAt(0).toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={MODEL_PERF} barSize={36}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
            <YAxis domain={[70, 100]} tick={{ fontSize: 11, fill: '#64748b' }} unit="%" />
            <Tooltip formatter={(v) => [`${v.toFixed(2)}%`, metric.toUpperCase()]}
              contentStyle={{ borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 12 }} />
            <Bar dataKey={metric} radius={[6, 6, 0, 0]}>
              {MODEL_PERF.map((entry, i) => <Cell key={i} fill={entry.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Comparison table */}
      <div className="glass-card p-5 overflow-x-auto">
        <p className="font-semibold text-slate-700 mb-4">Detailed Results</p>
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="text-xs text-slate-400 uppercase tracking-wide border-b border-slate-100">
              <th className="pb-2 pr-4 font-semibold">Model</th>
              <th className="pb-2 pr-4 text-center font-semibold">Accuracy</th>
              <th className="pb-2 pr-4 text-center font-semibold">Precision</th>
              <th className="pb-2 pr-4 text-center font-semibold">Recall</th>
              <th className="pb-2 text-center font-semibold">F1</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_PERF.map((m, i) => (
              <tr key={i} className="border-b border-slate-50 last:border-0">
                <td className="py-2.5 pr-4">
                  <span className="flex items-center gap-2 font-medium text-slate-700">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: m.color }} />
                    {m.name}
                    {m.name === 'QSVC' && <span className="text-[10px] bg-violet-100 text-violet-700 px-1.5 py-0.5 rounded-full font-bold">Best</span>}
                  </span>
                </td>
                {['accuracy','precision','recall','f1'].map(k => (
                  <td key={k} className="py-2.5 pr-4 text-center font-semibold text-slate-700">
                    {m[k].toFixed(2)}%
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-[11px] text-slate-400 mt-3">
          QSVC (tuned: minmax×π, reps=2, circular) achieves <strong>94.62%</strong> — beats paper's 94.09% by +0.53pp.
          Pegasos achieves 91.94%. Paper (Prabhu et al., 2023): QSVC 94.09%, SVM 83.33%, Pegasos 93.05%.
        </p>
      </div>
    </div>
  );
}

// ─── Quantum Insights Tab ─────────────────────────────────────────────────────
function QuantumInsightsTab() {
  return (
    <div className="flex flex-col gap-5">
      <div className="glass-card p-5">
        <p className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <Atom className="w-4 h-4 text-violet-600" /> Quantum Kernel — How It Works
        </p>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-slate-600 leading-relaxed">
          <div className="bg-violet-50 rounded-xl p-3 border border-violet-100">
            <p className="font-semibold text-violet-700 mb-1 text-xs uppercase tracking-wide">1. Data Encoding</p>
            Classical 9-D ECG features are encoded into quantum states using
            <strong> ZZFeatureMap</strong> — a 9-qubit circuit with 2 repetitions
            and linear entanglement. Each feature becomes a rotation angle on a qubit.
          </div>
          <div className="bg-violet-50 rounded-xl p-3 border border-violet-100">
            <p className="font-semibold text-violet-700 mb-1 text-xs uppercase tracking-wide">2. Hilbert Space Embedding</p>
            The ZZFeatureMap maps 9-D data into a 2⁹ = <strong>512-dimensional</strong> complex
            Hilbert space — far beyond classical RBF kernel separability.
          </div>
          <div className="bg-violet-50 rounded-xl p-3 border border-violet-100">
            <p className="font-semibold text-violet-700 mb-1 text-xs uppercase tracking-wide">3. Quantum Kernel</p>
            <code className="text-xs font-mono text-violet-800">K(x,z) = |⟨ψ(x)|ψ(z)⟩|²</code>
            <br />Fidelity between quantum states measures similarity. Computed via statevector
            inner products — exact, reproducible, zero noise.
          </div>
          <div className="bg-violet-50 rounded-xl p-3 border border-violet-100">
            <p className="font-semibold text-violet-700 mb-1 text-xs uppercase tracking-wide">4. Classification</p>
            The quantum kernel matrix is passed to a standard SVM optimizer
            (or Pegasos SGD) which finds the optimal hyperplane in the
            512-D quantum feature space.
          </div>
        </div>
      </div>

      <div className="glass-card p-5">
        <p className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-600" /> Implementation Details
        </p>
        <div className="flex flex-col gap-2 text-sm">
          {[
            ['Feature Maps',      'QSVC: circular (reps=2) | Pegasos: linear (reps=2)'],
            ['Kernel Method',     'Statevector fidelity: |⟨ψᵢ|ψⱼ⟩|² via matrix multiplication'],
            ['Backend',           'Qiskit Statevector simulator (exact, no shot noise)'],
            ['Hilbert Space Dim', '2⁹ = 512 complex dimensions'],
            ['Kernel Matrix',     '742×742 (train) | 186×742 (test) | computed in ~0.1s'],
            ['QSVC Solver',       'sklearn SVC, precomputed, C=5.0, feats×π, reps=2, circular  →  94.62%'],
            ['Pegasos Solver',    'Custom SGD — per-model tuned C & τ  →  91.94% (3-pass grid search)'],
            ['Circuit Count',     'N=742 circuits (statevector), NOT N²=550,564 (naive approach)'],
          ].map(([k, v]) => (
            <div key={k} className="flex gap-3 py-2 border-b border-slate-50 last:border-0">
              <span className="text-xs font-semibold text-slate-500 w-36 flex-shrink-0">{k}</span>
              <span className="text-xs text-slate-700 font-mono">{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card p-5">
        <p className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-500" /> Quantum Advantage on This Dataset
        </p>
        <div className="flex items-center gap-4">
          <div className="text-center flex-1 p-3 bg-blue-50 rounded-xl border border-blue-100">
            <p className="text-2xl font-extrabold text-blue-600">84.95%</p>
            <p className="text-xs text-blue-500 mt-0.5">Classical SVM</p>
          </div>
          <div className="text-slate-400 font-bold text-lg">→</div>
          <div className="text-center flex-1 p-3 bg-violet-50 rounded-xl border border-violet-200">
            <p className="text-2xl font-extrabold text-violet-600">94.62%</p>
            <p className="text-xs text-violet-500 mt-0.5">QSVC (tuned) ⭐</p>
          </div>
          <div className="text-center flex-1 p-3 bg-emerald-50 rounded-xl border border-emerald-100">
            <p className="text-2xl font-extrabold text-emerald-600">+9.67%</p>
            <p className="text-xs text-emerald-500 mt-0.5">Best Quantum Gain</p>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
          <div className="bg-slate-50 rounded-lg p-2 border border-slate-100">
            <p className="font-bold text-slate-700">84.95%</p>
            <p className="text-slate-400">Classical SVM</p>
          </div>
          <div className="bg-violet-50 rounded-lg p-2 border border-violet-100">
            <p className="font-bold text-violet-700">94.62%</p>
            <p className="text-violet-500">QSVC ⭐ Best</p>
          </div>
          <div className="bg-purple-50 rounded-lg p-2 border border-purple-200">
            <p className="font-bold text-purple-700">91.94%</p>
            <p className="text-purple-500">Pegasos (tuned)</p>
          </div>
        </div>
        <p className="text-xs text-slate-400 mt-3 leading-relaxed">
          The quantum kernel's 512-D Hilbert space captures non-linear ECG morphology patterns
          beyond classical RBF kernels. Key insight: scaling features to <code className="text-violet-600">[0, π]</code> before
          ZZFeatureMap encoding maps values to full rotation angles — boosting QSVC from 89.78% to
          <strong className="text-violet-700"> 94.62%</strong>, beating the paper's 94.09%.
        </p>
      </div>

      <div className="glass-card p-5">
        <p className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <CircuitBoard className="w-4 h-4 text-slate-500" /> 9-Qubit ZZFeatureMap Circuit
        </p>
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden p-4 flex justify-center">
          <img src="/zzfeaturemap_circuit.png" alt="9-Qubit ZZFeatureMap Circuit Diagram" className="w-full max-w-2xl object-contain mix-blend-multiply" />
        </div>
        <p className="text-[10px] text-slate-500 mt-2">
          Generated via Qiskit. This circuit embeds the 9-dimensional classical ECG data into a 512-dimensional quantum state space.
        </p>
      </div>

      <div className="glass-card p-5">
        <p className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <Globe className="w-4 h-4 text-sky-500" /> Feature Angle Mapping — Bloch Sphere Visualization
        </p>
        <p className="text-xs text-slate-500 mb-4">
          Visualizing the <code className="text-sky-600 font-mono">[0, π]</code> rotation applied to each qubit during the first encoding layer for a patient vector.
        </p>
        <div className="flex justify-between items-center bg-slate-50 border border-slate-100 p-4 rounded-xl overflow-hidden">
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} className="flex flex-col items-center gap-2 relative">
              <div className="w-[50px] h-[50px] rounded-full border border-sky-200/50 flex items-center justify-center relative overflow-hidden bg-white shadow-sm">
                <div className="absolute w-[46px] h-[46px] border border-slate-200 rounded-full" />
                <div className="absolute w-[46px] h-[46px] border border-slate-100 rounded-full" style={{ transform: 'rotateX(75deg)' }} />
                <div className="absolute w-0.5 h-full bg-slate-100" />
                <div className="absolute w-full h-0.5 bg-slate-100" />
                {/* Randomly angled vector arrow for demo */}
                <motion.div
                  initial={{ rotate: 0 }}
                  animate={{ rotate: (i * 35 + 20) % 180 }}
                  transition={{ duration: 2, ease: "easeOut" }}
                  className="absolute w-0.5 h-[22px] bg-rose-500 origin-bottom rounded-full z-10 bottom-1/2"
                />
              </div>
              <span className="text-[9px] font-mono font-bold text-slate-400">q{i}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Confidence Arc SVG ───────────────────────────────────────────────────────
function ConfidenceArc({ value }) {
  const r = 14, circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(Math.max(value, 0), 1));
  const color = value >= 0.85 ? '#10b981' : value >= 0.6 ? '#f59e0b' : '#ef4444';
  return (
    <svg width="38" height="38" viewBox="0 0 38 38">
      <circle cx="19" cy="19" r={r} fill="none" stroke="#e2e8f0" strokeWidth="3" />
      <motion.circle cx="19" cy="19" r={r} fill="none"
        stroke={color} strokeWidth="3"
        strokeDasharray={circ} strokeLinecap="round"
        transform="rotate(-90 19 19)"
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1.2, delay: 0.6, ease: 'easeOut' }}
      />
      <text x="19" y="23" textAnchor="middle" fontSize="7.5" fontWeight="bold" fill={color}>
        {Math.round(value * 100)}%
      </text>
    </svg>
  );
}

// ─── Animated connector between pipeline nodes ────────────────────────────────
function PipelineConnector({ delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, scaleX: 0 }}
      animate={{ opacity: 1, scaleX: 1 }}
      transition={{ delay, duration: 0.3, ease: 'easeOut' }}
      style={{ transformOrigin: 'left' }}
      className="flex-shrink-0"
    >
      <ChevronRight className="w-5 h-5 text-blue-300" />
    </motion.div>
  );
}

// ─── Loading: animated step-by-step pipeline ─────────────────────────────────
const LOADING_STAGES = [
  { label: 'OTSU Preprocessing',   sub: 'Threshold · grid removal · 340×340', color: '#3b82f6' },
  { label: 'ResNet50 Extraction',   sub: 'pool1_pool → 462,400-D features',    color: '#8b5cf6' },
  { label: 'SVD Reduction',         sub: '462K → 9 latent dimensions',          color: '#06b6d4' },
  { label: 'Classification',        sub: 'SVM / ZZFeatureMap quantum kernel',   color: '#a855f7' },
];

function LivePipelineLoader({ model }) {
  const [activeStep, setActiveStep] = useState(0);
  const [elapsed, setElapsed]       = useState(0);

  // Step timing: classical ~1s total, quantum ~15s (most time in step 3)
  const delays = model === 'quantum'
    ? [0, 500, 1400, 2500]
    : [0, 300,  900, 1150];

  useEffect(() => {
    setTimeout(() => { setActiveStep(0); setElapsed(0); }, 0);
    const timers = delays.map((d, i) => setTimeout(() => setActiveStep(i), d));
    const tick   = setInterval(() => setElapsed(p => p + 100), 100);
    return () => { timers.forEach(clearTimeout); clearInterval(tick); };
  }, [model]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="glass-card p-8 min-h-[420px] flex flex-col items-center justify-center gap-6">
      {/* Pulsing heart icon */}
      <div className="relative w-14 h-14">
        <div className="absolute inset-0 rounded-full border-2 border-blue-200 animate-ping opacity-40" />
        <div className="absolute inset-0 rounded-full border-2 border-blue-400 animate-pulse" />
        <HeartPulse className="absolute inset-0 m-auto w-7 h-7 text-blue-500" />
      </div>

      <div className="text-center">
        <h3 className="font-semibold text-slate-800 text-sm">Running Pipeline</h3>
        <p className="text-[10px] text-slate-400 mt-0.5 font-mono">{(elapsed / 1000).toFixed(1)}s elapsed</p>
      </div>

      {/* Pipeline nodes */}
      <div className="flex items-start gap-1">
        {LOADING_STAGES.map((stage, i) => {
          const done    = i < activeStep;
          const active  = i === activeStep;
          const pending = i > activeStep;
          return (
            <div key={i} className="flex items-center">
              <div className="flex flex-col items-center w-[72px]">
                <motion.div
                  animate={active ? { scale: [1, 1.1, 1], borderColor: [stage.color, stage.color] } : {}}
                  transition={{ repeat: Infinity, duration: 1 }}
                  className="w-11 h-11 rounded-xl flex items-center justify-center mb-1.5 border-2 transition-all"
                  style={{
                    background: done ? '#f0fdf4' : active ? `${stage.color}18` : '#f8fafc',
                    borderColor: done ? '#86efac' : active ? stage.color : '#e2e8f0',
                    boxShadow: active ? `0 0 12px ${stage.color}40` : 'none',
                  }}
                >
                  {done
                    ? <CheckCircle className="w-5 h-5 text-emerald-500" />
                    : active
                      ? <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                          className="w-4 h-4 border-2 border-t-transparent rounded-full" style={{ borderColor: stage.color, borderTopColor: 'transparent' }} />
                      : <div className="w-3 h-3 rounded-full bg-slate-200" />
                  }
                </motion.div>
                <p className={`text-[9px] font-semibold text-center leading-tight ${done ? 'text-emerald-600' : active ? 'text-slate-700' : 'text-slate-400'}`}>
                  {stage.label}
                </p>
                <AnimatePresence>
                  {active && (
                    <motion.p key="sub" initial={{ opacity: 0, y: 2 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      className="text-[8px] text-center leading-tight mt-0.5" style={{ color: stage.color }}>
                      {stage.sub}
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>
              {i < LOADING_STAGES.length - 1 && (
                <motion.div
                  animate={{ opacity: i < activeStep ? 1 : 0.2 }}
                  transition={{ duration: 0.4 }}
                  className="w-5 h-0.5 rounded-full mb-8 flex-shrink-0"
                  style={{ background: i < activeStep ? stage.color : '#e2e8f0' }}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Current step description */}
      <AnimatePresence mode="wait">
        <motion.div key={activeStep}
          initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
          className="text-center">
          <p className="text-xs text-slate-600 font-medium">{LOADING_STAGES[activeStep]?.label}</p>
          <p className="text-[10px] text-slate-400">{LOADING_STAGES[activeStep]?.sub}</p>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

// ─── Pipeline Dataflow Visualization ─────────────────────────────────────────
function PipelineFlow({ result, originalImage }) {
  if (!result || !result.processing_time) return null;
  const timing = result.processing_time;
  const sev    = result.class_info?.severity || 'normal';
  const severityBorder = sev === 'critical' ? 'border-rose-300 bg-rose-50'
                        : sev === 'warning'  ? 'border-amber-300 bg-amber-50'
                        :                      'border-emerald-300 bg-emerald-50';
  const severityText   = sev === 'critical' ? 'text-rose-700'
                        : sev === 'warning'  ? 'text-amber-700'
                        :                      'text-emerald-700';

  return (
    <div className="glass-card p-5 mt-4 overflow-x-auto">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
        <Activity className="w-3.5 h-3.5" /> Pipeline Dataflow
      </p>

      <div className="flex items-center gap-3 min-w-max pb-2">

        {/* Node 1 — Raw Upload */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
          className="flex flex-col items-center w-28">
          <div className="w-22 h-22 w-[88px] h-[88px] rounded-xl border border-slate-200 bg-slate-50 overflow-hidden mb-2 flex items-center justify-center p-1">
            <img src={originalImage} alt="Raw" className="max-w-full max-h-full object-contain mix-blend-multiply" />
          </div>
          <p className="text-[11px] font-bold text-slate-700">Raw Upload</p>
          <p className="text-[9px] text-slate-400">Original ECG</p>
        </motion.div>

        <PipelineConnector delay={0.1} />

        {/* Node 2 — Preprocessed */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          className="flex flex-col items-center w-28">
          <div className="w-[88px] h-[88px] rounded-xl border border-blue-200 bg-slate-50 overflow-hidden mb-2 flex items-center justify-center p-1">
            {result.preprocessed_image
              ? <img src={result.preprocessed_image} alt="Preprocessed" className="max-w-full max-h-full object-contain mix-blend-multiply" />
              : <Activity className="w-6 h-6 text-slate-300" />}
          </div>
          <p className="text-[11px] font-bold text-slate-700">Preprocessed</p>
          <p className="text-[9px] text-blue-500">{timing.preprocessing_ms.toFixed(1)} ms</p>
        </motion.div>

        <PipelineConnector delay={0.25} />

        {/* Node 3 — ResNet50 feature grid */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}
          className="flex flex-col items-center w-28">
          <div className="w-[88px] h-[88px] rounded-xl bg-slate-900 border border-slate-700 mb-2 flex items-center justify-center relative overflow-hidden">
            {/* Animated feature-map grid */}
            <div className="grid grid-cols-5 gap-0.5 w-14 h-14 opacity-70">
              {[...Array(25)].map((_, i) => (
                <motion.div key={i}
                  animate={{ opacity: [0.15, 0.9, 0.15] }}
                  transition={{ duration: 1.5 + (i % 5) * 0.3, repeat: Infinity, delay: i * 0.06 }}
                  className="rounded-sm"
                  style={{ background: ['#60a5fa','#a78bfa','#34d399','#f472b6','#fb923c'][i % 5] }}
                />
              ))}
            </div>
            <div className="absolute inset-0 flex items-end justify-center pb-1.5">
              <span className="bg-slate-900/90 px-1.5 py-0.5 rounded text-[8px] font-mono text-blue-300 border border-blue-500/30">462,400-D</span>
            </div>
          </div>
          <p className="text-[11px] font-bold text-slate-700">ResNet50</p>
          <p className="text-[9px] text-purple-500">{timing.feature_extraction_ms.toFixed(1)} ms</p>
        </motion.div>

        <PipelineConnector delay={0.4} />

        {/* Node 4 — SVD 9-D bar chart */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
          className="flex flex-col items-center w-28">
          <div className="w-[88px] h-[88px] rounded-xl bg-slate-900 border border-slate-700 mb-2 flex items-end justify-center px-2 pb-2 gap-[3px] overflow-hidden">
            {result.svd_features
              ? result.svd_features.map((val, i) => (
                  <motion.div key={i}
                    initial={{ height: 0 }}
                    animate={{ height: `${Math.max(val, 0.05) * 100}%` }}
                    transition={{ delay: 0.55 + i * 0.05, duration: 0.5, ease: 'easeOut' }}
                    className="flex-1 rounded-t-sm"
                    style={{ background: `hsl(${180 + i * 20}, 80%, 60%)` }}
                  />
                ))
              : <span className="text-[9px] text-slate-500 self-center">No data</span>
            }
          </div>
          <p className="text-[11px] font-bold text-slate-700">SVD 9-D</p>
          <p className="text-[9px] text-cyan-500">{timing.svd_reduction_ms.toFixed(1)} ms</p>
        </motion.div>

        <PipelineConnector delay={0.6} />

        {/* Node 5 — Quantum / SVM kernel */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}
          className="flex flex-col items-center w-28">
          <div className="w-[88px] h-[88px] rounded-xl bg-slate-900 border border-violet-500/50 mb-2 flex items-center justify-center relative overflow-hidden">
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
              className="absolute w-14 h-14 border border-violet-400/40 rounded-full" />
            <motion.div animate={{ rotate: -360 }} transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              className="absolute w-10 h-10 border border-fuchsia-400/40 rounded-full" />
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 6, repeat: Infinity, ease: 'linear' }}
              className="absolute w-[52px] h-[52px] border border-cyan-400/30 rounded-full" style={{ transform: 'rotateX(60deg)' }} />
            <Atom className="w-5 h-5 text-violet-400 z-10" />
            <span className="absolute bottom-1.5 right-1.5 bg-violet-900/80 px-1 py-0.5 rounded text-[7px] font-mono text-violet-300">512-D ψ</span>
          </div>
          <p className="text-[11px] font-bold text-slate-700 truncate max-w-[110px]">{result.model_used}</p>
          <p className="text-[9px] text-emerald-500">{timing.classification_ms.toFixed(1)} ms</p>
        </motion.div>

        <PipelineConnector delay={0.8} />

        {/* Node 6 — Prediction result */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.9, type: 'spring', stiffness: 200, damping: 15 }}
          className="flex flex-col items-center w-28">
          <div className={`w-[88px] h-[88px] rounded-xl border-2 mb-2 flex flex-col items-center justify-center gap-1 ${severityBorder}`}>
            <p className={`text-[9px] font-bold text-center px-1 leading-tight ${severityText}`}>
              {CLASS_LABELS[result.prediction] || result.prediction}
            </p>
            <ConfidenceArc value={result.confidence} />
          </div>
          <p className="text-[11px] font-bold text-slate-700">Prediction</p>
          <p className={`text-[9px] font-semibold ${severityText}`}>
            {(result.confidence * 100).toFixed(1)}% conf.
          </p>
        </motion.div>

      </div>

      {/* Total pipeline time strip */}
      <div className="mt-3 pt-3 border-t border-slate-100 flex items-center gap-4 text-[10px] text-slate-400 flex-wrap">
        {[
          { label: 'Preprocess', ms: timing.preprocessing_ms,      color: '#3b82f6' },
          { label: 'ResNet50',   ms: timing.feature_extraction_ms,  color: '#8b5cf6' },
          { label: 'SVD',        ms: timing.svd_reduction_ms,        color: '#06b6d4' },
          { label: 'Classify',   ms: timing.classification_ms,       color: '#10b981' },
        ].map(({ label, ms, color }) => (
          <span key={label} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
            {label}: <strong className="text-slate-600">{ms.toFixed(1)} ms</strong>
          </span>
        ))}
        <span className="ml-auto font-bold text-slate-600">Total: {timing.total_ms.toFixed(1)} ms</span>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [imageWarning, setImageWarning] = useState(null);
  const [selectedModel, setSelectedModel] = useState('classical');
  const [activeTab, setActiveTab] = useState('diagnosis');
  const [isDark, setIsDark]       = useState(false);
  const [history, setHistory]     = useState([]);
  const [activeHistId, setActiveHistId] = useState(null);
  const [patient, setPatient]     = useState({ name: '', id: '', age: '', sex: '', doctor: '' });
  const [skipGatekeeper, setSkipGatekeeper] = useState(false);
  const resultRef = useRef(null);

  // Dark mode on <html>
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
  }, [isDark]);


  // Image quality check using canvas
  const checkImageQuality = useCallback((f, dataUrl) => {
    const img = new Image();
    img.onload = () => {
      if (img.width < 100 || img.height < 100) {
        setImageWarning('⚠️ Image resolution is very low (<100×100). Results may be unreliable.');
        return;
      }
      // Sample a 50×50 grid spread across the full image to get a representative average.
      // ECG images are mostly white paper with thin black signal lines — a 10×10 sample
      // frequently lands on pure background and incorrectly triggers the blank warning.
      const SIZE = 50;
      const canvas = document.createElement('canvas');
      canvas.width = SIZE; canvas.height = SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, SIZE, SIZE);
      const pixels = ctx.getImageData(0, 0, SIZE, SIZE).data;
      let sum = 0;
      for (let i = 0; i < pixels.length; i += 4) sum += pixels[i];
      const avg = sum / (pixels.length / 4);
      // Only warn if virtually every pixel is pure white (avg > 253) — true blank/solid-white image.
      // Normal ECG scans score 200-240 because of the signal lines in the sample.
      if (avg > 253) setImageWarning('⚠️ Image appears completely blank. Please upload a valid ECG scan.');
      else if (avg < 5) setImageWarning('⚠️ Image appears completely black. Preprocessing may not work correctly.');
      else setImageWarning(null);
    };
    img.src = dataUrl;
  }, []);

  const handleFileSelect = (f) => {
    setFile(f); setResult(null); setError(null); setImageWarning(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      checkImageQuality(f, e.target.result);
    };
    reader.readAsDataURL(f);
  };

  const handleClear = () => {
    setFile(null); setPreview(null); setResult(null);
    setError(null); setImageWarning(null); setActiveHistId(null);
    setSkipGatekeeper(false);
  };

  const handleAnalyze = async () => {
    if (!file || !file.size) return;
    setIsLoading(true); setError(null); setResult(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const gkParam = skipGatekeeper ? '&skip_gatekeeper=true' : '';
      const res = await fetch(`/predict?model=${encodeURIComponent(selectedModel)}${gkParam}`, { method: 'POST', body: formData });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Server error'); }
      const data = await res.json();
      setResult(data);
      const entry = {
        id: Date.now(),
        filename: file.name.length > 22 ? file.name.slice(0, 19) + '…' : file.name,
        timestamp: new Date().toLocaleTimeString(),
        preview,
        result: data,
      };
      setHistory(h => [entry, ...h.slice(0, 19)]);
      setActiveHistId(entry.id);
      setActiveTab('diagnosis');
    } catch (e) {
      setError(e.message || 'Failed to connect to the API. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  const handleHistorySelect = (item) => {
    setActiveHistId(item.id);
    setResult(item.result);
    setPreview(item.preview);
    setFile({ name: item.filename });
    setActiveTab('diagnosis');
  };

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space' && file && !isLoading) { e.preventDefault(); handleAnalyze(); }
      if (e.code === 'Escape') handleClear();
      if (e.key === 'r' || e.key === 'R') handleClear();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, isLoading]);

  const TABS = [
    { id: 'diagnosis',   label: 'Diagnosis',         Icon: HeartPulse },
    { id: 'batch',       label: 'Batch Upload',       Icon: Layers     },
    { id: 'performance', label: 'Model Performance',  Icon: BarChart3  },
    { id: 'quantum',     label: 'Quantum Insights',   Icon: Atom       },
    { id: 'api',         label: 'API Docs',           Icon: Code       },
  ];

  return (
    <div className={`min-h-screen ${isDark ? 'dark' : ''} bg-slate-50`}>

      {/* ── Navbar ────────────────────────────────────────────── */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-blue-800 flex items-center justify-center shadow">
              <HeartPulse className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-slate-900 tracking-tight">QuCardio</span>
            <span className="text-[10px] bg-blue-100 text-blue-700 font-semibold px-1.5 py-0.5 rounded-full">v2.0</span>
          </div>
          <div className="hidden md:flex items-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1"><Activity className="w-3.5 h-3.5 text-emerald-500" /> ResNet50 → SVD(9D) → Quantum SVM</span>
            <span className="flex items-center gap-1"><Zap className="w-3.5 h-3.5 text-violet-500" /> QSVC 94.62% · Pegasos 91.94% · SVM 84.95%</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden sm:flex items-center gap-1 text-xs text-slate-400">
              <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono">Space</kbd> Analyze
              <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono ml-1">Esc</kbd> Clear
            </span>
            <button onClick={() => setIsDark(d => !d)}
              className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors">
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 py-6 flex gap-5">

        {/* ── Left Sidebar ──────────────────────────────────── */}
        <aside className="hidden lg:flex flex-col gap-4 w-64 shrink-0">

          {/* Model selector */}
          <div className="glass-card p-4 sidebar">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400 mb-2">Select Model</p>
            <div className="flex flex-col gap-1.5">
              {[
                { key: 'classical', label: 'Classical SVM',  sub: '84.95% · RBF C=10',            Icon: Cpu,          color: 'blue'   },
                { key: 'quantum',   label: 'QSVC',           sub: '94.62% · reps=2 circular · feats×π', Icon: FlaskConical, color: 'violet' },
                { key: 'pegasos',   label: 'Pegasos QSVC',   sub: '91.94% · per-model tuned · 6 models', Icon: Atom,  color: 'purple' },
              ].map(({ key, label, sub, Icon, color }) => (
                <button key={key} onClick={() => setSelectedModel(key)}
                  className={`flex items-center gap-2 px-3 py-2.5 rounded-xl border text-left transition-all text-xs
                    ${selectedModel === key
                      ? color === 'blue'   ? 'bg-blue-50 border-blue-300 text-blue-800'
                      : color === 'violet' ? 'bg-violet-50 border-violet-300 text-violet-800'
                      :                      'bg-purple-50 border-purple-300 text-purple-800'
                      : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'}`}>
                  <Icon className={`w-3.5 h-3.5 flex-shrink-0 ${
                    selectedModel === key
                      ? color === 'blue' ? 'text-blue-600' : color === 'violet' ? 'text-violet-600' : 'text-purple-600'
                      : 'text-slate-400'}`} />
                  <div>
                    <div className="font-semibold leading-tight">{label}
                      {selectedModel === key && <span className="ml-1.5 text-[9px] font-bold opacity-60">ACTIVE</span>}
                    </div>
                    <div className="text-[10px] opacity-60 mt-0.5">{sub}</div>
                  </div>
                </button>
              ))}
            </div>
            <p className="mt-2 text-[10px] text-slate-400 leading-relaxed">
              {selectedModel === 'classical' && '⚡ Fastest inference. Calibrated probabilities via isotonic regression.'}
              {selectedModel === 'quantum'   && '⚛️ Quantum kernel SVM. Computes ZZFeatureMap statevector vs 742 training samples (~15s on CPU).'}
              {selectedModel === 'pegasos'   && '🔬 Pegasos SGD · Algorithm 1 decision tree · 3 binary models per prediction (~3–4s).'}
            </p>
          </div>

          {/* Session history */}
          <div className="glass-card p-4 flex-1 sidebar">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-1.5">
              <History className="w-3.5 h-3.5" /> Session History
            </p>
            <HistorySidebar
              history={history}
              activeId={activeHistId}
              onSelect={handleHistorySelect}
              onClear={() => { setHistory([]); setActiveHistId(null); }}
            />
          </div>

          {/* Pipeline reference */}
          <div className="glass-card p-4 sidebar">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400 mb-2.5">Pipeline</p>
            <ol className="flex flex-col gap-2 text-[11px] text-slate-600">
              {PIPELINE_STEPS.map(([step, detail], i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <span className="flex-shrink-0 w-4 h-4 rounded-full bg-blue-100 text-blue-600 font-bold flex items-center justify-center text-[9px]">{i+1}</span>
                  <span><strong>{step}</strong><br /><span className="text-slate-400">{detail}</span></span>
                </li>
              ))}
            </ol>
          </div>

          {/* Session stats */}
          <StatsBar history={history} />

        </aside>

        {/* ── Main Content ────────────────────────────────────── */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">

          {/* Tab bar */}
          <div className="tab-bar glass-card overflow-hidden">
            <div className="flex border-b border-slate-200">
              {TABS.map(({ id, label, Icon }) => (
                <button key={id} onClick={() => setActiveTab(id)}
                  className={`flex items-center gap-1.5 px-5 py-3 text-sm font-medium flex-1 justify-center transition-colors
                    ${activeTab === id ? 'tab-active bg-white' : 'tab-inactive'}`}>
                  <Icon className="w-3.5 h-3.5" /> {label}
                </button>
              ))}
            </div>
          </div>

          {/* Tab content */}
          <AnimatePresence mode="wait">

            {/* ── Diagnosis Tab ─────────────────── */}
            {activeTab === 'diagnosis' && (
              <motion.div key="diag" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="grid lg:grid-cols-5 gap-4 items-start">

                {/* Upload column */}
                <div className="lg:col-span-2 flex flex-col gap-4">
                  <div className="glass-card p-4">
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">Upload ECG</p>
                      {file && <button onClick={handleClear} className="text-slate-400 hover:text-slate-600"><X className="w-4 h-4" /></button>}
                    </div>

                    {/* 1-Click demo presets */}
                    {skipGatekeeper && (
                      <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
                        <span>⚠️ OOD / Research mode — gatekeeper bypassed</span>
                        <button onClick={() => setSkipGatekeeper(false)} className="ml-auto text-amber-500 hover:text-amber-700">✕</button>
                      </div>
                    )}
                    <PresetButtons
                      onSelect={(f) => { setSkipGatekeeper(false); handleFileSelect(f); }}
                      onOodSelect={(f) => { setSkipGatekeeper(true); handleFileSelect(f); }}
                    />

                    {/* Patient details (collapsible) */}
                    <PatientForm patient={patient} setPatient={setPatient} />

                    {!file
                      ? <DropZone onFileSelect={(f) => { setSkipGatekeeper(false); handleFileSelect(f); }} isDragging={isDragging} setIsDragging={setIsDragging} />
                      : <ImageTabs original={preview} preprocessed={result?.preprocessed_image || null}
                                    file={file} selectedModel={selectedModel} />
                    }

                    {file && (
                      <div className="mt-2 px-3 py-1.5 bg-slate-50 rounded-lg flex items-center justify-between text-xs text-slate-500 border border-slate-100">
                        <span className="truncate max-w-[160px] font-medium">{file.name}</span>
                        {file.size && <span>{(file.size / 1024).toFixed(0)} KB</span>}
                      </div>
                    )}

                    {imageWarning && (
                      <div className="mt-2 flex items-start gap-2 p-2.5 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-700">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" /> {imageWarning}
                      </div>
                    )}

                    {error && (
                      <div className="mt-2 flex items-start gap-2 p-2.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                        <span>
                          {error.startsWith('Invalid ECG image:')
                            ? <>
                                <strong>Not an ECG image.</strong>{' '}
                                {error.replace('Invalid ECG image:', '').trim()}
                                <br /><span className="text-rose-500 mt-0.5 block">Please upload a real ECG scan (JPG/PNG of an ECG printout).</span>
                              </>
                            : error
                          }
                        </span>
                      </div>
                    )}

                    <button onClick={handleAnalyze} disabled={!file || !file.size || isLoading}
                      className={`w-full mt-3 py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
                        ${(!file || !file.size || isLoading) ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                          : 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md shadow-blue-200 hover:shadow-lg hover:-translate-y-0.5'}`}>
                      {isLoading
                        ? <><svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                          </svg> Analyzing…</>
                        : <><Zap className="w-4 h-4" /> Analyze ECG</>
                      }
                    </button>
                  </div>
                </div>

                {/* Results column */}
                <div className="lg:col-span-3">
                  <AnimatePresence mode="wait">
                    {/* Empty state */}
                    {!result && !isLoading && (
                      <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="glass-card p-12 flex flex-col items-center justify-center text-center min-h-[420px]">
                        <EcgAnimation />
                        <h3 className="text-base font-semibold text-slate-700 mb-2">Ready for Analysis</h3>
                        <p className="text-sm text-slate-400 max-w-xs">
                          Upload an ECG image and click <strong>Analyze ECG</strong> — or press <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-xs font-mono">Space</kbd>
                        </p>
                        <div className="mt-4 flex gap-3 text-xs text-slate-400">
                          <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-emerald-400" />Drag &amp; drop</span>
                          <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-emerald-400" />Any ECG format</span>
                          <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-emerald-400" />~15s (QSVC) / ~1s (SVM)</span>
                        </div>
                      </motion.div>
                    )}

                    {/* Loading state — animated step-by-step pipeline */}
                    {isLoading && (
                      <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        <LivePipelineLoader model={selectedModel} />
                      </motion.div>
                    )}

                    {/* Results — AlertBanner + ResultPanel + ComparisonPanel */}
                    {result && !isLoading && (
                      <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-4">
                        <AlertBanner result={result} />
                        <ResultPanel result={result} file={file} selectedModel={selectedModel} resultRef={resultRef} />
                        <PipelineFlow result={result} originalImage={preview} />
                        <ComparisonPanel file={file} currentModel={selectedModel} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>
            )}

            {/* ── Model Performance Tab ─────────── */}
            {activeTab === 'performance' && (
              <motion.div key="perf" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <ModelPerformanceTab />
              </motion.div>
            )}

            {/* ── Quantum Insights Tab ─────────── */}
            {activeTab === 'quantum' && (
              <motion.div key="qml" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <QuantumInsightsTab />
              </motion.div>
            )}

            {/* ── Batch Upload Tab ─────────────── */}
            {activeTab === 'batch' && (
              <motion.div key="batch" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <BatchUploadTab selectedModel={selectedModel} />
              </motion.div>
            )}

            {/* ── API Docs Tab ─────────────────── */}
            {activeTab === 'api' && (
              <motion.div key="api" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <div className="glass-card overflow-hidden h-[800px] border-slate-200">
                  <div className="bg-slate-800 text-white px-4 py-2 text-xs font-semibold flex items-center justify-between">
                    <span className="flex items-center gap-2"><Code className="w-4 h-4" /> Swagger UI Interactive Documentation</span>
                    <a href="/docs" target="_blank" rel="noreferrer" className="text-blue-400 hover:text-blue-300 flex items-center gap-1">Open in new tab <Zap className="w-3 h-3"/></a>
                  </div>
                  <iframe src="/docs" className="w-full h-full border-0 bg-white" title="API Docs" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="mt-4 pb-6 text-center text-[11px] text-slate-400">
        <p>QuCardio v2.0 — Hybrid Classical-Quantum ECG Classification · Prabhu et al. (2023) · SJEC, Mangaluru</p>
        <p className="mt-0.5">⚠️ Not a medical device. Do not use for clinical diagnosis without qualified physician review.</p>
      </footer>
    </div>
  );
}
