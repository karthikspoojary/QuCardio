import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, Upload, Image, AlertTriangle, CheckCircle,
  Zap, ChevronRight, Info, Eye, Cpu, FlaskConical,
  HeartPulse, BarChart3, ShieldAlert, X
} from 'lucide-react';

// ─── Helpers ────────────────────────────────────────────────────────────────
const SEVERITY_STYLES = {
  normal:   { badge: 'severity-normal',   icon: CheckCircle,  color: '#15803d' },
  warning:  { badge: 'severity-warning',  icon: AlertTriangle, color: '#b45309' },
  critical: { badge: 'severity-critical', icon: ShieldAlert,  color: '#be123c' },
};

const CONFIDENCE_BAR_COLOR = {
  high:   'bg-emerald-500',
  medium: 'bg-amber-400',
  low:    'bg-rose-500',
};

const CLASS_LABELS = {
  Normal: 'Normal',
  Arrhythmia: 'Arrhythmia',
  Myocardial_Infarction: 'Myocardial Infarction',
  History_of_MI: 'History of MI',
};

// ─── Sub-components ─────────────────────────────────────────────────────────

function ModelToggle({ isQuantum, onToggle }) {
  return (
    <div className="flex items-center gap-3 p-3 glass-card rounded-xl">
      <div className="flex items-center gap-2 text-sm">
        <Cpu className="w-4 h-4 text-blue-600" />
        <span className={!isQuantum ? 'font-semibold text-blue-700' : 'text-slate-400'}>Classical SVM</span>
      </div>
      <button onClick={onToggle} className="toggle-switch" aria-label="Toggle model">
        <div className={`toggle-track ${isQuantum ? 'active' : ''}`}>
          <div className="toggle-thumb" />
        </div>
      </button>
      <div className="flex items-center gap-2 text-sm">
        <FlaskConical className="w-4 h-4 text-violet-600" />
        <span className={isQuantum ? 'font-semibold text-violet-700' : 'text-slate-400'}>Quantum SVC</span>
        <span className="text-[10px] font-semibold bg-violet-100 text-violet-700 px-1.5 py-0.5 rounded-full">Soon</span>
      </div>
    </div>
  );
}

function DropZone({ onFileSelect, isDragging, setIsDragging }) {
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) onFileSelect(f);
  };

  return (
    <div
      className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center gap-3 py-12 px-6 transition-all duration-200 cursor-pointer
        ${isDragging ? 'dropzone-active border-blue-400' : 'border-slate-200 hover:border-blue-300 hover:bg-blue-50/40'}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <div className={`w-14 h-14 rounded-full flex items-center justify-center ${isDragging ? 'bg-blue-100' : 'bg-slate-100'}`}>
        <Upload className={`w-7 h-7 ${isDragging ? 'text-blue-500' : 'text-slate-400'}`} />
      </div>
      <div className="text-center">
        <p className="font-semibold text-slate-700">Drop ECG image here</p>
        <p className="text-sm text-slate-400 mt-1">PNG, JPG or JPEG · Max 10 MB</p>
      </div>
      <button className="mt-1 px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors">
        Browse Files
      </button>
      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={e => e.target.files?.[0] && onFileSelect(e.target.files[0])} />
    </div>
  );
}

function ImageTabs({ original, preprocessed }) {
  const [tab, setTab] = useState('original');
  return (
    <div className="rounded-2xl overflow-hidden border border-slate-200">
      <div className="flex border-b border-slate-200 bg-slate-50">
        {[{ id: 'original', label: 'Original', Icon: Image }, { id: 'preprocessed', label: 'Preprocessed (340×340)', Icon: Eye }].map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors flex-1 justify-center
              ${tab === id ? 'tab-active bg-white' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <Icon className="w-3.5 h-3.5" /> {label}
          </button>
        ))}
      </div>
      <div className="relative bg-slate-900 flex items-center justify-center min-h-[220px]">
        <AnimatePresence mode="wait">
          {tab === 'original' ? (
            <motion.img key="orig" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              src={original} alt="Original ECG" className="max-h-[260px] w-full object-contain" />
          ) : preprocessed ? (
            <motion.img key="pre" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              src={preprocessed} alt="Preprocessed ECG" className="max-h-[260px] w-full object-contain" />
          ) : (
            <motion.div key="wait" className="text-slate-500 text-sm flex flex-col items-center gap-2">
              <Activity className="w-8 h-8 opacity-40 animate-pulse" />
              <span>Run analysis to see preprocessed output</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {tab === 'preprocessed' && preprocessed && (
        <div className="bg-blue-50 px-4 py-2 flex items-start gap-2 text-xs text-blue-700 border-t border-blue-100">
          <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          <span>OTSU adaptive thresholding + grid removal applied. This is the exact image seen by ResNet50.</span>
        </div>
      )}
    </div>
  );
}

function ProbabilityBar({ label, value, isTop }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className={isTop ? 'font-semibold text-slate-800' : 'text-slate-500'}>
          {CLASS_LABELS[label] || label}
        </span>
        <span className={isTop ? 'font-bold text-slate-800' : 'text-slate-400'}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.9, delay: 0.1 }}
          className={`h-full rounded-full ${isTop ? 'bg-blue-500' : 'bg-slate-300'}`}
        />
      </div>
    </div>
  );
}

function ResultPanel({ result }) {
  const sev = SEVERITY_STYLES[result.class_info?.severity || 'normal'];
  const SevIcon = sev.icon;
  const isLowConf = result.confidence_level === 'low';

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col gap-4"
    >
      {/* Diagnosis card */}
      <div className="glass-card p-5">
        <div className="flex items-start justify-between mb-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Diagnosis</p>
          <span className={`flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full ${sev.badge}`}>
            <SevIcon className="w-3 h-3" />
            {result.class_info?.severity?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1" style={{ color: sev.color }}>
          {CLASS_LABELS[result.prediction] || result.prediction}
        </h2>
        <p className="text-sm text-slate-500">{result.class_info?.description}</p>

        {result.class_info?.action && (
          <div className={`mt-3 flex items-start gap-2 p-3 rounded-xl text-sm ${sev.badge}`}>
            <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span><strong>Recommended Action:</strong> {result.class_info.action}</span>
          </div>
        )}
      </div>

      {/* Confidence card */}
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
        <div className="flex items-end gap-2 mb-3">
          <span className="text-4xl font-extrabold text-slate-900">{(result.confidence * 100).toFixed(1)}</span>
          <span className="text-lg font-medium text-slate-400 mb-1">%</span>
        </div>
        <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${result.confidence * 100}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className={`h-full rounded-full ${CONFIDENCE_BAR_COLOR[result.confidence_level]}`}
          />
        </div>
        {isLowConf && (
          <div className="mt-3 flex items-start gap-2 p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span><strong>Low Confidence Warning:</strong> The model is uncertain. This may be caused by an atypical ECG style or image quality. Manual clinical review is strongly recommended.</span>
          </div>
        )}
        <p className="text-xs text-slate-400 mt-2 flex items-center gap-1">
          <Cpu className="w-3 h-3" /> {result.model_used}
        </p>
      </div>

      {/* Probabilities */}
      <div className="glass-card p-5">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" /> Class Probabilities
        </p>
        <div className="flex flex-col gap-3">
          {Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([label, value]) => (
              <ProbabilityBar key={label} label={label} value={value} isTop={label === result.prediction} />
            ))}
        </div>
      </div>
    </motion.div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isQuantum, setIsQuantum] = useState(false);

  const handleFileSelect = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(f);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    if (isQuantum) {
      setError('Quantum SVC is coming soon. Please use Classical SVM for now.');
      return;
    }
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Server error');
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message || 'Failed to connect to the API. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* ── Navbar ─────────────────────────────────── */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-blue-800 flex items-center justify-center shadow-md">
              <HeartPulse className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-lg text-slate-900 tracking-tight">QuCardio</span>
              <span className="ml-2 text-xs bg-blue-100 text-blue-700 font-semibold px-2 py-0.5 rounded-full">v1.1</span>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-6 text-sm text-slate-500">
            <span className="flex items-center gap-1.5"><Activity className="w-4 h-4 text-emerald-500" /> Pipeline: ResNet50 → SVD(9D) → SVM</span>
            <span className="flex items-center gap-1.5"><Zap className="w-4 h-4 text-blue-500" /> Accuracy: 91.4%</span>
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-10">
        {/* ── Header ─────────────────────────────────── */}
        <div className="text-center mb-10">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight mb-3">
              ECG Classification Dashboard
            </h1>
            <p className="text-slate-500 text-lg max-w-2xl mx-auto">
              Upload a 12-lead ECG image for AI-powered classification using the QuCardio methodology.
              Supports raw scans from any source via adaptive preprocessing.
            </p>
          </motion.div>
        </div>

        <div className="grid lg:grid-cols-5 gap-6 items-start">
          {/* ── Left Panel (upload + model) ──────────── */}
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45 }}
            className="lg:col-span-2 flex flex-col gap-4"
          >
            {/* Model selector */}
            <div className="glass-card p-4">
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Select Model</p>
              <ModelToggle isQuantum={isQuantum} onToggle={() => setIsQuantum(q => !q)} />
              {isQuantum && (
                <p className="mt-2 text-xs text-violet-600 bg-violet-50 p-2 rounded-lg">
                  Quantum SVC will be available in the next branch. Currently using Classical SVM as fallback.
                </p>
              )}
            </div>

            {/* Upload card */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Upload ECG</p>
                {file && (
                  <button onClick={handleClear} className="text-slate-400 hover:text-slate-600 transition-colors">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {!file ? (
                <DropZone onFileSelect={handleFileSelect} isDragging={isDragging} setIsDragging={setIsDragging} />
              ) : (
                <ImageTabs
                  original={preview}
                  preprocessed={result?.preprocessed_image || null}
                />
              )}

              {file && (
                <div className="mt-3 px-3 py-2 bg-slate-50 rounded-lg flex items-center justify-between text-xs text-slate-500">
                  <span className="truncate max-w-[180px] font-medium">{file.name}</span>
                  <span>{(file.size / 1024).toFixed(0)} KB</span>
                </div>
              )}

              {error && (
                <div className="mt-3 flex items-start gap-2 p-3 bg-rose-50 border border-rose-200 rounded-xl text-sm text-rose-700">
                  <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  {error}
                </div>
              )}

              <button
                onClick={handleAnalyze}
                disabled={!file || isLoading}
                className={`w-full mt-4 py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
                  ${!file || isLoading
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                    : isQuantum
                      ? 'bg-gradient-to-r from-violet-600 to-purple-700 text-white shadow-md shadow-violet-200 hover:shadow-lg hover:-translate-y-0.5'
                      : 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md shadow-blue-200 hover:shadow-lg hover:-translate-y-0.5'
                  }`}
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Analyzing ECG...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Analyze ECG
                  </>
                )}
              </button>
            </div>

            {/* Pipeline info */}
            <div className="glass-card p-4">
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Pipeline (QuCardio Paper)</p>
              <ol className="flex flex-col gap-2.5 text-xs text-slate-600">
                {[
                  ['OTSU Adaptive Preprocessing', '340×340 grayscale'],
                  ['ResNet50 Feature Extraction', 'pool1_pool layer'],
                  ['Truncated SVD', '462K → 9 dimensions'],
                  ['MinMax Scaling', '[0, 1] normalization'],
                  ['RBF SVM Classifier', 'C=500, γ=scale'],
                ].map(([step, detail], i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 text-blue-600 font-bold flex items-center justify-center text-[10px]">{i + 1}</span>
                    <span><strong>{step}</strong> — {detail}</span>
                  </li>
                ))}
              </ol>
            </div>
          </motion.div>

          {/* ── Right Panel (results) ────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45, delay: 0.1 }}
            className="lg:col-span-3"
          >
            <AnimatePresence mode="wait">
              {!result && !isLoading && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="glass-card p-12 flex flex-col items-center justify-center text-center min-h-[500px]">
                  <div className="w-20 h-20 rounded-full bg-slate-50 border border-slate-200 flex items-center justify-center mb-5">
                    <HeartPulse className="w-10 h-10 text-slate-300" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-700 mb-2">No Analysis Yet</h3>
                  <p className="text-sm text-slate-400 max-w-xs">
                    Upload an ECG image on the left and click <strong>Analyze ECG</strong> to receive AI-powered classification results.
                  </p>
                </motion.div>
              )}

              {isLoading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="glass-card p-12 flex flex-col items-center justify-center text-center min-h-[500px]">
                  <div className="relative w-20 h-20 mb-6">
                    <div className="absolute inset-0 rounded-full border-t-2 border-blue-500 animate-spin" />
                    <div className="absolute inset-2 rounded-full border-r-2 border-blue-200 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
                    <HeartPulse className="absolute inset-0 m-auto w-8 h-8 text-blue-500 animate-pulse" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-800 mb-4">Processing Pipeline</h3>
                  <div className="text-left flex flex-col gap-2.5 text-sm">
                    {[
                      'Adaptive OTSU preprocessing...',
                      'Extracting ResNet50 pool1_pool features...',
                      'Applying Truncated SVD (9D)...',
                      'Running SVM classifier...',
                    ].map((step, i) => (
                      <div key={i} className="flex items-center gap-2" style={{ animationDelay: `${i * 0.3}s` }}>
                        <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" style={{ animationDelay: `${i * 0.2}s` }} />
                        <span className="text-slate-500">{step}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {result && !isLoading && (
                <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <ResultPanel result={result} />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* ── Footer ─────────────────────────────────── */}
        <footer className="mt-12 pt-6 border-t border-slate-200 text-center text-xs text-slate-400">
          <p>QuCardio — ECG Classification System · Based on Prabhu et al. · For research purposes only.</p>
          <p className="mt-1">⚠️ Not a medical device. Do not use for clinical diagnosis without qualified physician review.</p>
        </footer>
      </main>
    </div>
  );
}
