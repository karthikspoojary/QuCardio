import React, { useState, useRef } from 'react';
import { Upload, Activity, AlertTriangle, FileText, Download, FileJson } from 'lucide-react';

const MAX_BATCH_FILES = 20;

// CSV-safe: escape double quotes and neutralise Excel formula prefixes (=, +, -, @)
function csvCell(value) {
  if (value == null) return '""';
  const str = String(value).replace(/"/g, '""');
  // Prefix with a tab if the value starts with a formula character so Excel doesn't execute it
  const safe = /^[=+\-@]/.test(str) ? '\t' + str : str;
  return `"${safe}"`;
}

export default function BatchUploadTab({ selectedModel }) {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const ref = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (droppedFiles.length > 0) {
      setFiles(prev => [...prev, ...droppedFiles].slice(0, MAX_BATCH_FILES));
    }
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    if (selectedFiles.length > 0) {
      setFiles(prev => [...prev, ...selectedFiles].slice(0, MAX_BATCH_FILES));
    }
  };

  const handleProcess = async () => {
    if (files.length === 0) return;
    setIsProcessing(true);
    setError(null);
    setResults([]);

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    try {
      // Use relative URL so nginx proxy works in Docker/deployed environments
      const res = await fetch(`/predict/batch?model=${encodeURIComponent(selectedModel)}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Batch processing failed on server');
      }

      const data = await res.json();
      setResults(data.batch_results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadCSV = () => {
    if (results.length === 0) return;

    const rows = ['Filename,Status,Prediction,Confidence,Triage_Action'];

    results.forEach(r => {
      if (r.status === 'success') {
        const conf = (r.result.confidence * 100).toFixed(1) + '%';
        // Use the backend's class_info.action for correct triage (e.g., MI = Emergency Care, not Standard)
        const triage = r.result.confidence < 0.75
          ? 'REFER_TO_CARDIOLOGIST'
          : (r.result.class_info?.action ?? 'No action required');
        rows.push([
          csvCell(r.filename),
          csvCell(r.status),
          csvCell(r.result.prediction?.replace(/_/g, ' ')),
          csvCell(conf),
          csvCell(triage),
        ].join(','));
      } else {
        rows.push([
          csvCell(r.filename),
          csvCell(r.status),
          csvCell(`Error: ${r.detail}`),
          '""',
          '""',
        ].join(','));
      }
    });

    const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `batch_results_${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Download PDF — match by result index (not filename) to handle duplicate basenames
  const downloadPDF = async (resultIndex) => {
    const fileObj = files[resultIndex];
    const r = results[resultIndex];
    if (!fileObj || !r) return;
    try {
      const formData = new FormData();
      formData.append('file', fileObj);
      // Pass model as query param (not form field) — the backend /predict/pdf declares it as a query param
      const res = await fetch(`/predict/pdf?model=${encodeURIComponent(selectedModel)}`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Failed to generate PDF');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `QuCardio_Report_${fileObj.name}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Error downloading PDF: ' + err.message);
    }
  };

  // Derive triage label from class_info.action (correct per Copilot review)
  const triageLabel = (r) => {
    if (r.result.confidence < 0.75) return { text: 'REFER TO CARDIOLOGIST', cls: 'bg-rose-100 text-rose-700' };
    const severity = r.result.class_info?.severity;
    if (severity === 'critical') return { text: 'EMERGENCY CARE', cls: 'bg-red-100 text-red-800 font-bold' };
    if (severity === 'warning')  return { text: 'FOLLOW UP', cls: 'bg-amber-100 text-amber-700' };
    return { text: 'Standard Care', cls: 'bg-slate-100 text-slate-600' };
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="glass-card p-6">
        <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" /> Batch Upload &amp; Processing
        </h2>
        {files.length >= MAX_BATCH_FILES && (
          <p className="text-xs text-amber-600 mb-2">⚠️ Maximum {MAX_BATCH_FILES} files per batch reached.</p>
        )}

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
            <p className="font-semibold text-slate-700 text-sm">Drop folder or multiple ECG images here</p>
            <p className="text-xs text-slate-400 mt-0.5">PNG, JPG · Max {MAX_BATCH_FILES} files per batch</p>
          </div>
          <input ref={ref} type="file" multiple accept="image/*" className="hidden" onChange={handleFileSelect} />
        </div>

        {files.length > 0 && (
          <div className="mt-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">{files.length} file{files.length !== 1 ? 's' : ''} selected</span>
              <button
                onClick={() => { setFiles([]); setResults([]); }}
                className="text-xs text-rose-500 hover:text-rose-600 font-medium"
              >
                Clear All
              </button>
            </div>

            <button
              onClick={handleProcess}
              disabled={isProcessing}
              className={`w-full py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
                ${isProcessing ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md shadow-blue-200 hover:shadow-lg hover:-translate-y-0.5'}`}
            >
              {isProcessing ? (
                <><Activity className="w-4 h-4 animate-spin" /> Processing {files.length} images...</>
              ) : (
                <><Activity className="w-4 h-4" /> Run Batch Analysis</>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 flex items-start gap-2 p-3 bg-rose-50 border border-rose-200 rounded-xl text-sm text-rose-700">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {results.length > 0 && (
        <div className="glass-card p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-md font-bold text-slate-800">
              Batch Results — {results.filter(r => r.status === 'success').length}/{results.length} succeeded
            </h3>
            <button
              onClick={downloadCSV}
              className="px-3 py-1.5 text-xs font-semibold flex items-center gap-1.5 text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors"
            >
              <Download className="w-3.5 h-3.5" /> Export CSV
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="text-xs text-slate-500 bg-slate-50 uppercase">
                <tr>
                  <th className="px-4 py-3 rounded-tl-lg">Filename</th>
                  <th className="px-4 py-3">Prediction</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Triage</th>
                  <th className="px-4 py-3 rounded-tr-lg">Action</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => {
                  if (r.status !== 'success') {
                    return (
                      <tr key={i} className="border-b border-slate-100">
                        <td className="px-4 py-3 font-medium text-slate-800">{r.filename}</td>
                        <td colSpan="4" className="px-4 py-3 text-rose-500">Error: {r.detail}</td>
                      </tr>
                    );
                  }

                  const isLowConf = r.result.confidence < 0.75;
                  const triage = triageLabel(r);

                  return (
                    <tr key={i} className={`border-b border-slate-100 hover:bg-slate-50 ${isLowConf ? 'bg-rose-50/30' : ''}`}>
                      <td className="px-4 py-3 font-medium text-slate-800 truncate max-w-[150px]" title={r.filename}>
                        {r.filename}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`font-semibold ${r.result.prediction?.includes('Normal') ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {r.result.prediction?.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span>{(r.result.confidence * 100).toFixed(1)}%</span>
                          {isLowConf && <AlertTriangle className="w-3 h-3 text-rose-500" title="Low Confidence" />}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-xs ${triage.cls}`}>{triage.text}</span>
                      </td>
                      <td className="px-4 py-3">
                        {/* Index-based matching — safe with duplicate filenames */}
                        <button
                          onClick={() => downloadPDF(i)}
                          className="text-blue-600 hover:text-blue-800 flex items-center gap-1 text-xs font-semibold"
                          disabled={!files[i]}
                        >
                          <FileJson className="w-3 h-3" /> PDF
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
