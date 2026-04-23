import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileType, CheckCircle, AlertTriangle, Activity, Zap, Server, ChevronRight } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      handleFileSelect(droppedFile);
    } else {
      setError("Please upload a valid image file.");
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
    
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const clearSelection = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to analyze image");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (prediction) => {
    if (prediction === 'Normal') return 'text-emerald-400';
    if (prediction === 'Unknown') return 'text-slate-400';
    return 'text-rose-400';
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-blue-500/30">
      
      {/* Navbar */}
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-300">
              QuCardio
            </span>
          </div>
          <div className="flex items-center space-x-4 text-sm font-medium text-slate-400">
            <span className="flex items-center"><Server className="w-4 h-4 mr-1.5" /> Classical ML Pipeline</span>
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-white">
              AI-Powered ECG Analysis
            </h1>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Upload an electrocardiogram image for instant classification using our ResNet50 + SVM classical pipeline, reducing 2048 dimensions to 9 via Truncated SVD.
            </p>
          </motion.div>
        </div>

        <div className="grid md:grid-cols-2 gap-8 items-start">
          
          {/* Upload Section */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-slate-800/50 rounded-2xl border border-slate-700/50 p-6 shadow-xl backdrop-blur-sm"
          >
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <FileType className="w-5 h-5 mr-2 text-blue-400" />
              Upload ECG Image
            </h2>

            {!preview ? (
              <div
                className={`border-2 border-dashed rounded-xl p-10 text-center transition-all duration-300 cursor-pointer flex flex-col items-center justify-center min-h-[300px]
                  ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 hover:border-blue-400 hover:bg-slate-700/30'}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4 shadow-inner">
                  <UploadCloud className={`w-8 h-8 ${isDragging ? 'text-blue-400' : 'text-slate-400'}`} />
                </div>
                <h3 className="text-lg font-medium text-slate-200 mb-1">Drag & drop your ECG image</h3>
                <p className="text-sm text-slate-400 mb-4">PNG, JPG or JPEG (Max 5MB)</p>
                <button className="px-4 py-2 rounded-lg bg-slate-700 text-sm font-medium hover:bg-slate-600 transition-colors">
                  Browse Files
                </button>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              <div className="rounded-xl overflow-hidden bg-slate-900 border border-slate-700 relative group">
                <img src={preview} alt="ECG Preview" className="w-full h-auto object-cover max-h-[300px] opacity-90 group-hover:opacity-100 transition-opacity" />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-4">
                  <span className="text-sm font-medium text-white truncate max-w-[200px]">{file.name}</span>
                  <button 
                    onClick={(e) => { e.stopPropagation(); clearSelection(); }}
                    className="text-xs px-3 py-1.5 bg-slate-800/80 backdrop-blur rounded-md text-slate-300 hover:text-white transition-colors"
                  >
                    Change Image
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-start text-rose-400 text-sm">
                <AlertTriangle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div className="mt-6 pt-6 border-t border-slate-700/50">
              <button
                onClick={handleAnalyze}
                disabled={!file || isLoading}
                className={`w-full py-3.5 rounded-xl font-medium flex items-center justify-center transition-all duration-300
                  ${!file || isLoading 
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-0.5'}`}
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing ECG...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5 mr-2" />
                    Run Classification Pipeline
                  </>
                )}
              </button>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="h-full"
          >
            <AnimatePresence mode="wait">
              {!result && !isLoading && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full bg-slate-800/30 rounded-2xl border border-slate-700/30 border-dashed p-8 flex flex-col items-center justify-center text-center text-slate-500 min-h-[400px]"
                >
                  <Activity className="w-12 h-12 mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">Awaiting Analysis</h3>
                  <p className="max-w-xs text-sm">Upload an ECG image and run the pipeline to see the classification results here.</p>
                </motion.div>
              )}

              {isLoading && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full bg-slate-800/50 rounded-2xl border border-slate-700/50 p-8 flex flex-col items-center justify-center min-h-[400px]"
                >
                  <div className="relative w-24 h-24 mb-6">
                    <div className="absolute inset-0 border-t-2 border-blue-500 rounded-full animate-spin"></div>
                    <div className="absolute inset-2 border-r-2 border-indigo-400 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                    <Activity className="absolute inset-0 m-auto w-8 h-8 text-blue-400 animate-pulse" />
                  </div>
                  <h3 className="text-xl font-semibold text-slate-200 mb-2">Processing Image</h3>
                  <div className="flex flex-col gap-2 w-full max-w-xs mt-4">
                    <div className="flex items-center text-xs text-slate-400"><CheckCircle className="w-3 h-3 mr-2 text-emerald-500" /> Image preprocessed & resized</div>
                    <div className="flex items-center text-xs text-slate-400"><CheckCircle className="w-3 h-3 mr-2 text-emerald-500" /> ResNet50 pool1_pool extracted</div>
                    <div className="flex items-center text-xs text-blue-400 animate-pulse"><ChevronRight className="w-3 h-3 mr-2" /> Applying Truncated SVD...</div>
                    <div className="flex items-center text-xs text-slate-500"><ChevronRight className="w-3 h-3 mr-2" /> Classical SVM inference...</div>
                  </div>
                </motion.div>
              )}

              {result && !isLoading && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-slate-800/80 rounded-2xl border border-slate-700 p-6 shadow-2xl relative overflow-hidden"
                >
                  {/* Decorative glow */}
                  <div className={`absolute -top-24 -right-24 w-48 h-48 rounded-full blur-3xl opacity-20 pointer-events-none
                    ${result.prediction === 'Normal' ? 'bg-emerald-500' : 'bg-rose-500'}`} />

                  <h2 className="text-xl font-semibold mb-6 flex items-center">
                    <CheckCircle className="w-5 h-5 mr-2 text-emerald-400" />
                    Analysis Complete
                  </h2>

                  <div className="bg-slate-900/50 rounded-xl p-6 mb-6 border border-slate-700/50">
                    <div className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-1">Predicted Class</div>
                    <div className={`text-4xl font-bold tracking-tight mb-2 ${getStatusColor(result.prediction)}`}>
                      {result.prediction.replace(/_/g, ' ')}
                    </div>
                    
                    <div className="mt-4 flex items-center justify-between">
                      <span className="text-sm text-slate-400">Confidence Score</span>
                      <span className="text-sm font-bold text-white">{(result.confidence * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-2 mt-2 overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className={`h-2 rounded-full ${result.prediction === 'Normal' ? 'bg-emerald-500' : 'bg-rose-500'}`}
                      />
                    </div>
                  </div>

                  <div className="mb-2">
                    <h3 className="text-sm font-medium text-slate-300 mb-3 px-1">Class Probabilities</h3>
                    <div className="space-y-3">
                      {Object.entries(result.probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([className, prob]) => (
                        <div key={className} className="relative">
                          <div className="flex justify-between text-xs mb-1 px-1">
                            <span className={className === result.prediction ? 'text-white font-medium' : 'text-slate-400'}>
                              {className.replace(/_/g, ' ')}
                            </span>
                            <span className="text-slate-400">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-800/50 rounded-full h-1.5 overflow-hidden">
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.8, delay: 0.2 }}
                              className={`h-full rounded-full ${className === result.prediction ? 'bg-blue-500' : 'bg-slate-600'}`}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

export default App;
