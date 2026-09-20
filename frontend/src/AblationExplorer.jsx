import { useState, useEffect, useMemo } from 'react';
import { Search, Filter, AlertCircle, Database, ChevronDown, ChevronUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';

export default function AblationExplorer() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortConfig, setSortConfig] = useState({ key: 'kta', direction: 'desc' });
  const [filterEncoding, setFilterEncoding] = useState('all');

  useEffect(() => {
    fetch('/ablation_data')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load ablation data');
        return res.json();
      })
      .then(json => {
        // Data format: { "key": { encoding, entanglement, reps, C, kta } }
        const arr = Object.values(json).map(item => ({
          ...item,
          kta: item.kta !== null ? item.kta : -1,
        }));
        setData(arr);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const sortedData = useMemo(() => {
    let sortableItems = [...data];
    if (filterEncoding !== 'all') {
      sortableItems = sortableItems.filter(item => item.encoding === filterEncoding);
    }
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
        if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }
    return sortableItems;
  }, [data, sortConfig, filterEncoding]);

  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) return <span className="w-4 inline-block" />;
    return sortConfig.direction === 'asc' ? <ChevronUp className="w-3 h-3 inline" /> : <ChevronDown className="w-3 h-3 inline" />;
  };

  // Group data by reps to show a chart of reps vs KTA
  const chartData = useMemo(() => {
    const map = new Map();
    sortedData.forEach(item => {
      if (item.kta <= 0) return; // skip failed runs
      const key = `${item.reps}`;
      if (!map.has(key)) map.set(key, { name: key, kta: [] });
      map.get(key).kta.push(item.kta);
    });
    return Array.from(map.values()).map(d => ({
      name: `Reps ${d.name}`,
      avgKta: d.kta.reduce((a, b) => a + b, 0) / d.kta.length,
      maxKta: Math.max(...d.kta)
    })).sort((a, b) => a.name.localeCompare(b.name));
  }, [sortedData]);

  if (loading) return <div className="p-10 text-center text-slate-500">Loading 320 ablation configurations...</div>;
  if (error) return <div className="p-10 text-center text-rose-500 flex justify-center items-center gap-2"><AlertCircle className="w-5 h-5"/> {error}</div>;

  const encodings = ['all', ...Array.from(new Set(data.map(d => d.encoding)))];

  return (
    <div className="flex flex-col gap-5">
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold text-slate-800 mb-2 flex items-center gap-2">
          <Database className="w-5 h-5 text-indigo-600" /> Quantum Feature Map Ablation Explorer
        </h2>
        <p className="text-sm text-slate-500 mb-6">
          Explore the hyperparameter sweep over 320 configurations of the QSVC model. This justifies the choice of <code className="bg-slate-100 text-pink-600 px-1 py-0.5 rounded text-xs">minmax_0pi</code>, <code className="bg-slate-100 text-pink-600 px-1 py-0.5 rounded text-xs">circular</code> entanglement, and <code className="bg-slate-100 text-pink-600 px-1 py-0.5 rounded text-xs">reps=2</code>.
        </p>

        <div className="flex gap-4 items-center mb-6">
          <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" /> Filter Encoding:
          </label>
          <select 
            className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm bg-white"
            value={filterEncoding} 
            onChange={e => setFilterEncoding(e.target.value)}
          >
            {encodings.map(enc => <option key={enc} value={enc}>{enc === 'all' ? 'All Encodings' : enc}</option>)}
          </select>
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-1 bg-slate-50 p-4 rounded-xl border border-slate-100">
                <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4">Max KTA Score by Repetitions</h3>
                <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                            <XAxis dataKey="name" tick={{fontSize: 10}} axisLine={false} tickLine={false} />
                            <YAxis domain={['auto', 'auto']} tick={{fontSize: 10}} axisLine={false} tickLine={false} />
                            <Tooltip contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                            <Line type="monotone" dataKey="maxKta" stroke="#6366f1" strokeWidth={2} dot={{r: 4}} name="Max KTA" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
            
            <div className="lg:col-span-2">
              <div className="overflow-x-auto border border-slate-200 rounded-xl max-h-[400px] overflow-y-auto">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-50 sticky top-0 z-10">
                    <tr className="text-xs text-slate-500 uppercase tracking-wide border-b border-slate-200">
                      <th className="py-3 px-4 font-semibold cursor-pointer hover:bg-slate-100 transition-colors" onClick={() => requestSort('encoding')}>
                        Encoding {getSortIcon('encoding')}
                      </th>
                      <th className="py-3 px-4 font-semibold cursor-pointer hover:bg-slate-100 transition-colors" onClick={() => requestSort('entanglement')}>
                        Entanglement {getSortIcon('entanglement')}
                      </th>
                      <th className="py-3 px-4 font-semibold cursor-pointer hover:bg-slate-100 transition-colors" onClick={() => requestSort('reps')}>
                        Reps {getSortIcon('reps')}
                      </th>
                      <th className="py-3 px-4 font-semibold cursor-pointer hover:bg-slate-100 transition-colors" onClick={() => requestSort('C')}>
                        C {getSortIcon('C')}
                      </th>
                      <th className="py-3 px-4 font-semibold cursor-pointer hover:bg-slate-100 transition-colors" onClick={() => requestSort('kta')}>
                        KTA Score {getSortIcon('kta')}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {sortedData.map((item, i) => (
                      <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                        <td className="py-2.5 px-4"><span className="px-2 py-1 bg-blue-50 text-blue-700 rounded font-mono text-xs">{item.encoding}</span></td>
                        <td className="py-2.5 px-4"><span className="px-2 py-1 bg-purple-50 text-purple-700 rounded font-mono text-xs">{item.entanglement}</span></td>
                        <td className="py-2.5 px-4 font-medium text-slate-700">{item.reps}</td>
                        <td className="py-2.5 px-4 text-slate-500">{item.C}</td>
                        <td className="py-2.5 px-4 font-semibold text-emerald-600">
                          {item.kta > 0 ? item.kta.toFixed(5) : <span className="text-slate-300 font-normal">Failed</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
        </div>
      </div>
    </div>
  );
}
