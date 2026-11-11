import './index.css'
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Upload, TrendingUp, Calendar, DollarSign, Download, Play, Loader, BarChart3, FileText, Wifi, WifiOff } from 'lucide-react';

// Set VITE_API_URL in .env (see .env.example); the default is the hosted API.
const API_URL = import.meta.env.VITE_API_URL || 'https://retail-sales-api.onrender.com';

const fmt = (n, digits = 1) => (Number.isFinite(n) ? n.toFixed(digits) : '–');

const RetailSalesML = () => {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [training, setTraining] = useState(false);
  const [results, setResults] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [logs, setLogs] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [apiStatus, setApiStatus] = useState('checking');

  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        setApiStatus('connected');
        addLog('✓ API connection successful');
      } else {
        setApiStatus('error');
      }
    } catch {
      setApiStatus('error');
      addLog("✗ Can't reach the API. The free server sleeps when idle and needs about a minute to wake up; try again shortly.");
    }
  };

  const addLog = (message) => {
    setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message }]);
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      addLog(`File loaded: ${uploadedFile.name}`);
      
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const rows = text.split('\n').filter(row => row.trim());
        
        const parsedData = rows.slice(1).map(row => {
          const values = row.split(',');
          return {
            date: values[0],
            sales: parseFloat(values[1])
          };
        }).filter(row => row.date && !isNaN(row.sales));
        
        setData(parsedData);
        addLog(`Loaded ${parsedData.length} records`);
      };
      reader.readAsText(uploadedFile);
    }
  };

  const loadRealData = async () => {
    addLog('Loading real sales data (UCI Online Retail II, 2009-2011)...');
    try {
      const response = await fetch(`${API_URL}/sample?dataset=online-retail`);
      if (!response.ok) throw new Error(`API error (${response.status})`);
      const result = await response.json();
      setData(result.data);
      addLog(`✓ Loaded ${result.data.length} trading days from ${result.source}`);
    } catch (error) {
      addLog(`✗ ${error.message}. The free server may be waking up; try again in a minute.`);
    }
  };

  const generateSampleData = async () => {
    addLog('Fetching sample data from API...');
    
    try {
      const response = await fetch(`${API_URL}/sample`);
      if (response.ok) {
        const result = await response.json();
        setData(result.data);
        addLog(`✓ Created ${result.data.length} records`);
      } else {
        throw new Error('API error');
      }
    } catch {
      // Fallback to local generation
      addLog('Generating local sample data...');
      const sampleData = [];
      const startDate = new Date('2023-01-01');
      
      for (let i = 0; i < 365; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        
        const dayOfWeek = date.getDay();
        const month = date.getMonth();
        
        let sales = 1000;
        sales += Math.sin(i / 365 * 2 * Math.PI) * 200;
        sales += (dayOfWeek === 0 || dayOfWeek === 6) ? 300 : 0;
        sales += (month === 11) ? 500 : 0;
        sales += Math.random() * 100 - 50;
        sales += i * 0.5;
        
        sampleData.push({
          date: date.toISOString().split('T')[0],
          sales: Math.round(sales)
        });
      }
      
      setData(sampleData);
      addLog(`Created ${sampleData.length} records`);
    }
  };

  const trainModel = async () => {
    if (!data || data.length === 0) {
      addLog('Error: No data available');
      return;
    }

    if (data.length < 100) {
      addLog('Error: At least 100 records are required');
      return;
    }

    setTraining(true);
    addLog('Starting ML model training...');

    try {
      // Try FastAPI backend first
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      });

      const analysisResults = await response.json();
      if (!response.ok) {
        throw new Error(analysisResults.detail || `API request failed (${response.status})`);
      }

      showResults(analysisResults);
    } catch (error) {
      console.error('Error:', error);
      // No fake results: if the API is unavailable, say so.
      addLog(`✗ ${error.message}`);
      addLog('If the server was asleep it is waking up now; press "Train" again in a minute.');
    } finally {
      setTraining(false);
    }
  };

  const trainModelWithFile = async () => {
    if (!file) {
      addLog('Error: No file selected');
      return;
    }

    setTraining(true);
    addLog('Uploading file and training...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      const analysisResults = await response.json();
      if (!response.ok) {
        throw new Error(analysisResults.detail || `Upload failed (${response.status})`);
      }

      showResults(analysisResults);
    } catch (error) {
      console.error('Error:', error);
      addLog(`✗ ${error.message}`);
    } finally {
      setTraining(false);
    }
  };

  const showResults = (analysisResults) => {
    setResults(analysisResults);
    setPredictions(analysisResults.test_predictions);
    const best = analysisResults.models.find(m => m.name === analysisResults.best_model);
    const bt = analysisResults.backtest;
    addLog(`✓ Backtest: ${bt.folds} cut-offs × ${bt.horizon_days}-day forecasts`);
    addLog(`✓ Best: ${analysisResults.best_model} (WAPE ${fmt(best.wape)}%, ${fmt(best.skill_vs_seasonal_naive)}% vs same-day-last-week)`);
  };

  const downloadCSV = () => {
    if (!predictions) return;
    
    const csv = 'Date,Actual,Predicted,SeasonalNaive\n' +
      predictions.map(p => `${p.date},${p.actual},${p.predicted},${p.baseline}`).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    a.click();
    addLog('Downloaded predictions as CSV');
  };

  const bestModel = results?.models.find(m => m.name === results.best_model);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-emerald-900 to-slate-900">
      {/* Header */}
      <div className="bg-black bg-opacity-30 backdrop-blur-sm border-b border-emerald-500/30">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-r from-emerald-500 to-green-500 p-3 rounded-lg">
                <TrendingUp size={32} className="text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Retail Sales AI</h1>
                <p className="text-emerald-300">FastAPI + Machine Learning</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
                apiStatus === 'connected' ? 'bg-green-500/20 text-green-300' :
                apiStatus === 'error' ? 'bg-red-500/20 text-red-300' :
                'bg-yellow-500/20 text-yellow-300'
              }`}>
                {apiStatus === 'connected' ? <Wifi size={18} /> : <WifiOff size={18} />}
                <span className="text-sm font-semibold">
                  {apiStatus === 'connected' ? 'API Online' :
                   apiStatus === 'error' ? 'API Offline' :
                   'Checking...'}
                </span>
              </div>
              <button
                onClick={downloadCSV}
                disabled={!predictions}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                <Download size={18} />
                Export
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Section */}
        {!data && (
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-8 border border-emerald-500/30">
              <div className="text-center">
                <Upload className="mx-auto text-emerald-400 mb-4" size={48} />
                <h2 className="text-2xl font-bold text-white mb-3">Upload Data</h2>
                <p className="text-emerald-200 mb-6">CSV with columns: date, sales</p>
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <div className="bg-emerald-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-emerald-700 transition inline-block">
                    Choose File
                  </div>
                </label>
                {file && (
                  <p className="mt-3 text-green-300 text-sm">✓ {file.name}</p>
                )}
              </div>
            </div>

            <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-8 border border-emerald-500/30">
              <div className="text-center">
                <BarChart3 className="mx-auto text-green-400 mb-4" size={48} />
                <h2 className="text-2xl font-bold text-white mb-3">Demo Data</h2>
                <p className="text-emerald-200 mb-6">2 years of synthetic data, or real UK retail sales (2009–2011)</p>
                <button
                  onClick={generateSampleData}
                  className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition"
                >
                  Generate Sample
                </button>
                <button
                  onClick={loadRealData}
                  className="ml-3 bg-emerald-800 text-white px-6 py-3 rounded-lg font-semibold hover:bg-emerald-900 transition"
                >
                  Real Data (UCI)
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Train Model */}
        {data && !results && (
          <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-8 border border-emerald-500/30 mb-8">
            <div className="text-center">
              <div className="flex items-center justify-center gap-4 mb-6">
                <div className="bg-green-500 w-3 h-3 rounded-full animate-pulse"></div>
                <p className="text-xl text-white">
                  Loaded <span className="font-bold text-emerald-300">{data.length}</span> records
                </p>
              </div>
              
              <div className="flex gap-4 justify-center">
                {file && apiStatus === 'connected' && (
                  <button
                    onClick={trainModelWithFile}
                    disabled={training}
                    className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white text-lg font-bold rounded-xl hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 transition shadow-lg"
                  >
                    {training ? (
                      <>
                        <Loader className="animate-spin" size={24} />
                        Upload & Train...
                      </>
                    ) : (
                      <>
                        <Upload size={24} />
                        Upload to FastAPI
                      </>
                    )}
                  </button>
                )}
                
                <button
                  onClick={trainModel}
                  disabled={training}
                  className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-emerald-600 to-green-600 text-white text-lg font-bold rounded-xl hover:from-emerald-700 hover:to-green-700 disabled:opacity-50 transition shadow-lg"
                >
                  {training ? (
                    <>
                      <Loader className="animate-spin" size={24} />
                      Training...
                    </>
                  ) : (
                    <>
                      <Play size={24} />
                      Train Models
                    </>
                  )}
                </button>
              </div>
              
              {apiStatus === 'error' && (
                <p className="mt-4 text-yellow-300 text-sm">
                  ⚠️ API offline - local mode will be used
                </p>
              )}
            </div>
          </div>
        )}

        {/* Results Dashboard */}
        {results && (
          <>
            {/* Tabs */}
            <div className="flex gap-2 mb-6 overflow-x-auto">
              {['overview', 'predictions', 'features', 'logs'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-3 rounded-lg font-semibold transition whitespace-nowrap ${
                    activeTab === tab
                      ? 'bg-emerald-600 text-white'
                      : 'bg-white bg-opacity-10 text-emerald-200 hover:bg-opacity-20'
                  }`}
                >
                  {tab === 'overview' && 'Overview'}
                  {tab === 'predictions' && 'Predictions'}
                  {tab === 'features' && 'Features'}
                  {tab === 'logs' && 'Logs'}
                </button>
              ))}
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {bestModel && (
                    <>
                      <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl p-6 text-white">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm opacity-90">Better than same day last week</p>
                          <TrendingUp size={20} />
                        </div>
                        <p className="text-3xl font-bold">{fmt(bestModel.skill_vs_seasonal_naive)}%</p>
                      </div>
                      <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl p-6 text-white">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm opacity-90">MAE</p>
                          <DollarSign size={20} />
                        </div>
                        <p className="text-3xl font-bold">{fmt(bestModel.mae, 0)}</p>
                      </div>
                      <div className="bg-gradient-to-br from-emerald-500 to-green-600 rounded-xl p-6 text-white">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm opacity-90">WAPE (30-day forecasts)</p>
                          <BarChart3 size={20} />
                        </div>
                        <p className="text-3xl font-bold">{fmt(bestModel.wape)}%</p>
                      </div>
                      <div className="bg-gradient-to-br from-orange-500 to-red-600 rounded-xl p-6 text-white">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm opacity-90">RMSE</p>
                          <Calendar size={20} />
                        </div>
                        <p className="text-3xl font-bold">{fmt(bestModel.rmse, 0)}</p>
                      </div>
                    </>
                  )}
                </div>

                <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-6 border border-emerald-500/30 mb-6">
                  <h3 className="text-xl font-bold text-white mb-1">Model Comparison</h3>
                  <p className="text-emerald-300 text-sm mb-4">
                    30-day forecasts from {results.backtest.folds} cut-off dates ({results.backtest.cutoffs.join(', ')}), each model
                    feeding its own predictions forward. Baselines are included: a model is only useful if it beats them.
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="border-b border-emerald-500/30">
                          <th className="pb-3 text-emerald-300 font-semibold">Model</th>
                          <th className="pb-3 text-emerald-300 font-semibold">MAE</th>
                          <th className="pb-3 text-emerald-300 font-semibold">WAPE</th>
                          <th className="pb-3 text-emerald-300 font-semibold">MASE</th>
                          <th className="pb-3 text-emerald-300 font-semibold">vs seasonal naive</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...results.models].sort((a, b) => a.mae - b.mae).map((model, i) => (
                          <tr key={i} className={`border-b border-emerald-500/10 ${model.name === results.best_model ? 'bg-emerald-500/20' : ''}`}>
                            <td className="py-3 text-white font-medium">
                              {model.name}
                              {model.kind === 'baseline' && (
                                <span className="ml-2 text-xs bg-slate-600 px-2 py-1 rounded">baseline</span>
                              )}
                              {model.name === results.best_model && (
                                <span className="ml-2 text-xs bg-green-500 px-2 py-1 rounded">BEST</span>
                              )}
                            </td>
                            <td className="py-3 text-emerald-200">{fmt(model.mae, 2)}</td>
                            <td className="py-3 text-emerald-200">{fmt(model.wape)}%</td>
                            <td className="py-3 text-emerald-200">{fmt(model.mase, 2)}</td>
                            <td className="py-3 text-emerald-200">{model.skill_vs_seasonal_naive > 0 ? '+' : ''}{fmt(model.skill_vs_seasonal_naive)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {/* Predictions Tab */}
            {activeTab === 'predictions' && predictions && (
              <>
                <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-6 border border-emerald-500/30 mb-6">
                  <h3 className="text-xl font-bold text-white mb-4">Last backtest window: 30-day forecast vs actual</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={predictions}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                      <XAxis 
                        dataKey="date" 
                        stroke="#6ee7b7"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      />
                      <YAxis stroke="#6ee7b7" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#052e16', border: '1px solid #10b981', borderRadius: '8px' }}
                        labelStyle={{ color: '#6ee7b7' }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={2} name="Actual" />
                      <Line type="monotone" dataKey="predicted" stroke="#22c55e" strokeWidth={2} strokeDasharray="5 5" name={results.best_model} />
                      <Line type="monotone" dataKey="baseline" stroke="#94a3b8" strokeWidth={1} strokeDasharray="2 4" name="Same day last week" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {results.future_predictions && (
                  <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-6 border border-emerald-500/30">
                    <h3 className="text-xl font-bold text-white mb-4">Next 30 days (80% interval from backtest errors)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={results.future_predictions}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                        <XAxis 
                          dataKey="date" 
                          stroke="#6ee7b7"
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        />
                        <YAxis stroke="#6ee7b7" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#052e16', border: '1px solid #10b981', borderRadius: '8px' }}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="upper" stroke="#6ee7b7" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Upper (90%)" />
                        <Line type="monotone" dataKey="predicted" stroke="#34d399" strokeWidth={3} name="Forecast" />
                        <Line type="monotone" dataKey="lower" stroke="#6ee7b7" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Lower (10%)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            )}

            {/* Features Tab */}
            {activeTab === 'features' && results.feature_importance && (
              <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-6 border border-emerald-500/30">
                <h3 className="text-xl font-bold text-white mb-4">Feature Importance ({results.feature_importance_model})</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={results.feature_importance} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis type="number" stroke="#6ee7b7" />
                    <YAxis type="category" dataKey="feature" stroke="#6ee7b7" width={150} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#052e16', border: '1px solid #10b981', borderRadius: '8px' }}
                    />
                    <Bar dataKey="importance" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Logs Tab */}
            {activeTab === 'logs' && (
              <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-6 border border-emerald-500/30">
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="text-emerald-400" size={24} />
                  <h3 className="text-xl font-bold text-white">System Logs</h3>
                </div>
                <div className="bg-black bg-opacity-50 rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-sm">
                  {logs.map((log, i) => (
                    <div key={i} className="text-green-400 mb-1">
                      <span className="text-emerald-400">[{log.time}]</span> {log.message}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-6 text-center">
              <button
                onClick={() => {
                  setData(null);
                  setResults(null);
                  setPredictions(null);
                  setLogs([]);
                  setFile(null);
                }}
                className="px-6 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition"
              >
                New Analysis
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default RetailSalesML;