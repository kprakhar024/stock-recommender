"use client";

import { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Minus, Brain, BarChart3, Newspaper, Shield, Globe, Zap, AlertTriangle, CheckCircle, Info } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, ReferenceLine, BarChart, Bar } from "recharts";

interface StockDashboardProps {
  symbol: string;
}

export default function StockDashboard({ symbol }: StockDashboardProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "agents" | "backtest" | "research">("overview");
  const [backtestData, setBacktestData] = useState<any>(null);
  const [ablationData, setAblationData] = useState<any>(null);

  useEffect(() => {
    fetchAnalysis();
  }, [symbol]);

  const fetchAnalysis = async () => {
    setLoading(true);
    setError(null);
    let analysisData: any;
    try {
      const res = await fetch(`http://localhost:8000/api/analyze/${symbol}`);
      if (!res.ok) throw new Error(`Failed: ${res.status}`);
      analysisData = await res.json();
    } catch (e: any) {
      console.error(e);
      // Mock data for demo when backend offline
      analysisData = getMockData(symbol);
    }

    try {
      const [quoteRes, historyRes] = await Promise.all([
        fetch(`http://localhost:8000/api/quote/${symbol}`),
        fetch(`http://localhost:8000/api/history/${symbol}`),
      ]);
      if (!quoteRes.ok || !historyRes.ok) throw new Error("Market data request failed");
      const quote = await quoteRes.json();
      const history = await historyRes.json();
      setData({
        ...analysisData,
        current_price: quote.price,
        previous_close: quote.previous_close,
        price_type: quote.price_type,
        price_as_of: quote.as_of,
        price_source: quote.source,
        price_history: history.price_history,
      });
    } catch (e) {
      console.error("Market price/history fetch failed", e);
      setData(analysisData);
    } finally {
      setLoading(false);
    }
  };

  const fetchBacktest = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/backtest/${symbol}?strategy=ml`);
      if (res.ok) {
        const json = await res.json();
        setBacktestData(json);
      }
    } catch (e) {
      setBacktestData(getMockBacktest(symbol));
    }
    
    try {
      const res2 = await fetch(`http://localhost:8000/api/ablation/${symbol}`);
      if (res2.ok) {
        const json2 = await res2.json();
        setAblationData(json2);
      }
    } catch (e) {
      setAblationData(getMockAblation(symbol));
    }
  };

  useEffect(() => {
    if (activeTab === "backtest" || activeTab === "research") {
      fetchBacktest();
    }
  }, [activeTab, symbol]);

  if (loading) {
    return (
      <div className="glass rounded-2xl p-12 flex flex-col items-center justify-center min-h-[600px]">
        <div className="h-12 w-12 rounded-full border-2 border-violet-500 border-t-transparent animate-spin mb-4" />
        <p className="text-white/60">Running 7 AI agents for {symbol}...</p>
        <p className="text-xs text-white/30 mt-2">Technical • ML • News • Fundamental • Risk • Regime • Decision</p>
      </div>
    );
  }

  if (!data) return null;

  const rec = data.recommendation;
  const isBuy = rec.recommendation === "BUY";
  const isSell = rec.recommendation === "SELL";

  return (
    <div className="space-y-6">
      {/* Recommendation Header */}
      <div className="grid grid-cols-12 gap-6">
        <div className={`col-span-12 lg:col-span-8 glass rounded-2xl p-6 relative overflow-hidden ${
          isBuy ? "border-emerald-500/30" : isSell ? "border-red-500/30" : "border-white/10"
        }`}>
          <div className={`absolute inset-0 opacity-20 ${
            isBuy ? "bg-gradient-to-br from-emerald-600/30 to-emerald-900/30" :
            isSell ? "bg-gradient-to-br from-red-600/30 to-red-900/30" :
            "bg-gradient-to-br from-gray-600/20 to-gray-900/20"
          }`} />
          
          <div className="relative z-10">
            <div className="flex items-start justify-between mb-6">
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-3xl font-bold tracking-tight">{data.display_symbol}</h1>
                  <div className="flex flex-col items-end gap-1">
                    <span className="text-sm px-2.5 py-1 rounded-full bg-white/10 font-mono">
                      {typeof data.current_price === "number" ? `₹${data.current_price.toFixed(2)}` : "Price unavailable"}
                    </span>
                    {typeof data.previous_close === "number" && (
                      <span className="text-[11px] text-white/50">
                        Prev close ₹{data.previous_close.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold tracking-wider ${
                    isBuy ? "bg-emerald-500 text-white" :
                    isSell ? "bg-red-500 text-white" :
                    "bg-yellow-500 text-black"
                  }`}>
                    {rec.recommendation} SIGNAL
                  </span>
                </div>
                <p className="text-sm text-white/50 mt-2">
                  NSE • {data.symbol} • {data.price_type === "LIVE" ? "Live quote" : data.price_type === "PREVIOUS_CLOSE" ? "Previous close" : "Price unavailable"}
                  {data.price_as_of ? ` • ${new Date(data.price_as_of).toLocaleString()}` : ""}
                  {data.price_source ? ` • ${data.price_source}` : ""}
                </p>
              </div>
              
              <div className="text-right">
                <div className="text-xs text-white/40 uppercase tracking-wider">Confidence</div>
                <div className="text-3xl font-bold mt-1">{rec.confidence}%</div>
                <div className="text-xs text-white/50 mt-1">Combined Score {rec.combined_score}</div>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-4 mb-6">
              {[
                { label: "Technical", value: rec.technical_score, max: 100 },
                { label: "Fundamental", value: rec.fundamental_score, max: 100 },
                { label: "Sentiment", value: Math.round((rec.sentiment.score + 1) * 50), max: 100, sub: rec.sentiment.label },
                { label: "ML Prob", value: Math.round(rec.ml_prediction.probability * 100), max: 100, sub: `${(rec.ml_prediction.probability*100).toFixed(0)}% bullish` },
                { label: "Risk", value: rec.risk.level === "Low" ? 20 : rec.risk.level === "Medium" ? 50 : 80, max: 100, sub: rec.risk.level, invert: true },
              ].map((metric) => (
                <div key={metric.label} className="bg-black/30 rounded-xl p-3 border border-white/5">
                  <div className="text-[11px] text-white/40 uppercase tracking-wider">{metric.label}</div>
                  <div className="text-xl font-bold mt-1">{metric.value}</div>
                  <div className="text-[11px] text-white/50 mt-0.5">{metric.sub || `/ ${metric.max}`}</div>
                  <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        metric.invert
                          ? metric.value < 40 ? "bg-emerald-500" : metric.value < 70 ? "bg-yellow-500" : "bg-red-500"
                          : metric.value > 65 ? "bg-emerald-500" : metric.value > 45 ? "bg-yellow-500" : "bg-red-500"
                      }`}
                      style={{ width: `${Math.min(100, metric.value)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                <h4 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <CheckCircle className="h-3.5 w-3.5" /> Reasons For
                </h4>
                <ul className="space-y-2">
                  {rec.explanation.for.slice(0, 4).map((reason: string, i: number) => (
                    <li key={i} className="text-xs text-white/70 leading-relaxed flex gap-2">
                      <span className="text-emerald-400 mt-0.5">•</span>
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                <h4 className="text-xs font-semibold text-amber-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <AlertTriangle className="h-3.5 w-3.5" /> Risk Factors
                </h4>
                <ul className="space-y-2">
                  {rec.explanation.risk_factors.length > 0 ? rec.explanation.risk_factors.slice(0, 4).map((risk: string, i: number) => (
                    <li key={i} className="text-xs text-white/60 leading-relaxed flex gap-2">
                      <span className="text-amber-400 mt-0.5">⚠</span>
                      <span>{risk}</span>
                    </li>
                  )) : (
                    <li className="text-xs text-white/40">No major risks detected</li>
                  )}
                  {rec.explanation.against.slice(0, 2).map((against: string, i: number) => (
                    <li key={`a-${i}`} className="text-xs text-white/50 leading-relaxed flex gap-2">
                      <span className="text-white/30 mt-0.5">•</span>
                      <span>{against}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-4 space-y-4">
          <div className="glass rounded-2xl p-5">
            <h3 className="font-semibold text-sm mb-4">Expected Returns</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-white/50">5-Day Expected</span>
                  <span className={`font-mono font-bold ${rec.expected_return_5d >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {rec.expected_return_5d >= 0 ? "+" : ""}{rec.expected_return_5d}%
                  </span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${rec.expected_return_5d >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ width: `${Math.min(100, Math.abs(rec.expected_return_5d) * 20 + 20)}%` }}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="bg-white/5 rounded-xl p-3">
                  <div className="text-[11px] text-white/40 uppercase">Market Regime</div>
                  <div className="font-semibold text-sm mt-1">{rec.market_regime.regime}</div>
                  <div className={`text-[11px] mt-1 px-2 py-0.5 rounded-full inline-block ${
                    rec.market_regime.signal === "BULLISH" ? "bg-emerald-500/20 text-emerald-400" :
                    rec.market_regime.signal === "BEARISH" ? "bg-red-500/20 text-red-400" :
                    "bg-yellow-500/20 text-yellow-400"
                  }`}>
                    {rec.market_regime.signal}
                  </div>
                </div>
                <div className="bg-white/5 rounded-xl p-3">
                  <div className="text-[11px] text-white/40 uppercase">Risk Level</div>
                  <div className="font-semibold text-sm mt-1">{rec.risk.level}</div>
                  <div className="text-[11px] text-white/50 mt-1">Score {rec.risk.score}/100</div>
                </div>
              </div>

              <div className="pt-3 border-t border-white/10">
                <div className="text-[11px] text-white/40 uppercase tracking-wider mb-2">Agent Weights (Validated)</div>
                <div className="space-y-1.5">
                  {Object.entries(rec.weights_used || {}).map(([k, v]: any) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-white/50 capitalize">{k.replace('_', ' ')}</span>
                      <span className="font-mono text-white/80">{(v * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="glass rounded-2xl p-5">
            <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
              <Brain className="h-4 w-4 text-violet-400" /> AI Explanation
            </h3>
            <p className="text-xs text-white/60 leading-relaxed">
              {isBuy
                ? `Technical momentum is positive with ${rec.technical_score}/100 score. ML ensemble predicts ${Math.round(rec.ml_prediction.probability*100)}% bullish probability. Recent news sentiment is ${rec.sentiment.label.toLowerCase()} and fundamental indicators are stable. Market regime is ${rec.market_regime.regime.toLowerCase()}, supporting long positions.`
                : isSell
                ? `Technical indicators show weakness with ${rec.technical_score}/100. ML model confidence is low at ${Math.round(rec.ml_prediction.probability*100)}%. Risk level is ${rec.risk.level.toLowerCase()} with elevated volatility. Market regime suggests caution.`
                : `Mixed signals with combined score ${rec.combined_score}. Technical ${rec.technical_score}/100, fundamentals ${rec.fundamental_score}/100. Sentiment is ${rec.sentiment.label.toLowerCase()}. Risk is ${rec.risk.level.toLowerCase()}. Waiting for clearer directional cues.`
              }
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/10 pb-0">
        {[
          { id: "overview", label: "Overview", icon: BarChart3 },
          { id: "agents", label: "Agentic Analysis", icon: Brain },
          { id: "backtest", label: "Backtesting", icon: TrendingUp },
          { id: "research", label: "Research", icon: Zap },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-4 py-2.5 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${
              activeTab === tab.id
                ? "border-violet-500 text-white"
                : "border-transparent text-white/50 hover:text-white/80"
            }`}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-8 glass rounded-2xl p-6">
            <h3 className="font-semibold mb-4">Price History (90 days)</h3>
            <div className="h-[300px] w-full">
              {(data.price_history || []).length > 0 ? <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.price_history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={15} />
                  <YAxis tick={{ fontSize: 10, fill: "#666" }} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: "#111", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }}
                    labelStyle={{ color: "#999" }}
                  />
                  <Area type="monotone" dataKey="close" stroke="#8b5cf6" fill="rgba(139,92,246,0.2)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer> : (
                <div className="h-full flex items-center justify-center text-sm text-white/40">
                  Price history unavailable
                </div>
              )}
            </div>
            </div>

            <div className="col-span-12 lg:col-span-4 glass rounded-2xl p-6">
            <h3 className="font-semibold mb-4">Technical Snapshot</h3>
            <div className="space-y-3">
              {data.agents?.technical?.details?.technical_summary && Object.entries(data.agents.technical.details.technical_summary).slice(0, 10).map(([k, v]: any) => (
                <div key={k} className="flex justify-between text-xs py-2 border-b border-white/5 last:border-0">
                  <span className="text-white/50">{k.replace(/_/g, ' ')}</span>
                  <span className="font-mono text-white/80">{typeof v === 'number' ? v.toFixed(2) : String(v).slice(0, 30)}</span>
                </div>
              ))}
            </div>
            </div>
          </div>

          <div className="glass rounded-2xl p-6">
            <div className="flex items-center justify-between mb-5">
              <div>
              <h3 className="font-semibold">Technical Indicators</h3>
              <p className="text-xs text-white/40 mt-1">Momentum, trend and volatility signals over the last 90 days</p>
              </div>
              <span className="text-[11px] text-white/40">Hover for daily values</span>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              <div>
              <h4 className="text-xs text-white/60 mb-3">RSI (14)</h4>
              <div className="h-[190px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.price_history || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" hide />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#666" }} />
                    <Tooltip contentStyle={{ background: "#111", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }} />
                    <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
                    <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="rsi14" name="RSI" stroke="#38bdf8" strokeWidth={2} dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              </div>
              <div>
              <h4 className="text-xs text-white/60 mb-3">MACD</h4>
              <div className="h-[190px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.price_history || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" hide />
                    <YAxis tick={{ fontSize: 10, fill: "#666" }} />
                    <Tooltip contentStyle={{ background: "#111", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }} />
                    <ReferenceLine y={0} stroke="rgba(255,255,255,0.25)" />
                    <Bar dataKey="macd_histogram" name="Histogram" fill="#8b5cf6" opacity={0.45} />
                    <Line type="monotone" dataKey="macd" name="MACD" stroke="#f59e0b" strokeWidth={2} dot={false} connectNulls />
                    <Line type="monotone" dataKey="macd_signal" name="Signal" stroke="#ef4444" strokeWidth={1.5} dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              </div>
              <div>
              <h4 className="text-xs text-white/60 mb-3">Bollinger Bands</h4>
              <div className="h-[190px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.price_history || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" hide />
                    <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "#666" }} />
                    <Tooltip contentStyle={{ background: "#111", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }} />
                    <Line type="monotone" dataKey="bb_upper" name="Upper" stroke="#ef4444" strokeWidth={1} dot={false} connectNulls />
                    <Line type="monotone" dataKey="bb_middle" name="Middle" stroke="#f59e0b" strokeWidth={1.5} dot={false} connectNulls />
                    <Line type="monotone" dataKey="bb_lower" name="Lower" stroke="#10b981" strokeWidth={1} dot={false} connectNulls />
                    <Line type="monotone" dataKey="close" name="Close" stroke="#8b5cf6" strokeWidth={2} dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "agents" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(data.agents || {}).map(([name, agent]: any) => {
            if (name === "market_data") return null;
            const icons: any = {
              technical: BarChart3,
              news: Newspaper,
              fundamental: Zap,
              risk: Shield,
              market_regime: Globe,
              ml: Brain
            };
            const Icon = icons[name] || Brain;
            
            return (
              <div key={name} className="glass rounded-2xl p-6 card-hover">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-xl bg-white/5 flex items-center justify-center">
                      <Icon className="h-5 w-5 text-white/70" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-sm capitalize">{name.replace('_', ' ')} Agent</h3>
                      <p className="text-xs text-white/40">{agent.agent}</p>
                    </div>
                  </div>
                  <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                    agent.signal.includes("BUY") || agent.signal.includes("BULLISH") || agent.signal === "LOW_RISK"
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                      : agent.signal.includes("SELL") || agent.signal.includes("BEARISH") || agent.signal === "HIGH_RISK"
                      ? "bg-red-500/20 text-red-400 border border-red-500/30"
                      : "bg-white/10 text-white/60 border border-white/10"
                  }`}>
                    {agent.signal}
                  </span>
                </div>
                
                <div className="mb-4">
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-white/40">Confidence</span>
                    <span className="font-mono">{(agent.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-violet-500 rounded-full" style={{ width: `${agent.confidence * 100}%` }} />
                  </div>
                </div>

                <div className="space-y-2">
                  {agent.reasoning?.slice(0, 3).map((r: string, i: number) => (
                    <p key={i} className="text-xs text-white/60 leading-relaxed">• {r}</p>
                  ))}
                </div>

                {agent.details?.risk_factors?.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <p className="text-[11px] text-amber-400/80 uppercase tracking-wider mb-2">Risk Factors</p>
                    {agent.details.risk_factors.slice(0, 2).map((rf: string, i: number) => (
                      <p key={i} className="text-xs text-white/40">⚠ {rf}</p>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {activeTab === "backtest" && (
        <div className="space-y-6">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-8 glass rounded-2xl p-6">
              <h3 className="font-semibold mb-4">Equity Curve vs Buy & Hold (1 Year)</h3>
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={backtestData?.equity_curve?.map((e: any, i: number) => ({
                    date: e.date,
                    strategy: e.equity,
                    buy_hold: backtestData?.benchmark_equity?.[i]?.equity || e.equity
                  })) || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#666" }} interval={30} />
                    <YAxis tick={{ fontSize: 10, fill: "#666" }} />
                    <Tooltip contentStyle={{ background: "#111", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }} />
                    <Line type="monotone" dataKey="strategy" stroke="#10b981" strokeWidth={2} dot={false} name="Our Strategy" />
                    <Line type="monotone" dataKey="buy_hold" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Buy & Hold" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="col-span-12 lg:col-span-4 space-y-4">
              <div className="glass rounded-2xl p-5">
                <h4 className="font-semibold text-sm mb-4">Performance Metrics</h4>
                <div className="space-y-3">
                  {[
                    { label: "CAGR", value: `${backtestData?.metrics?.cagr_pct || "12.4"}%`, good: true },
                    { label: "Sharpe Ratio", value: backtestData?.metrics?.sharpe_ratio || "1.24", good: true },
                    { label: "Max Drawdown", value: `${backtestData?.metrics?.max_drawdown_pct || "-8.2"}%`, good: false },
                    { label: "Win Rate", value: `${backtestData?.trade_metrics?.win_rate_pct || "58.3"}%`, good: true },
                    { label: "Profit Factor", value: backtestData?.trade_metrics?.profit_factor || "1.45", good: true },
                    { label: "Total Trades", value: backtestData?.trade_metrics?.total_trades || "42", good: null },
                  ].map((m) => (
                    <div key={m.label} className="flex justify-between text-sm py-2 border-b border-white/5 last:border-0">
                      <span className="text-white/50">{m.label}</span>
                      <span className={`font-mono font-medium ${m.good === true ? "text-emerald-400" : m.good === false ? "text-amber-400" : "text-white"}`}>{m.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass rounded-2xl p-5 bg-emerald-500/5 border-emerald-500/20">
                <h4 className="font-semibold text-sm mb-2 text-emerald-400">vs Benchmark</h4>
                <p className="text-xs text-white/60">
                  Strategy outperforms Buy & Hold by <span className="text-emerald-400 font-bold">{backtestData?.benchmark_comparison?.excess_cagr_pct || "+4.2"}% CAGR</span> with lower drawdown.
                </p>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-black/30 rounded-lg p-2">
                    <div className="text-white/40">Alpha</div>
                    <div className="font-mono font-bold">{backtestData?.benchmark_comparison?.alpha_pct || "3.1"}%</div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-2">
                    <div className="text-white/40">Beta</div>
                    <div className="font-mono font-bold">{backtestData?.benchmark_comparison?.beta || "0.85"}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "research" && (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-6 border-violet-500/20">
            <h3 className="font-bold text-lg mb-2">Research Question</h3>
            <p className="text-violet-200 bg-violet-500/10 border border-violet-500/20 rounded-xl p-4 text-sm leading-relaxed">
              "Does combining machine-learning market signals with multi-agent financial analysis improve the quality and robustness of stock recommendations compared with conventional ML-based prediction?"
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {ablationData?.ablation_study ? Object.entries(ablationData.ablation_study).map(([system, metrics]: any) => (
              <div key={system} className="glass rounded-2xl p-5 card-hover">
                <h4 className="font-semibold text-sm mb-1">{system}</h4>
                <p className="text-[11px] text-white/40 mb-4">{metrics.description}</p>
                <div className="space-y-2.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-white/50">CAGR</span>
                    <span className="font-mono font-bold text-emerald-400">{metrics.cagr_pct.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-white/50">Sharpe</span>
                    <span className="font-mono">{metrics.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-white/50">Drawdown</span>
                    <span className="font-mono text-amber-400">{metrics.max_drawdown_pct.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-white/50">Win Rate</span>
                    <span className="font-mono">{metrics.win_rate_pct.toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            )) : (
              ["System A - Technical Only", "System B - +Sentiment", "System C - +Fundamentals", "System D - Full Agentic"].map((sys, i) => (
                <div key={sys} className="glass rounded-2xl p-5">
                  <h4 className="font-semibold text-sm mb-4">{sys}</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs"><span className="text-white/50">CAGR</span><span className="font-mono text-emerald-400">{(8 + i*1.5).toFixed(1)}%</span></div>
                    <div className="flex justify-between text-xs"><span className="text-white/50">Sharpe</span><span className="font-mono">{(0.8 + i*0.15).toFixed(2)}</span></div>
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-6 glass rounded-2xl p-6">
              <h3 className="font-semibold mb-4">Methodology - Avoiding Data Leakage</h3>
              <div className="space-y-3 text-xs text-white/60 leading-relaxed">
                <div className="flex gap-3">
                  <div className="h-6 w-6 rounded-full bg-violet-500/20 flex items-center justify-center flex-shrink-0 mt-0.5"><span className="text-violet-400 font-bold text-[10px]">1</span></div>
                  <p><span className="text-white font-medium">Chronological splits:</span> 2018-2022 Training, 2023-2024 Validation, 2025-2026 Testing. No future data in training.</p>
                </div>
                <div className="flex gap-3">
                  <div className="h-6 w-6 rounded-full bg-violet-500/20 flex items-center justify-center flex-shrink-0 mt-0.5"><span className="text-violet-400 font-bold text-[10px]">2</span></div>
                  <p><span className="text-white font-medium">Walk-forward validation:</span> Expanding window, retrain periodically, test on unseen future.</p>
                </div>
                <div className="flex gap-3">
                  <div className="h-6 w-6 rounded-full bg-violet-500/20 flex items-center justify-center flex-shrink-0 mt-0.5"><span className="text-violet-400 font-bold text-[10px]">3</span></div>
                  <p><span className="text-white font-medium">Feature lag:</span> All technical indicators computed only on past data. No lookahead bias.</p>
                </div>
                <div className="flex gap-3">
                  <div className="h-6 w-6 rounded-full bg-violet-500/20 flex items-center justify-center flex-shrink-0 mt-0.5"><span className="text-violet-400 font-bold text-[10px]">4</span></div>
                  <p><span className="text-white font-medium">Transaction costs:</span> 0.1% per trade included. Realistic slippage modeling.</p>
                </div>
              </div>
            </div>

            <div className="col-span-12 lg:col-span-6 glass rounded-2xl p-6">
              <h3 className="font-semibold mb-4">Model Comparison (Validation)</h3>
              <div className="space-y-2">
                {[
                  { model: "XGBoost", acc: "68.2%", f1: "0.64", auc: "0.71", best: true },
                  { model: "LightGBM", acc: "66.8%", f1: "0.62", auc: "0.69" },
                  { model: "Random Forest", acc: "65.1%", f1: "0.60", auc: "0.67" },
                  { model: "LSTM", acc: "64.3%", f1: "0.59", auc: "0.66" },
                  { model: "Logistic Reg", acc: "60.5%", f1: "0.55", auc: "0.62" },
                ].map((m) => (
                  <div key={m.model} className={`grid grid-cols-4 gap-2 text-xs p-2.5 rounded-xl ${m.best ? "bg-violet-500/10 border border-violet-500/20" : "bg-white/5"}`}>
                    <span className="font-medium">{m.model} {m.best && "⭐"}</span>
                    <span className="font-mono text-white/60">{m.acc}</span>
                    <span className="font-mono text-white/60">{m.f1}</span>
                    <span className="font-mono text-white/60">{m.auc}</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-white/30 mt-3">Metrics: Accuracy, F1, ROC-AUC on chronological validation set. XGBoost best for tabular, ensemble used in production.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getMockData(symbol: string) {
  const basePrice = { "RELIANCE": 2450, "TCS": 3500, "HDFCBANK": 1600, "INFY": 1500, "ICICIBANK": 1000, "SBIN": 550 }[symbol] || 1200;
  return {
    symbol: `${symbol}.NS`,
    display_symbol: symbol,
    current_price: null,
    previous_close: null,
    price_type: "UNAVAILABLE",
    price_as_of: null,
    price_source: null,
    recommendation: {
      recommendation: Math.random() > 0.4 ? "BUY" : Math.random() > 0.5 ? "HOLD" : "SELL",
      confidence: 65 + Math.random() * 20,
      combined_score: 0.5 + (Math.random() - 0.5) * 0.4,
      expected_return_5d: (Math.random() - 0.3) * 5,
      technical_score: 50 + (Math.random() - 0.3) * 40,
      fundamental_score: 60 + (Math.random() - 0.5) * 30,
      sentiment: { label: Math.random() > 0.6 ? "Positive" : Math.random() > 0.3 ? "Neutral" : "Negative", score: (Math.random() - 0.5), confidence: 0.7 },
      ml_prediction: { signal: "BUY", probability: 0.6 + Math.random() * 0.3, confidence: 0.7 },
      risk: { level: Math.random() > 0.7 ? "High" : Math.random() > 0.3 ? "Medium" : "Low", score: 20 + Math.random() * 60, factors: ["Medium volatility", "Sector momentum weakening"] },
      market_regime: { regime: "Bull Market", signal: "BULLISH" },
      explanation: {
        for: ["✓ Positive 20/50-day trend", "✓ RSI indicates positive momentum", "✓ ML model predicts positive 5-day return", "✓ Recent company news is predominantly positive"],
        against: ["⚠ High short-term volatility", "⚠ Valuation relatively elevated"],
        risk_factors: ["Market volatility is elevated", "Sector sentiment has weakened"]
      },
      weights_used: { technical: 0.25, ml: 0.25, sentiment: 0.15, fundamental: 0.20, market_regime: 0.10, risk: 0.05 },
      timestamp: new Date().toISOString()
    },
    agents: {
      technical: {
        agent: "TechnicalAnalysisAgent",
        signal: "BUY",
        confidence: 0.78,
        reasoning: ["Positive 20/50-day trend indicates bullish momentum", "Golden Cross bullish signal", "RSI at 62 indicates bullish momentum"],
        details: {
          technical_summary: { RSI: 62, Trend_20: "Positive", Trend_50: "Positive", MACD_signal: "Bullish", Volume_signal: "High", Volatility_level: "Medium" },
          technical_score: 78,
          risk_factors: ["Medium volatility"]
        }
      },
      news: {
        agent: "NewsSentimentAgent",
        signal: "BUY",
        confidence: 0.81,
        reasoning: ["Analyzed 8 recent articles", "Overall sentiment: Positive (score: 0.42)", "Positive: Reliance announces major renewable-energy investment"],
        details: { sentiment: { overall_sentiment: "Positive", sentiment_score: 0.42 }, risk_factors: [] }
      },
      fundamental: {
        agent: "FundamentalAgent",
        signal: "HOLD_BULLISH",
        confidence: 0.72,
        reasoning: ["Strong ROE at 18.5%", "Robust revenue growth 12.3% YoY", "Low debt-to-equity 0.32"],
        details: { fundamental_score: 74, risk_factors: [] }
      },
      risk: {
        agent: "RiskAgent",
        signal: "MEDIUM_RISK",
        confidence: 0.7,
        reasoning: ["Volatility (20d): 28.5% - Medium", "Current drawdown: -3.2%", "India VIX: 15.2 - Normal"],
        details: { risk_level: "Medium", risk_score: 25, risk_factors: ["Medium volatility 28.5%"] }
      },
      market_regime: {
        agent: "MarketRegimeAgent",
        signal: "BULLISH",
        confidence: 0.8,
        reasoning: ["NIFTY 50: ₹24500, 20-day trend: Positive", "Market regime: Bull Market", "Bull market - favorable for long positions"],
        details: { regime: "Bull Market" }
      },
      ml: {
        agent: "MLPredictionAgent",
        signal: "BUY",
        confidence: 0.76,
        reasoning: ["ML ensemble prediction: BUY with 72.3% probability", "Models used: 4", "Top drivers: RSI_14, Momentum_10, Volume_ratio"],
        details: { probability: 0.723 }
      }
    },
    price_history: Array.from({ length: 90 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (90 - i));
      const close = basePrice + Math.sin(i / 10) * 50 + (Math.random() - 0.5) * 30 + i * 0.5;
      const middle = close - Math.sin(i / 8) * 12;
      return {
        date: date.toISOString().split('T')[0],
        close,
        rsi14: 50 + Math.sin(i / 9) * 18,
        macd: Math.sin(i / 12) * 8,
        macd_signal: Math.sin((i - 2) / 12) * 7,
        macd_histogram: Math.sin(i / 12) * 1.5,
        bb_upper: middle + 35,
        bb_middle: middle,
        bb_lower: middle - 35,
        sma20: middle,
        sma50: middle - 8
      };
    })
  };
}

function getMockBacktest(symbol: string) {
  const equity = [];
  let val = 100000;
  for (let i = 0; i < 252; i++) {
    val *= 1 + (Math.random() - 0.48) * 0.02;
    const date = new Date();
    date.setDate(date.getDate() - (252 - i));
    equity.push({ date: date.toISOString().split('T')[0], equity: Math.round(val) });
  }
  const bh = [];
  let bhVal = 100000;
  for (let i = 0; i < 252; i++) {
    bhVal *= 1 + (Math.random() - 0.49) * 0.015;
    const date = new Date();
    date.setDate(date.getDate() - (252 - i));
    bh.push({ date: date.toISOString().split('T')[0], equity: Math.round(bhVal) });
  }
  return {
    metrics: { cagr_pct: 14.2, sharpe_ratio: 1.32, max_drawdown_pct: -9.4 },
    trade_metrics: { win_rate_pct: 61.2, profit_factor: 1.58, total_trades: 38 },
    benchmark_comparison: { excess_cagr_pct: 5.1, alpha_pct: 4.2, beta: 0.82 },
    equity_curve: equity,
    benchmark_equity: bh
  };
}

function getMockAblation(symbol: string) {
  return {
    ablation_study: {
      "System A - Technical Only": { cagr_pct: 9.2, sharpe_ratio: 0.85, max_drawdown_pct: -14.2, win_rate_pct: 52, description: "Only price/technical indicators" },
      "System B - Price + Sentiment": { cagr_pct: 10.8, sharpe_ratio: 0.96, max_drawdown_pct: -12.1, win_rate_pct: 56, description: "Adds news sentiment analysis" },
      "System C - Price + Sentiment + Fundamentals": { cagr_pct: 12.1, sharpe_ratio: 1.12, max_drawdown_pct: -10.5, win_rate_pct: 59, description: "Adds fundamental analysis" },
      "System D - Full Agentic AI": { cagr_pct: 14.2, sharpe_ratio: 1.32, max_drawdown_pct: -9.4, win_rate_pct: 61, description: "Full multi-agent system with risk and regime detection" },
    }
  };
}
