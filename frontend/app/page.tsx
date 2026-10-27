"use client";

import { useState, useEffect } from "react";
import { Search, TrendingUp, TrendingDown, Minus, BarChart3, Brain, Newspaper, Shield, Globe, Zap } from "lucide-react";
import StockDashboard from "../components/StockDashboard";

const NIFTY_50 = [
  "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC",
  "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE", "HCLTECH",
  "WIPRO", "SUNPHARMA", "ULTRACEMCO", "TITAN", "POWERGRID", "NTPC", "ONGC", "TATASTEEL"
];

export default function Home() {
  const [selectedStock, setSelectedStock] = useState("RELIANCE");
  const [searchQuery, setSearchQuery] = useState("");
  const [marketRegime, setMarketRegime] = useState<any>(null);
  const [isLoadingRegime, setIsLoadingRegime] = useState(true);

  useEffect(() => {
    fetchMarketRegime();
  }, []);

  const fetchMarketRegime = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/market/regime");
      if (res.ok) {
        const data = await res.json();
        setMarketRegime(data);
      }
    } catch (e) {
      console.log("Regime fetch failed, using mock");
      setMarketRegime({
        details: { regime: "Bull Market", nifty_close: 24500, vix: 14.5 },
        reasoning: ["NIFTY in bullish trend", "VIX normal at 14.5"]
      });
    } finally {
      setIsLoadingRegime(false);
    }
  };

  const filteredStocks = NIFTY_50.filter(s => 
    s.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-black/50 backdrop-blur-xl">
        <div className="mx-auto max-w-[1600px] px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-600 to-emerald-500 flex items-center justify-center">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">AI Stock Recommender</h1>
              <p className="text-xs text-white/50">NSE/BSE • Multi-Agent • Research Grade</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-6">
            <div className="flex items-center gap-2 text-sm">
              <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-white/70">NIFTY 50</span>
              <span className="font-mono font-medium">
                {marketRegime?.details?.nifty_close ? `₹${marketRegime.details.nifty_close.toFixed(0)}` : "24,500"}
              </span>
            </div>
            <div className="h-6 w-px bg-white/10" />
            <div className="flex items-center gap-2 text-sm">
              <Globe className="h-4 w-4 text-white/50" />
              <span className="text-white/70">{marketRegime?.details?.regime || "Bull Market"}</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                marketRegime?.details?.regime?.includes("Bull") ? "bg-emerald-500/20 text-emerald-400" :
                marketRegime?.details?.regime?.includes("Bear") ? "bg-red-500/20 text-red-400" :
                "bg-yellow-500/20 text-yellow-400"
              }`}>
                {marketRegime?.details?.regime || "BULLISH"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] px-6 py-8 grid grid-cols-12 gap-8">
        {/* Sidebar - Stock Selector */}
        <aside className="col-span-12 lg:col-span-3 space-y-6">
          <div className="glass rounded-2xl p-5">
            <h2 className="font-semibold mb-4 flex items-center gap-2">
              <Search className="h-4 w-4" /> Indian Stocks
            </h2>
            
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/40" />
              <input
                type="text"
                placeholder="Search RELIANCE, TCS..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-violet-500/50 focus:bg-white/10 transition-all"
              />
            </div>

            <div className="space-y-1 max-h-[400px] overflow-y-auto pr-1">
              {filteredStocks.map((stock) => (
                <button
                  key={stock}
                  onClick={() => setSelectedStock(stock)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl text-sm font-medium transition-all flex items-center justify-between group ${
                    selectedStock === stock
                      ? "bg-violet-600 text-white shadow-lg shadow-violet-600/20"
                      : "hover:bg-white/10 text-white/70 hover:text-white"
                  }`}
                >
                  <span>{stock}</span>
                  <span className={`text-xs ${selectedStock === stock ? "text-white/70" : "text-white/30 group-hover:text-white/50"}`}>
                    .NS
                  </span>
                </button>
              ))}
            </div>

            <div className="mt-6 pt-6 border-t border-white/10">
              <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-3">System Architecture</h3>
              <div className="space-y-2.5">
                {[
                  { icon: BarChart3, label: "Technical Agent", desc: "RSI, MACD, SMA" },
                  { icon: Brain, label: "ML Agent", desc: "XGBoost, LSTM" },
                  { icon: Newspaper, label: "News Agent", desc: "FinBERT Sentiment" },
                  { icon: Zap, label: "Fundamental", desc: "P/E, ROE, Growth" },
                  { icon: Shield, label: "Risk Agent", desc: "Volatility, VIX" },
                  { icon: Globe, label: "Market Regime", desc: "Bull/Bear Detection" },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-3 text-xs">
                    <div className="h-7 w-7 rounded-lg bg-white/5 flex items-center justify-center">
                      <item.icon className="h-3.5 w-3.5 text-white/60" />
                    </div>
                    <div>
                      <div className="font-medium text-white/80">{item.label}</div>
                      <div className="text-white/40 text-[11px]">{item.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="glass rounded-2xl p-5 bg-gradient-to-br from-violet-600/20 to-emerald-600/20 border-violet-500/20">
            <h3 className="font-semibold text-sm mb-2">Research Grade</h3>
            <p className="text-xs text-white/60 leading-relaxed">
              Chronological splits, walk-forward validation, no data leakage. Backtested vs Buy & Hold and NIFTY 50. Ablation study included.
            </p>
            <div className="mt-3 flex gap-2">
              <span className="text-[10px] px-2 py-1 rounded-full bg-white/10">Sharpe</span>
              <span className="text-[10px] px-2 py-1 rounded-full bg-white/10">CAGR</span>
              <span className="text-[10px] px-2 py-1 rounded-full bg-white/10">Drawdown</span>
            </div>
          </div>
        </aside>

        {/* Main Dashboard */}
        <div className="col-span-12 lg:col-span-9">
          <StockDashboard symbol={selectedStock} />
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-12 py-8">
        <div className="mx-auto max-w-[1600px] px-6 flex flex-col md:flex-row justify-between gap-4 text-xs text-white/40">
          <div>
            <p className="font-medium text-white/60">AI-Powered Indian Stock Recommendation System</p>
            <p className="mt-1">Research Project • Data: NSE/BSE via yfinance • Models: XGBoost, LSTM, Random Forest • NLP: FinBERT/VADER</p>
          </div>
          <div className="text-right">
            <p>⚠️ For research & educational purposes only</p>
            <p>Not financial advice • Backtest past performance ≠ future results</p>
          </div>
        </div>
      </footer>
    </main>
  );
}
