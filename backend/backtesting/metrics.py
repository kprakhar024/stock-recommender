"""
Backtesting Metrics - Research Grade
CAGR, Sharpe, Max Drawdown, Win Rate, Profit Factor, etc.
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class BacktestMetrics:
    @staticmethod
    def calculate_returns_metrics(equity_curve: pd.Series) -> Dict:
        """Calculate comprehensive metrics from equity curve"""
        if len(equity_curve) < 2:
            return {"error": "Insufficient data"}
        
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return {"error": "No returns"}
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        n_days = len(equity_curve)
        n_years = n_days / 252
        
        # CAGR
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(0.1, n_years)) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe (assuming 6% risk-free for India)
        risk_free = 0.06
        excess_returns = returns - risk_free / 252
        sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
        sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        max_drawdown_duration = BacktestMetrics._max_drawdown_duration(drawdown)
        
        # Calmar Ratio
        calmar = (cagr * 100 / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Win Rate etc will be calculated from trades
        return {
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr * 100, 2),
            "volatility_pct": round(volatility, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "max_drawdown_duration_days": max_drawdown_duration,
            "calmar_ratio": round(calmar, 3),
            "best_day_pct": round(returns.max() * 100, 2),
            "worst_day_pct": round(returns.min() * 100, 2),
            "positive_days_pct": round((returns > 0).mean() * 100, 1),
            "avg_daily_return_pct": round(returns.mean() * 100, 3),
            "equity_start": round(equity_curve.iloc[0], 2),
            "equity_end": round(equity_curve.iloc[-1], 2)
        }
    
    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate max drawdown duration in days"""
        # Find periods where drawdown < 0
        is_drawdown = drawdown < 0
        # Group consecutive drawdown periods
        groups = (is_drawdown != is_drawdown.shift()).cumsum()
        durations = is_drawdown.groupby(groups).sum()
        return int(durations.max()) if len(durations) > 0 else 0
    
    @staticmethod
    def calculate_trade_metrics(trades: List[Dict]) -> Dict:
        """Calculate metrics from trade list"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0,
                "avg_win_pct": 0,
                "avg_loss_pct": 0,
                "profit_factor": 0,
                "avg_trade_pct": 0
            }
        
        returns = [t.get('return_pct', 0) for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
        
        avg_trade = np.mean(returns) if returns else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 3) if profit_factor != float('inf') else 999,
            "avg_trade_pct": round(avg_trade, 2),
            "expectancy_pct": round(expectancy, 3),
            "gross_profit_pct": round(gross_profit, 2),
            "gross_loss_pct": round(gross_loss, 2),
            "largest_win_pct": round(max(returns) if returns else 0, 2),
            "largest_loss_pct": round(min(returns) if returns else 0, 2)
        }
    
    @staticmethod
    def compare_to_benchmark(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> Dict:
        """Compare strategy vs benchmark"""
        if len(strategy_equity) != len(benchmark_equity):
            # Align
            min_len = min(len(strategy_equity), len(benchmark_equity))
            strategy_equity = strategy_equity.iloc[-min_len:]
            benchmark_equity = benchmark_equity.iloc[-min_len:]
        
        strategy_metrics = BacktestMetrics.calculate_returns_metrics(strategy_equity)
        benchmark_metrics = BacktestMetrics.calculate_returns_metrics(benchmark_equity)
        
        # Excess return
        excess_cagr = strategy_metrics.get('cagr_pct', 0) - benchmark_metrics.get('cagr_pct', 0)
        
        # Correlation
        strat_returns = strategy_equity.pct_change().dropna()
        bench_returns = benchmark_equity.pct_change().dropna()
        if len(strat_returns) == len(bench_returns) and len(strat_returns) > 5:
            correlation = strat_returns.corr(bench_returns)
            # Beta
            covariance = np.cov(strat_returns, bench_returns)[0][1]
            benchmark_variance = np.var(bench_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            # Alpha (annualized)
            alpha = (strat_returns.mean() - beta * bench_returns.mean()) * 252 * 100
        else:
            correlation = 0
            beta = 1
            alpha = 0
        
        return {
            "strategy": strategy_metrics,
            "benchmark": benchmark_metrics,
            "excess_cagr_pct": round(excess_cagr, 2),
            "correlation": round(correlation, 3),
            "beta": round(beta, 3),
            "alpha_pct": round(alpha, 2),
            "outperforms": excess_cagr > 0
        }
