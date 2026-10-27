"""
Technical Indicators - Research Grade
All indicators computed without lookahead bias
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators - no data leakage"""
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)  # Risk veto: NIFTY < EMA_200
        
        # RSI
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # ATR - volatility
        df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
        
        # Momentum
        df['Momentum_10'] = ta.momentum.roc(df['Close'], window=10)
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volatility
        df['Volatility_10'] = df['Returns'].rolling(10).std() * np.sqrt(252)
        df['Volatility_30'] = df['Returns'].rolling(30).std() * np.sqrt(252)
        
        # Trend signals
        df['Trend_20'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Trend_50'] = np.where(df['Close'] > df['SMA_50'], 1, -1)
        df['Trend_200'] = np.where(df['Close'] > df['EMA_200'], 1, -1)  # long-term regime
        df['Golden_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
        df['Death_Cross'] = np.where(df['SMA_20'] < df['SMA_50'], 1, 0)

        # Rolling std of returns — used in dynamic volatility threshold
        # Threshold_t = max(0.015, 1.0 * rolling_std_20 / close * sqrt(5))
        # NOTE: This is std of *daily returns* (not annualised) — used raw in features.py
        df['Rolling_Std_20'] = df['Returns'].rolling(20).std()
        
        # Returns over different windows (features for ML)
        for window in [1, 5, 10, 20]:
            df[f'Return_{window}d'] = df['Close'].pct_change(window)
        
        # Price position
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Price_position_20'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        return df
    
    @staticmethod
    def get_technical_summary(df: pd.DataFrame) -> Dict:
        """Get latest technical snapshot for agent"""
        if len(df) < 50:
            return {"error": "Insufficient data"}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # RSI interpretation
        rsi = latest['RSI_14']
        if rsi > 70:
            rsi_signal = "Overbought"
        elif rsi < 30:
            rsi_signal = "Oversold"
        elif rsi > 60:
            rsi_signal = "Bullish momentum"
        elif rsi < 40:
            rsi_signal = "Bearish momentum"
        else:
            rsi_signal = "Neutral"
        
        # MACD
        macd_signal = "Bullish" if latest['MACD_histogram'] > 0 else "Bearish"
        macd_trend = "Positive" if latest['MACD_histogram'] > prev['MACD_histogram'] else "Weakening"
        
        # Trend
        trend_20 = "Positive" if latest['Close'] > latest['SMA_20'] else "Negative"
        trend_50 = "Positive" if latest['Close'] > latest['SMA_50'] else "Negative"
        
        # Bollinger
        bb_pos = latest['BB_position']
        if bb_pos > 0.8:
            bb_signal = "Near upper band - overbought risk"
        elif bb_pos < 0.2:
            bb_signal = "Near lower band - oversold bounce potential"
        else:
            bb_signal = "Within bands - normal"
        
        # Volume
        vol_signal = "High" if latest['Volume_ratio'] > 1.5 else "Normal" if latest['Volume_ratio'] > 0.8 else "Low"
        
        # Volatility
        vol = latest['Volatility_20']
        if vol > 0.35:
            vol_level = "High"
        elif vol > 0.25:
            vol_level = "Medium"
        else:
            vol_level = "Low"
        
        return {
            "RSI": round(rsi, 2),
            "RSI_signal": rsi_signal,
            "MACD": round(latest['MACD_histogram'], 4),
            "MACD_signal": macd_signal,
            "MACD_trend": macd_trend,
            "SMA_20": round(latest['SMA_20'], 2),
            "SMA_50": round(latest['SMA_50'], 2),
            "EMA_200": round(latest['EMA_200'], 2) if not pd.isna(latest.get('EMA_200', float('nan'))) else None,
            "Trend_20": trend_20,
            "Trend_50": trend_50,
            "Trend_200": "Positive" if latest.get('Trend_200', -1) == 1 else "Negative",
            "Golden_Cross": bool(latest['Golden_Cross']),
            "BB_position": round(bb_pos, 3),
            "BB_signal": bb_signal,
            "BB_width": round(latest['BB_width'], 4),
            "ATR": round(latest['ATR_14'], 2),
            "Volume_ratio": round(latest['Volume_ratio'], 2),
            "Volume_signal": vol_signal,
            "Momentum_10": round(latest['Momentum_10'], 2),
            "Volatility_20": round(vol, 4),
            "Volatility_level": vol_level,
            "Price_position_20": round(latest['Price_position_20'], 3),
            "Rolling_Std_20": round(latest['Rolling_Std_20'], 6) if not pd.isna(latest.get('Rolling_Std_20', float('nan'))) else None,
            "Close": round(latest['Close'], 2)
        }
    
    @staticmethod
    def compute_technical_score(summary: Dict) -> float:
        """Compute 0-100 technical score - transparent methodology"""
        score = 50  # neutral baseline
        
        # RSI component (20%)
        rsi = summary.get('RSI', 50)
        if 40 <= rsi <= 70:
            score += 10 if 50 <= rsi <= 65 else 5
        elif rsi < 30:
            score += 5  # oversold bounce potential
        elif rsi > 75:
            score -= 10
        
        # Trend component (30%)
        if summary.get('Trend_20') == "Positive":
            score += 10
        else:
            score -= 10
        
        if summary.get('Trend_50') == "Positive":
            score += 10
        else:
            score -= 10
        
        if summary.get('Golden_Cross'):
            score += 10
        
        # MACD component (20%)
        if summary.get('MACD_signal') == "Bullish":
            score += 10
        else:
            score -= 5
        
        if summary.get('MACD_trend') == "Positive":
            score += 5
        
        # Momentum (15%)
        mom = summary.get('Momentum_10', 0)
        if mom > 2:
            score += 8
        elif mom > 0:
            score += 4
        elif mom < -2:
            score -= 8
        
        # Volume (15%)
        vol_ratio = summary.get('Volume_ratio', 1)
        if vol_ratio > 1.2 and summary.get('Trend_20') == "Positive":
            score += 7  # bullish with volume confirmation
        
        return max(0, min(100, score))
