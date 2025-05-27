# scraper/pinescript_sources.py

import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
import re
import time
from urllib.parse import urljoin, urlparse, quote_plus
from .parser import extract_code as extract_code_from_html_markdown


class PinescriptSources:
    """Scrape Pinescript-specific sources for trading indicators, strategies, and utilities."""

    def __init__(self, user_agent: str, timeout: int = 15):
        self.user_agent = user_agent
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

        # Pinescript-specific keywords and patterns
        self.pinescript_keywords = {
            'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr', 'adx', 'cci', 'williams'],
            'strategies': ['strategy', 'entry', 'exit', 'long', 'short', 'position', 'profit', 'loss'],
            'price_action': ['candlestick', 'pattern', 'support', 'resistance', 'breakout', 'reversal'],
            'risk_management': ['stop_loss', 'take_profit', 'position_size', 'risk', 'drawdown'],
            'alerts': ['alert', 'notification', 'webhook', 'telegram', 'email'],
            'plotting': ['plot', 'hline', 'bgcolor', 'plotshape', 'plotchar', 'label'],
            'data_analysis': ['volume', 'open_interest', 'correlation', 'momentum', 'volatility'],
            'timeframes': ['daily', 'hourly', '4h', '1h', 'weekly', 'monthly', 'timeframe'],
            'market_structure': ['pivot', 'fibonacci', 'trend', 'channel', 'levels']
        }

        # TradingView specific patterns
        self.tradingview_patterns = [
            r'//@version\s*=\s*[0-9]+',
            r'strategy\s*\(',
            r'indicator\s*\(',
            r'study\s*\(',
            r'plot\s*\(',
            r'ta\.',
            r'math\.',
            r'request\.',
            r'timeframe\.'
        ]

    def fetch_tradingview_public_library(self, query: str, logger) -> List[str]:
        """Fetch scripts from TradingView's public library."""
        logger.info(f"Searching TradingView public library for: {query}")
        snippets = []

        try:
            # TradingView script search - simulated approach since API access is limited
            search_terms = self._expand_pinescript_query(query)

            for term in search_terms[:3]:  # Limit searches
                try:
                    # Note: In practice, you'd need to use web scraping carefully
                    # This is a template for the structure
                    scripts = self._search_tradingview_scripts(term, logger)
                    snippets.extend(scripts)
                    time.sleep(1)  # Be respectful
                except Exception as e:
                    logger.warning(f"Error searching TradingView for '{term}': {e}")

        except Exception as e:
            logger.error(f"Error in TradingView public library search: {e}")

        return snippets

    def fetch_pinescript_examples(self, query: str, logger) -> List[str]:
        """Fetch Pinescript examples from various sources."""
        logger.info(f"Fetching Pinescript examples for: {query}")
        snippets = []

        # Built-in examples for common indicators and strategies
        builtin_examples = self._get_builtin_pinescript_examples(query)
        snippets.extend(builtin_examples)

        # Educational content
        educational_snippets = self._fetch_educational_content(query, logger)
        snippets.extend(educational_snippets)

        return snippets

    def _expand_pinescript_query(self, query: str) -> List[str]:
        """Expand query with Pinescript-specific terms."""
        expanded = [query]
        query_lower = query.lower()

        # Add Pinescript-specific variations
        if any(indicator in query_lower for indicator in self.pinescript_keywords['indicators']):
            expanded.append(f"{query} pinescript")
            expanded.append(f"{query} indicator")
            expanded.append(f"{query} trading strategy")

        if 'strategy' in query_lower:
            expanded.append(f"{query} pinescript strategy")
            expanded.append(f"{query} trading algorithm")

        if 'alert' in query_lower:
            expanded.append(f"{query} pinescript alert")
            expanded.append(f"{query} notification")

        return list(set(expanded))

    def _search_tradingview_scripts(self, term: str, logger) -> List[str]:
        """Search for TradingView scripts (placeholder implementation)."""
        # Note: This would require careful web scraping or API access
        # For now, return built-in examples
        return self._get_builtin_pinescript_examples(term)

    def _get_builtin_pinescript_examples(self, query: str) -> List[str]:
        """Get built-in Pinescript examples based on query."""
        examples = []
        query_lower = query.lower()

        # Simple Moving Average
        if any(term in query_lower for term in ['sma', 'moving average', 'average']):
            examples.append('''
//@version=5
indicator("Simple Moving Average", shorttitle="SMA", overlay=true)

length = input.int(20, title="Length", minval=1)
source = input(close, title="Source")

sma_value = ta.sma(source, length)

plot(sma_value, color=color.blue, linewidth=2, title="SMA")

// Alert conditions
alertcondition(ta.crossover(close, sma_value), title="Price Cross Above SMA")
alertcondition(ta.crossunder(close, sma_value), title="Price Cross Below SMA")
            '''.strip())

        # RSI Indicator
        if any(term in query_lower for term in ['rsi', 'relative strength']):
            examples.append('''
//@version=5
indicator("RSI with Alerts", shorttitle="RSI", format=format.price, precision=2)

length = input.int(14, title="Length", minval=1)
source = input(close, title="Source")
upper_level = input.int(70, title="Overbought Level")
lower_level = input.int(30, title="Oversold Level")

rsi_value = ta.rsi(source, length)

plot(rsi_value, title="RSI", color=color.purple, linewidth=2)
hline(upper_level, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(lower_level, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(50, "Midline", color=color.gray, linestyle=hline.style_dotted)

// Background coloring
bgcolor(rsi_value > upper_level ? color.new(color.red, 90) : na)
bgcolor(rsi_value < lower_level ? color.new(color.green, 90) : na)

// Alerts
alertcondition(ta.crossover(rsi_value, upper_level), title="RSI Overbought")
alertcondition(ta.crossunder(rsi_value, lower_level), title="RSI Oversold")
            '''.strip())

        # Basic Strategy
        if any(term in query_lower for term in ['strategy', 'trading', 'backtest']):
            examples.append('''
//@version=5
strategy("SMA Crossover Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Input parameters
fast_length = input.int(10, title="Fast SMA Length", minval=1)
slow_length = input.int(20, title="Slow SMA Length", minval=1)
use_stop_loss = input.bool(true, title="Use Stop Loss")
stop_loss_pct = input.float(2.0, title="Stop Loss %", minval=0.1, maxval=10.0)

// Calculate SMAs
fast_sma = ta.sma(close, fast_length)
slow_sma = ta.sma(close, slow_length)

// Plot SMAs
plot(fast_sma, color=color.blue, title="Fast SMA")
plot(slow_sma, color=color.red, title="Slow SMA")

// Strategy logic
long_condition = ta.crossover(fast_sma, slow_sma)
short_condition = ta.crossunder(fast_sma, slow_sma)

if long_condition
    strategy.entry("Long", strategy.long)
    if use_stop_loss
        strategy.exit("Long Exit", "Long", stop=close * (1 - stop_loss_pct / 100))

if short_condition
    strategy.close("Long")

// Plot entry signals
plotshape(long_condition, title="Buy Signal", location=location.belowbar, 
          color=color.green, style=shape.labelup, text="BUY")
plotshape(short_condition, title="Sell Signal", location=location.abovebar, 
          color=color.red, style=shape.labeldown, text="SELL")
            '''.strip())

        # MACD Indicator
        if 'macd' in query_lower:
            examples.append('''
//@version=5
indicator("MACD with Divergence", shorttitle="MACD")

// Input parameters
fast_length = input.int(12, title="Fast Length")
slow_length = input.int(26, title="Slow Length")
signal_length = input.int(9, title="Signal Length")
source = input(close, title="Source")

// Calculate MACD
[macd_line, signal_line, histogram] = ta.macd(source, fast_length, slow_length, signal_length)

// Plot MACD
plot(macd_line, color=color.blue, title="MACD Line")
plot(signal_line, color=color.red, title="Signal Line")
plot(histogram, color=histogram >= 0 ? color.green : color.red, style=plot.style_columns, title="Histogram")

hline(0, "Zero Line", color=color.gray, linestyle=hline.style_dashed)

// Divergence detection (simplified)
bullish_div = ta.pivotlow(histogram, 2, 2) and histogram > histogram[4] and low < low[4]
bearish_div = ta.pivothigh(histogram, 2, 2) and histogram < histogram[4] and high > high[4]

plotshape(bullish_div, title="Bullish Divergence", location=location.bottom, 
          color=color.green, style=shape.triangleup, size=size.small)
plotshape(bearish_div, title="Bearish Divergence", location=location.top, 
          color=color.red, style=shape.triangledown, size=size.small)

// Alerts
alertcondition(ta.crossover(macd_line, signal_line), title="MACD Bullish Cross")
alertcondition(ta.crossunder(macd_line, signal_line), title="MACD Bearish Cross")
            '''.strip())

        # Bollinger Bands
        if any(term in query_lower for term in ['bollinger', 'bands', 'bb']):
            examples.append('''
//@version=5
indicator("Bollinger Bands with Squeeze", shorttitle="BB", overlay=true)

length = input.int(20, title="Length", minval=1)
mult = input.float(2.0, title="Multiplier", minval=0.1, step=0.1)
source = input(close, title="Source")

// Calculate Bollinger Bands
basis = ta.sma(source, length)
dev = mult * ta.stdev(source, length)
upper = basis + dev
lower = basis - dev

// Plot bands
plot(basis, color=color.blue, title="Middle Line")
upper_plot = plot(upper, color=color.red, title="Upper Band")
lower_plot = plot(lower, color=color.green, title="Lower Band")
fill(upper_plot, lower_plot, color=color.new(color.blue, 95), title="Band Fill")

// Squeeze detection
bb_range = upper - lower
kc_range = ta.atr(length) * 2
squeeze = bb_range < kc_range

bgcolor(squeeze ? color.new(color.yellow, 90) : na, title="Squeeze Background")

// Price touch alerts
alertcondition(ta.crossover(close, upper), title="Price Touch Upper Band")
alertcondition(ta.crossunder(close, lower), title="Price Touch Lower Band")
alertcondition(squeeze and not squeeze[1], title="Squeeze Started")
alertcondition(not squeeze and squeeze[1], title="Squeeze Ended")
            '''.strip())

        # Volume Profile (simplified)
        if any(term in query_lower for term in ['volume', 'profile', 'vwap']):
            examples.append('''
//@version=5
indicator("Volume Weighted Average Price", shorttitle="VWAP", overlay=true)

// VWAP calculation
vwap_value = ta.vwap(close)

// Plot VWAP
plot(vwap_value, color=color.orange, linewidth=2, title="VWAP")

// Volume-based coloring
vol_color = volume > ta.sma(volume, 20) ? color.green : color.red
plotcandle(open, high, low, close, color=vol_color, wickcolor=vol_color, bordercolor=vol_color)

// VWAP deviation bands
deviation = ta.stdev(close, 20)
upper_band = vwap_value + deviation
lower_band = vwap_value - deviation

plot(upper_band, color=color.new(color.red, 70), title="VWAP Upper")
plot(lower_band, color=color.new(color.green, 70), title="VWAP Lower")

// Alerts
alertcondition(ta.crossover(close, vwap_value), title="Price Above VWAP")
alertcondition(ta.crossunder(close, vwap_value), title="Price Below VWAP")
            '''.strip())

        # Support/Resistance levels
        if any(term in query_lower for term in ['support', 'resistance', 'levels', 'pivot']):
            examples.append('''
//@version=5
indicator("Support & Resistance Levels", shorttitle="S&R", overlay=true)

lookback = input.int(20, title="Lookback Period", minval=5)
strength = input.int(3, title="Pivot Strength", minval=1)

// Calculate pivot points
pivot_high = ta.pivothigh(high, strength, strength)
pivot_low = ta.pivotlow(low, strength, strength)

// Store levels
var float[] resistance_levels = array.new_float()
var float[] support_levels = array.new_float()

// Add new levels
if not na(pivot_high)
    array.push(resistance_levels, pivot_high)
    if array.size(resistance_levels) > 5
        array.shift(resistance_levels)

if not na(pivot_low)
    array.push(support_levels, pivot_low)
    if array.size(support_levels) > 5
        array.shift(support_levels)

// Plot current levels
if array.size(resistance_levels) > 0
    for i = 0 to array.size(resistance_levels) - 1
        level = array.get(resistance_levels, i)
        line.new(bar_index - lookback, level, bar_index, level, 
                 color=color.red, style=line.style_dashed, width=1)

if array.size(support_levels) > 0
    for i = 0 to array.size(support_levels) - 1
        level = array.get(support_levels, i)
        line.new(bar_index - lookback, level, bar_index, level, 
                 color=color.green, style=line.style_dashed, width=1)

// Alerts for level breaks
alertcondition(not na(pivot_high), title="New Resistance Level")
alertcondition(not na(pivot_low), title="New Support Level")
            '''.strip())

        return examples

    def _fetch_educational_content(self, query: str, logger) -> List[str]:
        """Fetch educational Pinescript content."""
        educational_examples = []
        query_lower = query.lower()

        # Basic syntax examples
        if any(term in query_lower for term in ['basic', 'syntax', 'tutorial', 'beginner']):
            educational_examples.append('''
// Pinescript Basics - Variables and Calculations
//@version=5
indicator("Pinescript Basics", overlay=false)

// Input declarations
my_input = input.int(14, title="Period", minval=1, maxval=100)
my_color = input.color(color.blue, title="Line Color")
my_source = input(close, title="Price Source")

// Variable declarations
var float cumulative_volume = 0.0
var int bar_count = 0

// Calculations
simple_average = (high + low + close) / 3
ema_value = ta.ema(my_source, my_input)
price_change = close - close[1]

// Conditional logic
is_green_candle = close > open
is_high_volume = volume > ta.sma(volume, 50)

// Update variables
cumulative_volume := cumulative_volume + volume
bar_count := bar_count + 1

// Plotting
plot(ema_value, color=my_color, title="EMA")
plot(simple_average, color=color.yellow, title="Typical Price")

// Background coloring
bgcolor(is_green_candle and is_high_volume ? color.new(color.green, 90) : na)

// Labels and annotations
if bar_count % 100 == 0
    label.new(bar_index, high, text="Every 100 bars", 
              color=color.white, textcolor=color.black, size=size.small)
            '''.strip())

        # Advanced features
        if any(term in query_lower for term in ['advanced', 'table', 'matrix', 'array']):
            educational_examples.append('''
// Advanced Pinescript Features - Tables and Arrays
//@version=5
indicator("Advanced Features Demo", overlay=true)

// Table for displaying statistics
var table stats_table = table.new(position.top_right, 2, 5, 
                                  bgcolor=color.white, border_width=1)

// Array for storing prices
var float[] price_array = array.new_float()

// Matrix example (if supported)
lookback = input.int(20, "Lookback Period")

// Update arrays
if array.size(price_array) >= lookback
    array.shift(price_array)
array.push(price_array, close)

// Calculate statistics
if array.size(price_array) > 0
    arr_avg = array.avg(price_array)
    arr_max = array.max(price_array)
    arr_min = array.min(price_array)
    arr_stdev = array.stdev(price_array)

    // Update table
    table.cell(stats_table, 0, 0, "Metric", text_color=color.black, bgcolor=color.gray)
    table.cell(stats_table, 1, 0, "Value", text_color=color.black, bgcolor=color.gray)
    table.cell(stats_table, 0, 1, "Average", text_color=color.black)
    table.cell(stats_table, 1, 1, str.tostring(arr_avg, "#.##"), text_color=color.black)
    table.cell(stats_table, 0, 2, "High", text_color=color.black)
    table.cell(stats_table, 1, 2, str.tostring(arr_max, "#.##"), text_color=color.black)
    table.cell(stats_table, 0, 3, "Low", text_color=color.black)
    table.cell(stats_table, 1, 3, str.tostring(arr_min, "#.##"), text_color=color.black)
    table.cell(stats_table, 0, 4, "StdDev", text_color=color.black)
    table.cell(stats_table, 1, 4, str.tostring(arr_stdev, "#.##"), text_color=color.black)

// User-defined functions
f_calculate_range() =>
    high - low

f_is_hammer() =>
    body_size = math.abs(close - open)
    lower_wick = open > close ? close - low : open - low
    upper_wick = open > close ? high - open : high - close
    lower_wick > body_size * 2 and upper_wick < body_size

// Use functions
current_range = f_calculate_range()
is_hammer_pattern = f_is_hammer()

plotshape(is_hammer_pattern, title="Hammer", location=location.belowbar, 
          color=color.blue, style=shape.triangleup)
            '''.strip())

        return educational_examples

    def fetch_pinescript_documentation(self, query: str, logger) -> List[str]:
        """Fetch from Pinescript documentation and reference materials."""
        logger.info(f"Fetching Pinescript documentation for: {query}")
        snippets = []

        try:
            # Documentation examples (these would be scraped from official docs)
            doc_examples = self._get_documentation_examples(query)
            snippets.extend(doc_examples)

        except Exception as e:
            logger.error(f"Error fetching Pinescript documentation: {e}")

        return snippets

    def _get_documentation_examples(self, query: str) -> List[str]:
        """Get documentation examples for specific topics."""
        examples = []
        query_lower = query.lower()

        # Built-in functions documentation
        if any(term in query_lower for term in ['ta.', 'math.', 'request.', 'str.']):
            examples.append('''
// Built-in Functions Reference Examples
//@version=5
indicator("Built-in Functions Demo", overlay=false)

// Technical Analysis functions (ta.*)
sma_20 = ta.sma(close, 20)
ema_12 = ta.ema(close, 12)
rsi_14 = ta.rsi(close, 14)
atr_14 = ta.atr(14)
stoch_k = ta.stoch(close, high, low, 14)

// Math functions (math.*)
price_log = math.log(close)
price_sqrt = math.sqrt(close)
price_abs = math.abs(close - open)
random_value = math.random(0, 100)

// String functions (str.*)
price_string = str.tostring(close, "#.##")
formatted_string = str.format("Price: {0}, Volume: {1}", close, volume)
string_length = str.length(syminfo.ticker)

// Request functions for multi-timeframe
htf_close = request.security(syminfo.tickerid, "1D", close)
htf_volume = request.security(syminfo.tickerid, "4h", volume)

plot(sma_20, title="SMA 20")
plot(rsi_14, title="RSI 14")
            '''.strip())

        return examples

    def fetch_github_pinescript(self, query: str, logger) -> List[str]:
        """Fetch Pinescript code from GitHub repositories."""
        logger.info(f"Searching GitHub for Pinescript: {query}")
        snippets = []

        try:
            # This would use GitHub API to search for .pine files
            # For now, providing examples of what would be found
            github_examples = self._get_github_style_examples(query)
            snippets.extend(github_examples)

        except Exception as e:
            logger.error(f"Error searching GitHub for Pinescript: {e}")

        return snippets

    def _get_github_style_examples(self, query: str) -> List[str]:
        """Get examples that represent typical GitHub Pinescript repositories."""
        examples = []
        query_lower = query.lower()

        if any(term in query_lower for term in ['library', 'function', 'utility']):
            examples.append('''
// @description Library of custom utility functions for Pinescript
//@version=5
library("TradingUtils")

// @function Calculate position size based on risk percentage
// @param capital The total capital
// @param risk_percent The risk percentage (e.g., 1 for 1%)
// @param entry_price The entry price
// @param stop_loss_price The stop loss price
// @returns The position size
export position_size(float capital, float risk_percent, float entry_price, float stop_loss_price) =>
    risk_amount = capital * (risk_percent / 100)
    price_diff = math.abs(entry_price - stop_loss_price)
    risk_amount / price_diff

// @function Check if current bar is a doji candle
// @param doji_threshold The threshold for body size (default 0.1%)
// @returns True if current bar is a doji
export is_doji(float doji_threshold = 0.1) =>
    body_size = math.abs(close - open)
    total_range = high - low
    total_range > 0 ? (body_size / total_range) <= (doji_threshold / 100) : false

// @function Calculate dynamic support/resistance levels
// @param lookback Number of bars to look back
// @param min_touches Minimum number of touches to confirm level
// @returns [support_level, resistance_level]
export get_sr_levels(int lookback = 50, int min_touches = 3) =>
    // Simplified implementation
    recent_high = ta.highest(high, lookback)
    recent_low = ta.lowest(low, lookback)
    [recent_low, recent_high]
            '''.strip())

        return examples


def search_pinescript_sources(query: str, logger) -> List[str]:
    """Main interface for Pinescript source searching."""
    try:
        from config import USER_AGENT, DEFAULT_REQUEST_TIMEOUT
    except ImportError:
        USER_AGENT = "RAGContentScraper/1.0"
        DEFAULT_REQUEST_TIMEOUT = 15

    pinescript_sources = PinescriptSources(USER_AGENT, DEFAULT_REQUEST_TIMEOUT)
    all_snippets = []

    # Fetch from different Pinescript sources
    source_methods = [
        pinescript_sources.fetch_pinescript_examples,
        pinescript_sources.fetch_tradingview_public_library,
        pinescript_sources.fetch_pinescript_documentation,
        pinescript_sources.fetch_github_pinescript
    ]

    for method in source_methods:
        try:
            snippets = method(query, logger)
            all_snippets.extend(snippets)
        except Exception as e:
            logger.error(f"Error in {method.__name__}: {e}")

    logger.info(f"Total Pinescript snippets found: {len(all_snippets)}")
    return all_snippets