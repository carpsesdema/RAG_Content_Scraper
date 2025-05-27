import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
import re
import time
from urllib.parse import urljoin, urlparse, quote_plus
from .parser import extract_code as extract_code_from_html_markdown
import os
from github import Github, GithubException, RateLimitExceededException


class PinescriptSources:
    def __init__(self, user_agent: str, timeout: int = 15):
        self.user_agent = user_agent
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
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

    def _search_tradingview_scripts(self, term: str, logger) -> List[str]:
        logger.info(f"Attempting to search TradingView for: {term}")
        snippets = []
        script_page_urls_from_search = []

        if not script_page_urls_from_search:
            logger.warning(
                f"Live TradingView search for '{term}' is not fully implemented or yielded no direct script URLs. Using built-in examples for this term.")
            return self._get_builtin_pinescript_examples(term)

        for script_url in script_page_urls_from_search[:3]:
            try:
                logger.debug(f"Fetching TradingView script page: {script_url}")
                response = self.session.get(script_url, timeout=self.timeout)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                script_content_element = soup.find("div", class_="tv-chart-view__script-text")

                if script_content_element:
                    code = script_content_element.get_text().strip()
                    if code and any(pattern in code for pattern in self.tradingview_patterns):
                        snippets.append(code)
                        logger.info(f"Extracted script from {script_url}")
                else:
                    logger.warning(f"Could not find script content element on {script_url}")

                time.sleep(getattr(self, 'TRADINGVIEW_DELAY_BETWEEN_REQUESTS', 1.0))

            except requests.RequestException as e:
                logger.error(f"Error fetching TradingView script page {script_url}: {e}")
            except Exception as e:
                logger.error(f"Error processing TradingView script page {script_url}: {e}", exc_info=True)

        logger.info(f"Found {len(snippets)} snippets from conceptual TradingView search for '{term}'")
        return snippets

    def fetch_tradingview_public_library(self, query: str, logger) -> List[str]:
        logger.info(f"Searching TradingView public library for: {query}")
        all_tv_snippets = []
        search_terms = self._expand_pinescript_query(query)

        for term in search_terms[:2]:
            try:
                scripts = self._search_tradingview_scripts(term, logger)
                all_tv_snippets.extend(scripts)
                if len(all_tv_snippets) >= getattr(self, 'TRADINGVIEW_MAX_SCRIPTS', 5):
                    break
                time.sleep(getattr(self, 'TRADINGVIEW_DELAY_BETWEEN_REQUESTS', 1.0))
            except Exception as e:
                logger.warning(f"Error during TradingView search for '{term}': {e}")

        if not all_tv_snippets or len(all_tv_snippets) < 3:
            logger.info(f"Live TradingView search yielded few/no results for '{query}', adding built-in examples.")
            all_tv_snippets.extend(self._get_builtin_pinescript_examples(query))
            all_tv_snippets = list(dict.fromkeys(all_tv_snippets))

        return all_tv_snippets[:getattr(self, 'TRADINGVIEW_MAX_SCRIPTS', 10)]

    def fetch_pinescript_examples(self, query: str, logger) -> List[str]:
        logger.info(f"Fetching Pinescript examples for: {query}")
        snippets = []
        builtin_examples = self._get_builtin_pinescript_examples(query)
        snippets.extend(builtin_examples)
        educational_snippets = self._fetch_educational_content(query, logger)
        snippets.extend(educational_snippets)
        return snippets

    def _expand_pinescript_query(self, query: str) -> List[str]:
        expanded = [query]
        query_lower = query.lower()
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

    def _get_builtin_pinescript_examples(self, query: str) -> List[str]:
        examples = []
        query_lower = query.lower()
        if any(term in query_lower for term in ['sma', 'moving average', 'average']):
            examples.append('''
//@version=5
indicator("Simple Moving Average", shorttitle="SMA", overlay=true)
length = input.int(20, title="Length", minval=1)
source = input(close, title="Source")
sma_value = ta.sma(source, length)
plot(sma_value, color=color.blue, linewidth=2, title="SMA")
alertcondition(ta.crossover(close, sma_value), title="Price Cross Above SMA")
alertcondition(ta.crossunder(close, sma_value), title="Price Cross Below SMA")
            '''.strip())
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
bgcolor(rsi_value > upper_level ? color.new(color.red, 90) : na)
bgcolor(rsi_value < lower_level ? color.new(color.green, 90) : na)
alertcondition(ta.crossover(rsi_value, upper_level), title="RSI Overbought")
alertcondition(ta.crossunder(rsi_value, lower_level), title="RSI Oversold")
            '''.strip())
        if any(term in query_lower for term in ['strategy', 'trading', 'backtest']):
            examples.append('''
//@version=5
strategy("SMA Crossover Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)
fast_length = input.int(10, title="Fast SMA Length", minval=1)
slow_length = input.int(20, title="Slow SMA Length", minval=1)
use_stop_loss = input.bool(true, title="Use Stop Loss")
stop_loss_pct = input.float(2.0, title="Stop Loss %", minval=0.1, maxval=10.0)
fast_sma = ta.sma(close, fast_length)
slow_sma = ta.sma(close, slow_length)
plot(fast_sma, color=color.blue, title="Fast SMA")
plot(slow_sma, color=color.red, title="Slow SMA")
long_condition = ta.crossover(fast_sma, slow_sma)
short_condition = ta.crossunder(fast_sma, slow_sma)
if long_condition
    strategy.entry("Long", strategy.long)
    if use_stop_loss
        strategy.exit("Long Exit", "Long", stop=close * (1 - stop_loss_pct / 100))
if short_condition
    strategy.close("Long")
plotshape(long_condition, title="Buy Signal", location=location.belowbar, 
          color=color.green, style=shape.labelup, text="BUY")
plotshape(short_condition, title="Sell Signal", location=location.abovebar, 
          color=color.red, style=shape.labeldown, text="SELL")
            '''.strip())
        if 'macd' in query_lower:
            examples.append('''
//@version=5
indicator("MACD with Divergence", shorttitle="MACD")
fast_length = input.int(12, title="Fast Length")
slow_length = input.int(26, title="Slow Length")
signal_length = input.int(9, title="Signal Length")
source = input(close, title="Source")
[macd_line, signal_line, histogram] = ta.macd(source, fast_length, slow_length, signal_length)
plot(macd_line, color=color.blue, title="MACD Line")
plot(signal_line, color=color.red, title="Signal Line")
plot(histogram, color=histogram >= 0 ? color.green : color.red, style=plot.style_columns, title="Histogram")
hline(0, "Zero Line", color=color.gray, linestyle=hline.style_dashed)
bullish_div = ta.pivotlow(histogram, 2, 2) and histogram > histogram[4] and low < low[4]
bearish_div = ta.pivothigh(histogram, 2, 2) and histogram < histogram[4] and high > high[4]
plotshape(bullish_div, title="Bullish Divergence", location=location.bottom, 
          color=color.green, style=shape.triangleup, size=size.small)
plotshape(bearish_div, title="Bearish Divergence", location=location.top, 
          color=color.red, style=shape.triangledown, size=size.small)
alertcondition(ta.crossover(macd_line, signal_line), title="MACD Bullish Cross")
alertcondition(ta.crossunder(macd_line, signal_line), title="MACD Bearish Cross")
            '''.strip())
        if any(term in query_lower for term in ['bollinger', 'bands', 'bb']):
            examples.append('''
//@version=5
indicator("Bollinger Bands with Squeeze", shorttitle="BB", overlay=true)
length = input.int(20, title="Length", minval=1)
mult = input.float(2.0, title="Multiplier", minval=0.1, step=0.1)
source = input(close, title="Source")
basis = ta.sma(source, length)
dev = mult * ta.stdev(source, length)
upper = basis + dev
lower = basis - dev
plot(basis, color=color.blue, title="Middle Line")
upper_plot = plot(upper, color=color.red, title="Upper Band")
lower_plot = plot(lower, color=color.green, title="Lower Band")
fill(upper_plot, lower_plot, color=color.new(color.blue, 95), title="Band Fill")
bb_range = upper - lower
kc_range = ta.atr(length) * 2
squeeze = bb_range < kc_range
bgcolor(squeeze ? color.new(color.yellow, 90) : na, title="Squeeze Background")
alertcondition(ta.crossover(close, upper), title="Price Touch Upper Band")
alertcondition(ta.crossunder(close, lower), title="Price Touch Lower Band")
alertcondition(squeeze and not squeeze[1], title="Squeeze Started")
alertcondition(not squeeze and squeeze[1], title="Squeeze Ended")
            '''.strip())
        if any(term in query_lower for term in ['volume', 'profile', 'vwap']):
            examples.append('''
//@version=5
indicator("Volume Weighted Average Price", shorttitle="VWAP", overlay=true)
vwap_value = ta.vwap(close)
plot(vwap_value, color=color.orange, linewidth=2, title="VWAP")
vol_color = volume > ta.sma(volume, 20) ? color.green : color.red
plotcandle(open, high, low, close, color=vol_color, wickcolor=vol_color, bordercolor=vol_color)
deviation = ta.stdev(close, 20)
upper_band = vwap_value + deviation
lower_band = vwap_value - deviation
plot(upper_band, color=color.new(color.red, 70), title="VWAP Upper")
plot(lower_band, color=color.new(color.green, 70), title="VWAP Lower")
alertcondition(ta.crossover(close, vwap_value), title="Price Above VWAP")
alertcondition(ta.crossunder(close, vwap_value), title="Price Below VWAP")
            '''.strip())
        if any(term in query_lower for term in ['support', 'resistance', 'levels', 'pivot']):
            examples.append('''
//@version=5
indicator("Support & Resistance Levels", shorttitle="S&R", overlay=true)
lookback = input.int(20, title="Lookback Period", minval=5)
strength = input.int(3, title="Pivot Strength", minval=1)
pivot_high = ta.pivothigh(high, strength, strength)
pivot_low = ta.pivotlow(low, strength, strength)
var float[] resistance_levels = array.new_float()
var float[] support_levels = array.new_float()
if not na(pivot_high)
    array.push(resistance_levels, pivot_high)
    if array.size(resistance_levels) > 5
        array.shift(resistance_levels)
if not na(pivot_low)
    array.push(support_levels, pivot_low)
    if array.size(support_levels) > 5
        array.shift(support_levels)
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
alertcondition(not na(pivot_high), title="New Resistance Level")
alertcondition(not na(pivot_low), title="New Support Level")
            '''.strip())
        return examples

    def _fetch_educational_content(self, query: str, logger) -> List[str]:
        educational_examples = []
        query_lower = query.lower()
        if any(term in query_lower for term in ['basic', 'syntax', 'tutorial', 'beginner']):
            educational_examples.append('''
//@version=5
indicator("Pinescript Basics", overlay=false)
my_input = input.int(14, title="Period", minval=1, maxval=100)
my_color = input.color(color.blue, title="Line Color")
my_source = input(close, title="Price Source")
var float cumulative_volume = 0.0
var int bar_count = 0
simple_average = (high + low + close) / 3
ema_value = ta.ema(my_source, my_input)
price_change = close - close[1]
is_green_candle = close > open
is_high_volume = volume > ta.sma(volume, 50)
cumulative_volume := cumulative_volume + volume
bar_count := bar_count + 1
plot(ema_value, color=my_color, title="EMA")
plot(simple_average, color=color.yellow, title="Typical Price")
bgcolor(is_green_candle and is_high_volume ? color.new(color.green, 90) : na)
if bar_count % 100 == 0
    label.new(bar_index, high, text="Every 100 bars", 
              color=color.white, textcolor=color.black, size=size.small)
            '''.strip())
        if any(term in query_lower for term in ['advanced', 'table', 'matrix', 'array']):
            educational_examples.append('''
//@version=5
indicator("Advanced Features Demo", overlay=true)
var table stats_table = table.new(position.top_right, 2, 5, 
                                  bgcolor=color.white, border_width=1)
var float[] price_array = array.new_float()
lookback = input.int(20, "Lookback Period")
if array.size(price_array) >= lookback
    array.shift(price_array)
array.push(price_array, close)
if array.size(price_array) > 0
    arr_avg = array.avg(price_array)
    arr_max = array.max(price_array)
    arr_min = array.min(price_array)
    arr_stdev = array.stdev(price_array)
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
f_calculate_range() =>
    high - low
f_is_hammer() =>
    body_size = math.abs(close - open)
    lower_wick = open > close ? close - low : open - low
    upper_wick = open > close ? high - open : high - close
    lower_wick > body_size * 2 and upper_wick < body_size
current_range = f_calculate_range()
is_hammer_pattern = f_is_hammer()
plotshape(is_hammer_pattern, title="Hammer", location=location.belowbar, 
          color=color.blue, style=shape.triangleup)
            '''.strip())
        return educational_examples

    def fetch_pinescript_documentation(self, query: str, logger) -> List[str]:
        logger.info(f"Fetching Pinescript documentation for: {query}")
        snippets = []
        try:
            doc_examples = self._get_documentation_examples(query)
            snippets.extend(doc_examples)
        except Exception as e:
            logger.error(f"Error fetching Pinescript documentation: {e}")
        return snippets

    def _get_documentation_examples(self, query: str) -> List[str]:
        examples = []
        query_lower = query.lower()
        if any(term in query_lower for term in ['ta.', 'math.', 'request.', 'str.']):
            examples.append('''
//@version=5
indicator("Built-in Functions Demo", overlay=false)
sma_20 = ta.sma(close, 20)
ema_12 = ta.ema(close, 12)
rsi_14 = ta.rsi(close, 14)
atr_14 = ta.atr(14)
stoch_k = ta.stoch(close, high, low, 14)
price_log = math.log(close)
price_sqrt = math.sqrt(close)
price_abs = math.abs(close - open)
random_value = math.random(0, 100)
price_string = str.tostring(close, "#.##")
formatted_string = str.format("Price: {0}, Volume: {1}", close, volume)
string_length = str.length(syminfo.ticker)
htf_close = request.security(syminfo.tickerid, "1D", close)
htf_volume = request.security(syminfo.tickerid, "4h", volume)
plot(sma_20, title="SMA 20")
plot(rsi_14, title="RSI 14")
            '''.strip())
        return examples

    def fetch_github_pinescript(self, query: str, logger) -> List[str]:
        logger.info(f"Searching GitHub for Pinescript files related to: {query}")
        snippets = []
        try:
            from config import (GITHUB_FILES_MAX_REPOS, GITHUB_FILES_PER_REPO_TARGET,
                                GITHUB_FILE_DOWNLOAD_TIMEOUT, USER_AGENT, GITHUB_MAX_FILE_SIZE_KB)
        except ImportError:
            logger.warning("Could not import GitHub config settings, using defaults.")
            GITHUB_FILES_MAX_REPOS = 5
            GITHUB_FILES_PER_REPO_TARGET = 3
            GITHUB_FILE_DOWNLOAD_TIMEOUT = 20
            GITHUB_MAX_FILE_SIZE_KB = 500

        token = os.getenv('GITHUB_TOKEN')
        if not token:
            logger.warning(
                "GITHUB_TOKEN environment variable not found. GitHub API requests will be severely rate-limited.")

        try:
            gh = Github(login_or_token=token, timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT, user_agent=self.user_agent)
            github_search_query = f"{query} language:Pine"
            logger.debug(f"GitHub API search query: {github_search_query}")

            desired_snippets_count = GITHUB_FILES_MAX_REPOS * GITHUB_FILES_PER_REPO_TARGET
            results = gh.search_code(query=github_search_query, sort="indexed", order="desc")

            files_processed_count = 0
            for file_item in results:
                if files_processed_count >= desired_snippets_count:
                    logger.info(f"Reached target snippet count ({desired_snippets_count}) from GitHub.")
                    break

                if not file_item.name.endswith(".pine"):
                    continue

                logger.debug(
                    f"Processing GitHub file: {file_item.repository.full_name}/{file_item.path} (Size: {file_item.size})")

                if file_item.size > (GITHUB_MAX_FILE_SIZE_KB * 1024):
                    logger.warning(f"Skipping large file: {file_item.path} ({file_item.size} bytes)")
                    continue

                try:
                    decoded_content = file_item.decoded_content.decode("utf-8", errors="ignore")
                    if decoded_content and any(pattern in decoded_content for pattern in self.tradingview_patterns):
                        snippets.append(decoded_content.strip())
                        files_processed_count += 1
                except RateLimitExceededException:
                    logger.error(
                        "GitHub API rate limit exceeded. Aborting GitHub search for this query. Try again later or use a GITHUB_TOKEN.")
                    break
                except GithubException as ge:
                    logger.warning(
                        f"GitHub API error processing file {file_item.path} in repo {file_item.repository.full_name}: {ge.status} {ge.data}")
                except Exception as e_file:
                    logger.error(
                        f"Error decoding or processing file {file_item.path} from {file_item.repository.full_name}: {e_file}",
                        exc_info=True)

            logger.info(f"Found {len(snippets)} actual Pinescript snippets from GitHub for '{query}'.")

        except RateLimitExceededException:
            logger.error(
                "GitHub API rate limit exceeded at the start of the search. Ensure GITHUB_TOKEN is set and valid.")
        except GithubException as e:
            logger.error(f"General GitHub API error during Pinescript search for '{query}': {e.status} {e.data}")
            if e.status == 403: logger.error(
                "This is often due to missing or invalid GITHUB_TOKEN or hitting secondary rate limits.")
        except Exception as e:
            logger.error(f"Unexpected error during GitHub Pinescript search for '{query}': {e}", exc_info=True)

        if not snippets or len(snippets) < (desired_snippets_count / 2):
            logger.info(
                f"Live GitHub search for Pinescript yielded few/no results for '{query}', adding built-in examples.")

        return snippets

    def _get_github_style_examples(self, query: str) -> List[str]:
        examples = []
        query_lower = query.lower()
        if any(term in query_lower for term in ['library', 'function', 'utility']):
            examples.append('''
//@version=5
library("TradingUtils")
export position_size(float capital, float risk_percent, float entry_price, float stop_loss_price) =>
    risk_amount = capital * (risk_percent / 100)
    price_diff = math.abs(entry_price - stop_loss_price)
    risk_amount / price_diff
export is_doji(float doji_threshold = 0.1) =>
    body_size = math.abs(close - open)
    total_range = high - low
    total_range > 0 ? (body_size / total_range) <= (doji_threshold / 100) : false
export get_sr_levels(int lookback = 50, int min_touches = 3) =>
    recent_high = ta.highest(high, lookback)
    recent_low = ta.lowest(low, lookback)
    [recent_low, recent_high]
            '''.strip())
        return examples


def search_pinescript_sources(query: str, logger) -> List[str]:
    try:
        from config import USER_AGENT, DEFAULT_REQUEST_TIMEOUT, PINESCRIPT_SOURCE_WEIGHTS
    except ImportError:
        logger.warning("Config not found in search_pinescript_sources, using fallbacks.")
        USER_AGENT = "RAGContentScraper/1.0 (fallback)"
        DEFAULT_REQUEST_TIMEOUT = 15
        PINESCRIPT_SOURCE_WEIGHTS = {
            'fetch_tradingview_public_library': 2.5,
            'fetch_github_pinescript': 1.5,
            'fetch_pinescript_examples': 2.0,
            'fetch_pinescript_documentation': 1.8
        }

    pinescript_sources_instance = PinescriptSources(USER_AGENT, DEFAULT_REQUEST_TIMEOUT)
    all_snippets = []

    source_methods_to_call = [
        pinescript_sources_instance.fetch_tradingview_public_library,
        pinescript_sources_instance.fetch_github_pinescript,
        pinescript_sources_instance.fetch_pinescript_examples,
        pinescript_sources_instance.fetch_pinescript_documentation,
    ]

    for method in source_methods_to_call:
        method_name = method.__name__
        try:
            logger.info(f"Calling Pinescript source method: {method_name} for query: {query}")
            snippets = method(query, logger)
            if snippets:
                all_snippets.extend(snippets)
                logger.info(f"Found {len(snippets)} from {method_name}.")
            else:
                logger.info(f"No snippets found from {method_name} for '{query}'.")
        except Exception as e:
            logger.error(f"Error in Pinescript source method {method_name}: {e}", exc_info=True)

    if all_snippets:
        unique_snippets = list(dict.fromkeys(all_snippets))
        logger.info(
            f"Total unique Pinescript snippets found after basic deduplication: {len(unique_snippets)} for query '{query}'")
        return unique_snippets
    else:
        logger.info(f"No Pinescript snippets found for query '{query}' after checking all sources.")
        return []