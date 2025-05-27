# utils/pinescript_categorizer.py

import re
from typing import Dict, List, Set
from collections import defaultdict


class PinescriptCategorizer:
    """Categorize Pinescript code snippets for better RAG retrieval."""

    def __init__(self):
        self.categories = {
            # Core script types
            'indicators': ['indicator(', 'study(', 'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic'],
            'strategies': ['strategy(', 'strategy.entry', 'strategy.exit', 'strategy.close', 'strategy.long',
                           'strategy.short'],
            'libraries': ['library(', 'export ', '@function', '@description'],

            # Technical analysis
            'moving_averages': ['sma', 'ema', 'wma', 'vwma', 'ta.sma', 'ta.ema', 'ta.wma', 'ta.vwma'],
            'oscillators': ['rsi', 'stochastic', 'cci', 'williams', 'ta.rsi', 'ta.stoch', 'ta.cci'],
            'momentum': ['macd', 'adx', 'atr', 'ta.macd', 'ta.adx', 'ta.atr', 'momentum'],
            'volume_analysis': ['volume', 'vwap', 'ta.vwap', 'obv', 'mfi', 'volume_profile'],
            'volatility': ['bollinger', 'atr', 'ta.bb', 'ta.atr', 'volatility', 'stdev'],
            'trend_analysis': ['trend', 'adx', 'parabolic', 'ichimoku', 'supertrend'],

            # Price action and patterns
            'price_action': ['candlestick', 'doji', 'hammer', 'engulfing', 'pattern'],
            'support_resistance': ['pivot', 'support', 'resistance', 'fibonacci', 'levels'],
            'chart_patterns': ['breakout', 'triangle', 'flag', 'pennant', 'head_shoulders'],

            # Risk management
            'risk_management': ['stop_loss', 'take_profit', 'position_size', 'risk', 'drawdown'],
            'money_management': ['equity', 'margin', 'leverage', 'capital', 'allocation'],

            # Alerts and notifications
            'alerts': ['alert(', 'alertcondition(', 'notification', 'webhook'],
            'automation': ['webhook', 'telegram', 'email', 'automated'],

            # Plotting and visualization
            'plotting': ['plot(', 'hline(', 'bgcolor(', 'plotshape(', 'plotchar(', 'plotcandle('],
            'visual_elements': ['label', 'line', 'box', 'table', 'polyline'],
            'styling': ['color', 'style', 'linewidth', 'transparency'],

            # Data and timeframes
            'multi_timeframe': ['request.security', 'timeframe', 'resolution'],
            'data_manipulation': ['array', 'matrix', 'map', 'request.'],
            'time_analysis': ['time', 'timestamp', 'dayofweek', 'month', 'year'],

            # Market structure
            'market_sessions': ['session', 'asian', 'london', 'new_york', 'regular_session'],
            'market_data': ['syminfo', 'ticker', 'currency', 'exchange'],
            'economic_calendar': ['earnings', 'dividend', 'splits', 'economic_events'],

            # Advanced features
            'user_defined_types': ['type ', 'method ', 'export method'],
            'arrays_matrices': ['array.', 'matrix.', 'map.'],
            'strings_formatting': ['str.', 'string', 'format'],
            'mathematical': ['math.', 'calculation', 'formula'],

            # Trading specific
            'backtesting': ['backtest', 'performance', 'equity_curve', 'strategy.'],
            'portfolio_management': ['portfolio', 'correlation', 'diversification'],
            'algo_trading': ['algorithm', 'systematic', 'quantitative', 'automated_trading']
        }

        self.complexity_patterns = {
            'beginner': ['plot(', 'input', 'close', 'high', 'low', 'open', 'volume'],
            'intermediate': ['ta.', 'strategy.', 'array.', 'request.', 'alertcondition'],
            'advanced': ['library(', 'export', 'method', 'type', 'matrix.', 'varip'],
            'expert': ['@function', 'polymorphic', 'recursive', 'advanced_arrays']
        }

        self.trading_value_indicators = {
            'high_value': {
                'professional_strategies': ['portfolio', 'multi_timeframe', 'risk_management', 'backtesting'],
                'institutional_tools': ['correlation', 'market_regime', 'volatility_modeling'],
                'automation': ['webhook', 'alerts', 'automated_trading'],
                'advanced_analytics': ['statistical', 'machine_learning', 'quantitative']
            },
            'medium_value': {
                'technical_indicators': ['custom_indicator', 'oscillator', 'momentum'],
                'trading_systems': ['entry_exit', 'signal_generation', 'trend_following'],
                'visualization': ['custom_plots', 'dashboard', 'table']
            },
            'educational': {
                'learning_tools': ['tutorial', 'example', 'educational'],
                'basic_concepts': ['introduction', 'beginner', 'fundamentals']
            }
        }

    def categorize_pinescript(self, code: str, metadata: Dict = None) -> Dict:
        """Categorize a Pinescript code snippet with detailed analysis."""
        if metadata is None:
            metadata = {}

        categories = set()
        patterns_found = set()
        complexity_level = 'beginner'
        trading_value_score = 0
        pinescript_version = self._extract_version(code)

        # Basic script type detection
        script_type = self._determine_script_type(code)
        if script_type:
            categories.add(script_type)

        # Extract features from code
        code_lower = code.lower()

        # Categorize based on content patterns
        for category, keywords in self.categories.items():
            if any(keyword.lower() in code_lower for keyword in keywords):
                categories.add(category)

        # Determine complexity
        for level, keywords in self.complexity_patterns.items():
            if any(keyword.lower() in code_lower for keyword in keywords):
                complexity_level = level

        # Detect specific patterns
        patterns_found = self._detect_pinescript_patterns(code)

        # Calculate trading value score
        for value_type, value_categories in self.trading_value_indicators.items():
            for subcategory, keywords in value_categories.items():
                if any(keyword in code_lower or keyword in ' '.join(categories) for keyword in keywords):
                    if value_type == 'high_value':
                        trading_value_score += 3
                    elif value_type == 'medium_value':
                        trading_value_score += 2
                    else:
                        trading_value_score += 1

        # Extract technical analysis components
        ta_components = self._extract_ta_components(code)

        # Detect trading logic
        trading_logic = self._analyze_trading_logic(code)

        # Use case detection for trading
        use_cases = self._determine_use_cases(categories, patterns_found)

        return {
            'categories': list(categories),
            'patterns': list(patterns_found),
            'complexity': complexity_level,
            'pinescript_version': pinescript_version,
            'script_type': script_type,
            'trading_value_score': trading_value_score,
            'ta_components': ta_components,
            'trading_logic': trading_logic,
            'use_cases': list(use_cases),
            'trading_relevant': len(use_cases) > 0,
            'client_value': self._assess_trading_value(categories, patterns_found, trading_value_score)
        }

    def _extract_version(self, code: str) -> str:
        """Extract Pinescript version from code."""
        version_match = re.search(r'//@version\s*=\s*([0-9]+)', code)
        return f"v{version_match.group(1)}" if version_match else "unknown"

    def _determine_script_type(self, code: str) -> str:
        """Determine the type of Pinescript (indicator, strategy, library)."""
        code_lower = code.lower()

        if 'library(' in code_lower:
            return 'library'
        elif 'strategy(' in code_lower:
            return 'strategy'
        elif 'indicator(' in code_lower or 'study(' in code_lower:
            return 'indicator'
        else:
            return 'script'

    def _detect_pinescript_patterns(self, code: str) -> Set[str]:
        """Detect specific Pinescript coding patterns."""
        patterns = set()

        # Function definitions
        if re.search(r'^\s*\w+\s*\([^)]*\)\s*=>', code, re.MULTILINE):
            patterns.add('user_defined_functions')

        # Variable declarations
        if re.search(r'var\s+\w+', code):
            patterns.add('variable_declarations')
        if re.search(r'varip\s+\w+', code):
            patterns.add('series_variables')

        # Input parameters
        if 'input(' in code or 'input.' in code:
            patterns.add('input_parameters')

        # Plotting
        if any(plot_func in code for plot_func in ['plot(', 'plotshape(', 'plotchar(', 'plotcandle(']):
            patterns.add('plotting')

        # Alerts
        if 'alertcondition(' in code or 'alert(' in code:
            patterns.add('alerts')

        # Arrays and advanced data structures
        if 'array.' in code:
            patterns.add('arrays')
        if 'matrix.' in code:
            patterns.add('matrices')
        if 'map.' in code:
            patterns.add('maps')

        # Tables and labels
        if 'table.' in code:
            patterns.add('tables')
        if 'label.' in code or 'line.' in code:
            patterns.add('annotations')

        # Request functions (multi-timeframe)
        if 'request.' in code:
            patterns.add('multi_timeframe')

        # Strategy specific
        if any(strat_func in code for strat_func in ['strategy.entry', 'strategy.exit', 'strategy.close']):
            patterns.add('strategy_orders')

        # Technical analysis
        if 'ta.' in code:
            patterns.add('builtin_ta_functions')

        # Custom calculations
        if any(math_func in code for math_func in ['math.', 'nz(', 'na(', 'barstate.']):
            patterns.add('mathematical_operations')

        return patterns

    def _extract_ta_components(self, code: str) -> List[str]:
        """Extract technical analysis components mentioned in the code."""
        ta_components = []

        # Common TA patterns
        ta_patterns = {
            'moving_averages': ['sma', 'ema', 'wma', 'vwma', 'hull', 'alma'],
            'oscillators': ['rsi', 'stoch', 'cci', 'williams', 'mfi'],
            'momentum': ['macd', 'adx', 'atr', 'roc', 'tsi'],
            'volatility': ['bb', 'kc', 'atr', 'stdev'],
            'volume': ['vwap', 'obv', 'ad', 'cmf', 'pvt'],
            'trend': ['adx', 'aroon', 'parabolic', 'supertrend'],
            'support_resistance': ['pivot', 'fibonacci', 'camarilla']
        }

        code_lower = code.lower()
        for category, indicators in ta_patterns.items():
            found_indicators = [ind for ind in indicators if ind in code_lower]
            if found_indicators:
                ta_components.extend([f"{category}:{ind}" for ind in found_indicators])

        return ta_components

    def _analyze_trading_logic(self, code: str) -> Dict[str, bool]:
        """Analyze the trading logic present in the code."""
        logic_analysis = {
            'has_entry_conditions': False,
            'has_exit_conditions': False,
            'has_risk_management': False,
            'has_position_sizing': False,
            'has_alerts': False,
            'has_backtesting': False,
            'has_multi_timeframe': False
        }

        code_lower = code.lower()

        # Entry conditions
        if any(term in code_lower for term in ['strategy.entry', 'long_condition', 'short_condition', 'buy', 'sell']):
            logic_analysis['has_entry_conditions'] = True

        # Exit conditions
        if any(term in code_lower for term in ['strategy.exit', 'strategy.close', 'take_profit', 'stop_loss']):
            logic_analysis['has_exit_conditions'] = True

        # Risk management
        if any(term in code_lower for term in ['stop_loss', 'take_profit', 'risk', 'drawdown']):
            logic_analysis['has_risk_management'] = True

        # Position sizing
        if any(term in code_lower for term in ['position_size', 'qty', 'quantity', 'default_qty']):
            logic_analysis['has_position_sizing'] = True

        # Alerts
        if any(term in code_lower for term in ['alert', 'notification', 'webhook']):
            logic_analysis['has_alerts'] = True

        # Backtesting
        if 'strategy(' in code_lower:
            logic_analysis['has_backtesting'] = True

        # Multi-timeframe
        if 'request.security' in code_lower or 'timeframe' in code_lower:
            logic_analysis['has_multi_timeframe'] = True

        return logic_analysis

    def _determine_use_cases(self, categories: Set[str], patterns: Set[str]) -> Set[str]:
        """Determine practical use cases for trading."""
        use_cases = set()

        # Strategy development
        if any(cat in categories for cat in ['strategies', 'backtesting', 'risk_management']):
            use_cases.add('strategy_development')

        # Technical analysis
        if any(cat in categories for cat in ['indicators', 'oscillators', 'momentum', 'volatility']):
            use_cases.add('technical_analysis')

        # Alert systems
        if 'alerts' in categories or 'alerts' in patterns:
            use_cases.add('alert_systems')

        # Visualization
        if any(cat in categories for cat in ['plotting', 'visual_elements']):
            use_cases.add('chart_visualization')

        # Automation
        if any(cat in categories for cat in ['automation', 'webhook']):
            use_cases.add('trading_automation')

        # Research and analysis
        if any(cat in categories for cat in ['multi_timeframe', 'data_manipulation']):
            use_cases.add('market_research')

        # Portfolio management
        if any(cat in categories for cat in ['portfolio_management', 'correlation']):
            use_cases.add('portfolio_analysis')

        # Education
        if any(pattern in patterns for pattern in ['input_parameters', 'plotting']):
            use_cases.add('educational_tools')

        return use_cases

    def _assess_trading_value(self, categories: Set[str], patterns: Set[str], trading_score: int) -> str:
        """Assess the trading value of this code."""
        high_value_categories = {
            'professional_trading', 'risk_management', 'portfolio_management',
            'automation', 'backtesting', 'multi_timeframe'
        }

        medium_value_categories = {
            'strategies', 'indicators', 'alerts', 'technical_analysis'
        }

        if trading_score >= 8 or any(cat in categories for cat in high_value_categories):
            return 'high_value_professional_trading'
        elif trading_score >= 4 or any(cat in categories for cat in medium_value_categories):
            return 'medium_value_trading_tools'
        elif any(cat in categories for cat in ['plotting', 'visual_elements']):
            return 'visualization_tools'
        else:
            return 'educational_examples'

    def generate_pinescript_tags(self, categorization: Dict) -> List[str]:
        """Generate optimized tags for Pinescript RAG retrieval."""
        tags = []

        # Add category tags
        for category in categorization['categories']:
            tags.append(f"category:{category}")

        # Add complexity tag
        tags.append(f"complexity:{categorization['complexity']}")

        # Add script type
        tags.append(f"type:{categorization['script_type']}")

        # Add version
        tags.append(f"version:{categorization['pinescript_version']}")

        # Add pattern tags
        for pattern in categorization['patterns']:
            tags.append(f"pattern:{pattern}")

        # Add use case tags
        for use_case in categorization['use_cases']:
            tags.append(f"usecase:{use_case}")

        # Add TA component tags
        for component in categorization['ta_components'][:5]:  # Limit to avoid tag explosion
            tags.append(f"ta:{component}")

        # Add trading relevance
        if categorization['trading_relevant']:
            tags.append("trading:relevant")
            tags.append(f"trading_score:{categorization['trading_value_score']}")

        # Add client value
        tags.append(f"client_value:{categorization['client_value']}")

        # Add trading logic tags
        trading_logic = categorization.get('trading_logic', {})
        for logic_type, has_logic in trading_logic.items():
            if has_logic:
                tags.append(f"logic:{logic_type}")

        return tags

    def suggest_related_pinescript_queries(self, categorization: Dict) -> List[str]:
        """Suggest related Pinescript search queries based on categorization."""
        suggestions = []

        categories = categorization['categories']
        script_type = categorization['script_type']

        # Script type based suggestions
        if script_type == 'strategy':
            suggestions.extend([
                'pinescript strategy examples', 'backtesting strategies', 'trading algorithm pinescript',
                'strategy entry exit conditions', 'risk management pinescript'
            ])
        elif script_type == 'indicator':
            suggestions.extend([
                'custom indicators pinescript', 'technical analysis indicators', 'trading indicators',
                'pinescript oscillators', 'trend indicators'
            ])
        elif script_type == 'library':
            suggestions.extend([
                'pinescript library functions', 'reusable pinescript code', 'pinescript utilities'
            ])

        # Category-based suggestions
        if 'moving_averages' in categories:
            suggestions.extend(['sma ema pinescript', 'moving average crossover', 'adaptive moving averages'])
        if 'oscillators' in categories:
            suggestions.extend(['rsi pinescript', 'stochastic oscillator', 'momentum oscillators'])
        if 'alerts' in categories:
            suggestions.extend(['pinescript alerts', 'trading notifications', 'webhook alerts pinescript'])
        if 'risk_management' in categories:
            suggestions.extend(['stop loss pinescript', 'position sizing', 'risk reward ratio'])
        if 'multi_timeframe' in categories:
            suggestions.extend(['multi timeframe analysis', 'higher timeframe pinescript', 'mtf indicators'])

        # Pattern-based suggestions
        patterns = categorization.get('patterns', [])
        if 'arrays' in patterns:
            suggestions.extend(['pinescript arrays', 'array functions pinescript'])
        if 'strategy_orders' in patterns:
            suggestions.extend(['strategy orders pinescript', 'entry exit signals'])

        return list(set(suggestions))  # Remove duplicates