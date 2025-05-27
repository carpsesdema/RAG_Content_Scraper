# config.py - Updated with Pinescript Support
"""
Centralized configuration for the RAG Content Scraper application.
Enhanced for freelance Python development, dual LLM systems, and Pinescript trading.
"""

# --- General Application Settings ---
APP_NAME = "RAGContentScraper"
DEFAULT_LOGGER_NAME = "rag_content_scraper_logger"
LOG_FILE_PATH = "rag_scraper.log"
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"

# --- GUI Settings ---
STYLESHEET_PATH = "gui/styles.qss"
DEFAULT_WINDOW_TITLE = "RAG Content Scraper - Max Power Edition with Pinescript" # Updated title!
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 700

# --- Fetcher Settings ---
DEFAULT_REQUEST_TIMEOUT = 15
USER_AGENT = f"{APP_NAME}/1.2-pinescript"  # Updated version

# --- Content Type Settings ---
CONTENT_TYPES = {
    'python': True,      # Original Python support
    'pinescript': True,  # NEW: Pinescript support
    'javascript': False, # Future expansion
    'sql': False,        # Future expansion
    'bash': False        # Future expansion
}

# Default content type (can be 'python', 'pinescript', or 'auto')
DEFAULT_CONTENT_TYPE = 'auto'  # Auto-detect based on query

# --- Pinescript-Specific Settings ---
PINESCRIPT_ENABLED = True
PINESCRIPT_SOURCES_ENABLED = True
TRADINGVIEW_SCRAPING_ENABLED = True
PINESCRIPT_DOCUMENTATION_ENABLED = True
PINESCRIPT_GITHUB_ENABLED = True

# Pinescript source priorities
PINESCRIPT_SOURCE_WEIGHTS = {
    'builtin_examples': 2.0,      # High priority for built-in examples
    'documentation': 1.8,         # Official docs
    'tradingview_library': 2.5,   # Highest priority for TradingView public scripts
    'github_pinescript': 1.5,     # GitHub repositories
    'educational': 1.3            # Educational content
}

# TradingView settings
TRADINGVIEW_MAX_SCRIPTS = 10
TRADINGVIEW_TIMEOUT = 20
TRADINGVIEW_RESPECT_RATE_LIMITS = True
TRADINGVIEW_DELAY_BETWEEN_REQUESTS = 2.0

# Pinescript categorization
PINESCRIPT_CATEGORIZATION_ENABLED = True
PINESCRIPT_COMPLEXITY_SCORING = True
TRADING_VALUE_SCORING = True

# --- Language Detection Settings ---
LANGUAGE_DETECTION_ENABLED = True
AUTO_DETECT_CONTENT_TYPE = True

# Keywords for content type detection
CONTENT_TYPE_KEYWORDS = {
    'python': [
        'import ', 'from ', 'def ', 'class ', 'if __name__', 'print(',
        'pandas', 'numpy', 'fastapi', 'django', 'flask', 'pytest'
    ],
    'pinescript': [
        '//@version', 'indicator(', 'strategy(', 'library(', 'plot(',
        'ta.', 'strategy.entry', 'alertcondition', 'input.', 'close', 'high', 'low',
        'sma', 'ema', 'rsi', 'macd', 'bollinger'
    ]
}

# --- Enhanced Source Settings ---
ADDITIONAL_SOURCES_ENABLED = True
FREELANCER_SOURCES_ENABLED = True  # Python freelance sources
PINESCRIPT_SOURCES_ENABLED = True  # NEW: Pinescript sources

# Updated search sources count to include Pinescript
SEARCH_SOURCES_COUNT = 6  # stdlib, stackoverflow, github_readme, github_files, additional/freelancer, pinescript

# --- Updated Source Priority Weights ---
SOURCE_PRIORITY_WEIGHTS = {
    'stdlib': 1.0,
    'stackoverflow': 1.2,
    'github_readme': 1.1,
    'github_files': 1.5,
    'documentation': 1.8,
    'freelancer': 2.0,        # Python freelance sources
    'additional': 1.3,
    'pinescript': 2.2,        # NEW: Pinescript sources get high priority
    'tradingview': 2.5        # NEW: TradingView gets highest priority for trading
}

# --- Trading-Specific Settings ---
TRADING_MODE = False  # Set to True to prioritize trading content
TRADING_FOCUS_AREAS = [
    'indicators', 'strategies', 'backtesting', 'risk_management',
    'alerts', 'automation', 'portfolio_analysis', 'market_analysis'
]

# High-value trading keywords
HIGH_VALUE_TRADING_KEYWORDS = [
    'strategy', 'backtest', 'risk_management', 'portfolio', 'automation',
    'multi_timeframe', 'alerts', 'webhook', 'quantitative', 'systematic'
]

# Trading project types priority
TRADING_PROJECT_TYPES = [
    'trading_strategies', 'custom_indicators', 'alert_systems',
    'portfolio_tools', 'market_analysis', 'risk_tools',
    'backtesting_frameworks', 'trading_automation'
]

# --- Content Processing Enhancement ---
MULTI_LANGUAGE_PROCESSING = True
CROSS_LANGUAGE_SUGGESTIONS = True  # Suggest related concepts across languages

# Language-specific processing
LANGUAGE_PROCESSORS = {
    'python': {
        'ast_analysis': True,
        'import_extraction': True,
        'function_extraction': True,
        'class_extraction': True
    },
    'pinescript': {
        'version_detection': True,
        'script_type_detection': True,
        'ta_component_extraction': True,
        'trading_logic_analysis': True
    }
}

# --- Updated Quality Filtering ---
QUALITY_FILTER_ENABLED = True
LANGUAGE_SPECIFIC_QUALITY_FILTERS = True

# Quality scoring by language
QUALITY_SCORE_WEIGHTS = {
    'python': {
        'min_score': 3,
        'complexity_weight': 1.0,
        'documentation_weight': 1.5,
        'testing_weight': 1.2
    },
    'pinescript': {
        'min_score': 4,
        'trading_logic_weight': 2.0,
        'version_weight': 1.3,
        'complexity_weight': 1.0,
        'practical_value_weight': 1.8
    }
}

# --- Updated Categorization Settings ---
CODE_CATEGORIZATION_ENABLED = True
LANGUAGE_SPECIFIC_CATEGORIZATION = True

# Categorization settings by language
CATEGORIZATION_SETTINGS = {
    'python': {
        'freelance_scoring': True,
        'complexity_analysis': True,
        'pattern_detection': True
    },
    'pinescript': {
        'trading_value_scoring': True,
        'ta_component_analysis': True,
        'strategy_analysis': True,
        'complexity_analysis': True
    }
}

# --- Enhanced Export Settings ---
EMBEDDING_RAG_EXPORT_ENABLED = True
DUAL_LLM_EXPORT = True
LANGUAGE_SPECIFIC_EXPORTS = True

# Export formats by content type
EXPORT_FORMATS_BY_LANGUAGE = {
    'python': ['jsonl', 'markdown', 'xml', 'yaml', 'dual_llm'],
    'pinescript': ['jsonl', 'markdown', 'tradingview_friendly', 'dual_llm']
}

# --- Query Enhancement Settings ---
QUERY_ENHANCEMENT_ENABLED = True
INTELLIGENT_QUERY_EXPANSION = True
CROSS_LANGUAGE_QUERY_MAPPING = True

# Query mapping between languages
QUERY_CROSS_MAPPINGS = {
    'python_to_pinescript': {
        'pandas': 'array analysis',
        'matplotlib': 'plot functions',
        'numpy': 'math calculations',
        'requests': 'request.security',
        'time': 'timeframe analysis'
    },
    'pinescript_to_python': {
        'strategy': 'backtesting framework',
        'indicator': 'technical analysis',
        'plot': 'visualization',
        'ta.': 'technical indicators',
        'alert': 'notification system'
    }
}

# --- Content-Specific Templates ---
QUERY_TEMPLATES_BY_LANGUAGE = {
    'python': {
        'api_request': 'Show me {framework} examples for {use_case} with error handling and validation',
        'data_task': 'Find {library} examples for {task} with performance considerations',
        'integration': 'Get {service} integration examples with authentication and error handling',
        'testing': 'Show me {framework} testing patterns for {component} with fixtures and mocks'
    },
    'pinescript': {
        'indicator_request': 'Show me {indicator} examples with {features} and proper plotting',
        'strategy_request': 'Find {strategy_type} strategies with {risk_features} and backtesting',
        'alert_request': 'Get {alert_type} alert examples with {notification_method}',
        'analysis_request': 'Show me {analysis_type} with {timeframe} and {visualization}'
    }
}

# --- Performance and Caching ---
LANGUAGE_SPECIFIC_CACHING = True
CACHE_BY_CONTENT_TYPE = True

# Cache settings by language
CACHE_SETTINGS = {
    'python': {
        'cache_duration_hours': 24,
        'max_cache_size_mb': 100
    },
    'pinescript': {
        'cache_duration_hours': 48,  # Longer cache for trading content
        'max_cache_size_mb': 50
    }
}

# --- Validation Settings ---
LANGUAGE_SPECIFIC_VALIDATION = True

VALIDATION_SETTINGS = {
    'python': {
        'syntax_validation': True,
        'import_validation': True
    },
    'pinescript': {
        'version_validation': True,
        'syntax_validation': True,
        'script_type_validation': True
    }
}

# --- UI Enhancements for Multi-Language ---
SHOW_LANGUAGE_SELECTOR = True
LANGUAGE_SPECIFIC_UI_HINTS = True
CROSS_LANGUAGE_SUGGESTIONS_UI = True

# Language display names
LANGUAGE_DISPLAY_NAMES = {
    'python': 'Python Development',
    'pinescript': 'Pinescript Trading',
    'auto': 'Auto-Detect'
}

# --- Advanced Features ---
EXPERIMENTAL_FEATURES = {
    'cross_language_learning': False,     # Learn patterns across languages
    'trading_signal_detection': True,     # Detect trading signals in Pinescript
    'code_conversion_suggestions': False, # Suggest equivalent code in other languages
    'market_context_awareness': True      # Understand market context in queries
}

# --- Legacy Compatibility ---
# Keep all existing Python-focused settings for backward compatibility
FREELANCE_MODE = True
QUALITY_FILTER_ENABLED = True
SMART_DEDUPLICATION_ENABLED = True
RAG_EXPORT_ENABLED = True

# Existing Python settings remain unchanged...
# (All the previous Python-specific configurations are preserved)

# --- Existing settings preserved for backward compatibility ---
STDLIB_DOCS_BASE_URL = "https://docs.python.org/3/library/{module_name}.html"
STDLIB_DOCS_TIMEOUT = 15
STACKEXCHANGE_API_BASE_URL = "https://api.stackexchange.com/2.3"
STACKOVERFLOW_SITE_NAME = "stackoverflow"
STACKOVERFLOW_SEARCH_ENDPOINT = "/search/advanced"
STACKOVERFLOW_ANSWERS_ENDPOINT = "/questions/{qid}/answers"
STACKOVERFLOW_SEARCH_MAX_RESULTS = 15
STACKOVERFLOW_SEARCH_TIMEOUT = 20
STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION = 5
STACKOVERFLOW_ANSWERS_TIMEOUT = 15
GITHUB_README_MAX_REPOS = 7
GITHUB_README_SNIPPETS_PER_REPO = 10
GITHUB_FILES_MAX_REPOS = 10
GITHUB_FILES_PER_REPO_TARGET = 15
GITHUB_FILES_CANDIDATE_MULTIPLIER = 3
GITHUB_MAX_FILE_SIZE_KB = 750
GITHUB_FILE_DOWNLOAD_TIMEOUT = 20
EXTRACT_WHOLE_SMALL_PY_FILES = True
MAX_LINES_FOR_WHOLE_FILE_EXTRACTION = 150
DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH = 60

# All other existing configurations remain the same...
# (This ensures backward compatibility while adding Pinescript support)