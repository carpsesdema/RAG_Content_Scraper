# config.py
"""
Centralized configuration for the RAG Content Scraper application.
Enhanced for freelance Python development and dual LLM systems.
"""

# --- General Application Settings ---
APP_NAME = "RAGContentScraper"
DEFAULT_LOGGER_NAME = "rag_content_scraper_logger"
LOG_FILE_PATH = "rag_scraper.log"
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"

# --- GUI Settings ---
STYLESHEET_PATH = "gui/styles.qss"
DEFAULT_WINDOW_TITLE = "RAG Content Scraper - Max Power Edition" # Fun title!
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 700

# --- Fetcher Settings ---
DEFAULT_REQUEST_TIMEOUT = 15  # Increased default timeout slightly
USER_AGENT = f"{APP_NAME}/1.1-greedy" # Indicate version/mode

# --- Crawler Settings ---
CRAWLER_ENABLED = True # Global toggle for new crawling features
MAX_CRAWL_QUERIES = 10 # Max new top-level search terms to generate from one initial query (excluding initial query)
MAX_CRAWL_QUEUE_SIZE = 50 # Absolute max size of the crawl queue to prevent runaway crawls

# GitHub Crawling
GITHUB_CRAWL_PARSE_DEPENDENCIES = True
# Stack Overflow Crawling
STACKOVERFLOW_CRAWL_EXPLORE_TAGS = True
# Documentation Site Crawling (for URL mode or discovered doc links)
DOCS_CRAWL_ENABLED = True
DOCS_CRAWL_MAX_DEPTH_PER_URL = 2 # How many link "hops" to make from an initial doc URL
DOCS_CRAWL_RESPECT_ROBOTS_TXT = True
# Allowed domains for DOCS_CRAWL - if empty, will only crawl same domain as entry URL
# Example: DOCS_CRAWL_ALLOWED_DOMAINS = ["docs.python.org", "realpython.com"]
DOCS_CRAWL_ALLOWED_DOMAINS = []

# --- Documentation Scraping Settings ---
DOCUMENTATION_SOURCES_ENABLED = True
DOC_SCRAPING_TIMEOUT = 20
DOC_MAX_PAGES_PER_SITE = 5
DOC_MAX_SNIPPETS_PER_PAGE = 10

# --- Searcher Settings ---

# Python Standard Library Docs
STDLIB_DOCS_BASE_URL = "https://docs.python.org/3/library/{module_name}.html"
STDLIB_DOCS_TIMEOUT = 15  # Increased

# Stack Overflow API
STACKEXCHANGE_API_BASE_URL = "https://api.stackexchange.com/2.3"
STACKOVERFLOW_SITE_NAME = "stackoverflow"
STACKOVERFLOW_SEARCH_ENDPOINT = "/search/advanced"
STACKOVERFLOW_ANSWERS_ENDPOINT = "/questions/{qid}/answers"

# MODIFIED: Increased Stack Overflow results
STACKOVERFLOW_SEARCH_MAX_RESULTS = 15         # Was 5, let's get more questions
STACKOVERFLOW_SEARCH_TIMEOUT = 20             # Was 15
STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION = 5    # Was 3, more answers per question
STACKOVERFLOW_ANSWERS_TIMEOUT = 15            # Was 10

# GitHub API
# MODIFIED: Increased GitHub README processing
GITHUB_README_MAX_REPOS = 7                   # Was 3, more repos for READMEs
GITHUB_README_SNIPPETS_PER_REPO = 10          # Was 5, more snippets per README

# MODIFIED: Significantly Increased GitHub File processing
GITHUB_FILES_MAX_REPOS = 10                   # Was 2, MANY more repos for files
GITHUB_FILES_PER_REPO_TARGET = 15             # Was 3, aim for more constructs per repo
GITHUB_FILES_CANDIDATE_MULTIPLIER = 3         # Was 2, fetch even more candidates to choose from
GITHUB_MAX_FILE_SIZE_KB = 750                 # Was 500, allow slightly larger files
GITHUB_FILE_DOWNLOAD_TIMEOUT = 20             # Was 15

# Option for fetch_github_file_snippets: If a .py file is small and no AST constructs are found,
# should we extract the whole file?
EXTRACT_WHOLE_SMALL_PY_FILES = True
MAX_LINES_FOR_WHOLE_FILE_EXTRACTION = 150 # If file has fewer lines than this, and EXTRACT_WHOLE_SMALL_PY_FILES is True

SEARCH_SOURCES_COUNT = 5 # Updated to include documentation sources

# --- Storage Settings ---
DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH = 60 # Slightly longer for potentially more varied snippets

# --- Enhanced Quality Filtering Settings ---
QUALITY_FILTER_ENABLED = True
MIN_SNIPPET_QUALITY_SCORE = 3  # Minimum score to include snippet
MIN_SNIPPET_LINES = 3
MAX_SNIPPET_LINES = 150
PREFER_DOCUMENTED_CODE = True  # Give bonus to code with docstrings

# --- Smart Deduplication Settings ---
SMART_DEDUPLICATION_ENABLED = True
SIMILARITY_THRESHOLD = 0.85  # How similar snippets need to be to be considered duplicates
SEMANTIC_DEDUPLICATION = False  # Enable when you have sentence-transformers installed

# --- Enhanced Source Settings ---
ADDITIONAL_SOURCES_ENABLED = True
FREELANCER_SOURCES_ENABLED = True
REAL_PYTHON_ENABLED = True
PYPI_EXAMPLES_ENABLED = True
PYTHON_ORG_ENABLED = True
AWESOME_PYTHON_ENABLED = False  # Enable when implemented

# Real Python settings
REAL_PYTHON_MAX_ARTICLES = 3
REAL_PYTHON_TIMEOUT = 20

# PyPI settings
PYPI_CHECK_ENABLED = True  # Check if query is a package name
PYPI_FETCH_DOCS = True     # Try to fetch documentation links

# --- Freelance-Specific Settings ---
FREELANCE_MODE = True
PRIORITIZE_PRACTICAL_EXAMPLES = True
INCLUDE_CLIENT_INTEGRATION_PATTERNS = True
FOCUS_ON_DELIVERABLE_CODE = True

# Content boosting for freelance relevance
BOOST_FASTAPI_CONTENT = 2.0
BOOST_TESTING_PATTERNS = 1.5
BOOST_DEPLOYMENT_EXAMPLES = 2.0
BOOST_CLIENT_INTEGRATIONS = 2.5
BOOST_DATA_PROCESSING = 1.8

# --- Code Categorization Settings ---
CODE_CATEGORIZATION_ENABLED = True
GENERATE_RELATED_QUERIES = True
FREELANCE_VALUE_SCORING = True
AUTO_GENERATE_TAGS = True

# --- Embedding RAG Export Settings ---
EMBEDDING_RAG_EXPORT_ENABLED = True
DUAL_LLM_EXPORT = True
RAG_EXPORT_ENABLED = True
DEFAULT_RAG_FORMAT = 'jsonl'  # 'jsonl', 'markdown', 'xml', 'yaml'
RAG_INCLUDE_METADATA = True
RAG_INCLUDE_QUALITY_SCORES = True
RAG_INCLUDE_TAGS = True

# Embedding optimization
MAX_EMBEDDING_CHUNK_SIZE = 1000
EMBEDDING_OVERLAP_SIZE = 100
MIN_EMBEDDING_CHUNK_SIZE = 50
GENERATE_IMPLEMENTATION_TIPS = True
IDENTIFY_COMMON_PITFALLS = True
SUGGEST_RELATED_CONCEPTS = True

# --- Performance & Rate Limiting ---
PARALLEL_SOURCE_FETCHING = True
CACHE_EMBEDDINGS = True
INCREMENTAL_RAG_UPDATES = True
SMART_RATE_LIMITING = True
REQUEST_DELAY_SECONDS = 0.5
MAX_WORKER_THREADS = 4
BATCH_SIZE = 10

# --- Context-Aware Search ---
CONTEXT_AWARE_SEARCH = True
QUERY_INTENT_DETECTION = True
PRESERVE_CHAT_CONTEXT = True
INTELLIGENT_QUERY_EXPANSION = True
MAX_QUERY_EXPANSION_DEPTH = 2
QUERY_EXPANSION_MIN_SCORE = 5

# --- Enhanced GitHub Settings ---
GITHUB_SEARCH_NOTEBOOKS = True      # Include Jupyter notebooks
GITHUB_PREFER_EXAMPLES = True       # Prioritize example/demo directories
GITHUB_SKIP_TESTS = False          # Whether to skip test files
GITHUB_INCLUDE_SETUP_PY = True     # Include setup.py files for package structure

# --- Enhanced Stack Overflow Settings ---
STACKOVERFLOW_PREFER_ACCEPTED = True    # Prioritize accepted answers
STACKOVERFLOW_MIN_SCORE = 1            # Minimum answer score
STACKOVERFLOW_INCLUDE_QUESTIONS = True  # Include question bodies, not just answers

# --- Freelance Project Types Priority ---
FREELANCE_PROJECT_TYPES = [
    'web_apis', 'data_processing', 'automation_scripts',
    'integrations', 'dashboards', 'scrapers', 'cli_tools',
    'payment_systems', 'authentication', 'deployment',
    'testing_frameworks', 'monitoring', 'ci_cd'
]

# --- High-Value Freelance Keywords ---
HIGH_VALUE_FREELANCE_KEYWORDS = [
    'fastapi', 'django', 'flask', 'stripe', 'twilio', 'sendgrid',
    'aws', 'docker', 'kubernetes', 'pytest', 'selenium', 'pandas',
    'oauth', 'jwt', 'rest', 'graphql', 'postgresql', 'redis',
    'celery', 'nginx', 'gunicorn', 'uvicorn'
]

# --- Content Processing Settings ---
EXTRACT_IMPORTS = True             # Extract and analyze import statements
EXTRACT_FUNCTION_SIGNATURES = True # Extract just function signatures for quick reference
EXTRACT_CLASS_HIERARCHIES = True  # Map class inheritance
INCLUDE_INLINE_COMMENTS = False   # Whether to preserve inline comments
ANALYZE_CODE_COMPLEXITY = True    # Analyze cyclomatic complexity
DETECT_CODE_PATTERNS = True       # Detect common coding patterns

# --- Storage Enhancement Settings ---
CREATE_INDEX_FILE = True           # Create searchable index of all snippets
COMPRESS_OUTPUT = False            # Compress output files (requires gzip)
BACKUP_EXISTING = True            # Backup existing files before overwriting
CREATE_METADATA_FILES = True      # Create separate metadata files
STORE_ORIGINAL_SOURCES = True     # Keep track of original source URLs

# --- Validation Settings ---
VALIDATE_PYTHON_SYNTAX = True     # Check if Python code is syntactically valid
SKIP_INVALID_SYNTAX = False       # Skip or include syntactically invalid code
SYNTAX_ERROR_AS_TEXT = True       # Include syntax errors as text snippets
VALIDATE_IMPORTS = True           # Check if imported modules exist

# --- Export Format Preferences ---
DEFAULT_DUAL_LLM_FORMAT = True
INCLUDE_IMPLEMENTATION_GUIDANCE = True
GENERATE_QUERY_SUGGESTIONS = True
CREATE_CROSS_REFERENCE_INDEX = True
EXPORT_SEPARATE_CHAT_CODE_FILES = True

# --- Experimental Features ---
AUTO_GENERATE_SUMMARIES = False    # Generate summaries of code snippets (needs AI)
EXTRACT_CODE_PATTERNS = True      # Identify common coding patterns
CLUSTER_SIMILAR_SNIPPETS = False  # Group similar snippets together
SEMANTIC_SEARCH_ENHANCEMENT = False # Enable semantic similarity matching
AUTO_CATEGORIZATION = True        # Automatically categorize snippets

# --- Development and Debug Settings ---
DEBUG_MODE = False
VERBOSE_LOGGING = False
SAVE_RAW_RESPONSES = False  # Save raw API responses for debugging
PROFILE_PERFORMANCE = False # Profile function execution times

# --- Integration Settings ---
ENABLE_WEBHOOKS = False           # Enable webhook notifications
WEBHOOK_URL = ""                  # URL to send webhook notifications
ENABLE_API_SERVER = False        # Enable built-in API server
API_SERVER_PORT = 8080           # Port for API server

# --- Query Templates for Enhanced LLM Integration ---
QUERY_TEMPLATES = {
    'api_request': 'Show me {framework} examples for {use_case} with error handling and validation',
    'data_task': 'Find {library} examples for {task} with performance considerations',
    'integration': 'Get {service} integration examples with authentication and error handling',
    'testing': 'Show me {framework} testing patterns for {component} with fixtures and mocks',
    'deployment': 'Find {platform} deployment examples for {app_type} with best practices',
    'automation': 'Get {tool} automation scripts for {task} with error handling'
}

# --- Freelance Client Value Indicators ---
CLIENT_VALUE_INDICATORS = {
    'high_value': {
        'payment_processing': ['stripe', 'paypal', 'square', 'braintree'],
        'communication': ['twilio', 'sendgrid', 'mailgun', 'slack'],
        'cloud_services': ['aws', 'azure', 'gcp', 'digitalocean'],
        'ecommerce': ['shopify', 'woocommerce', 'magento'],
        'analytics': ['google_analytics', 'mixpanel', 'amplitude']
    },
    'medium_value': {
        'automation': ['selenium', 'beautifulsoup', 'scrapy'],
        'data_processing': ['pandas', 'numpy', 'openpyxl'],
        'testing': ['pytest', 'unittest', 'mock'],
        'api_development': ['fastapi', 'django_rest', 'flask_restful']
    },
    'utility': {
        'file_operations': ['pathlib', 'shutil', 'os'],
        'text_processing': ['re', 'string', 'textwrap'],
        'date_time': ['datetime', 'dateutil', 'arrow']
    }
}

# --- Source Priority Weights ---
SOURCE_PRIORITY_WEIGHTS = {
    'stdlib': 1.0,
    'stackoverflow': 1.2,
    'github_readme': 1.1,
    'github_files': 1.5,  # Higher weight for actual implementation
    'documentation': 1.8,  # High priority for official docs
    'freelancer': 2.0,    # Highest weight for freelancer-specific sources
    'additional': 1.3
}