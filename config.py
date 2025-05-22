# config.py
"""
Centralized configuration for the RAG Content Scraper application.
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

# --- Crawler Settings (NEW) ---
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

SEARCH_SOURCES_COUNT = 4 # Stays the same for now unless we add a new *type* of source

# --- Storage Settings ---
DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH = 60 # Slightly longer for potentially more varied snippets