import os
import re
import requests
from bs4 import BeautifulSoup
from github import Github, GithubException
import ast
import logging
from .parser import _clean_snippet_text, extract_code as extract_code_from_html_markdown, extract_code_from_ipynb

# Enhanced features imports
try:
    from .quality_filter import CodeQualityFilter
    from .additional_sources import search_additional_sources
    from .freelancer_sources import search_freelancer_sources
    from .pinescript_sources import search_pinescript_sources  # NEW
    from ..utils.deduplicator import SmartDeduplicator
    from ..utils.code_categorizer import CodeCategorizer
    from ..utils.pinescript_categorizer import PinescriptCategorizer  # NEW

    ENHANCED_FEATURES_AVAILABLE = True
    FREELANCER_FEATURES_AVAILABLE = True
    PINESCRIPT_FEATURES_AVAILABLE = True  # NEW
except ImportError as e:
    print(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    FREELANCER_FEATURES_AVAILABLE = False
    PINESCRIPT_FEATURES_AVAILABLE = False

# Import configurations with fallbacks
try:
    from config import (
        # Existing Python configurations
        STDLIB_DOCS_BASE_URL, STDLIB_DOCS_TIMEOUT,
        STACKEXCHANGE_API_BASE_URL, STACKOVERFLOW_SITE_NAME,
        STACKOVERFLOW_SEARCH_ENDPOINT, STACKOVERFLOW_ANSWERS_ENDPOINT,
        STACKOVERFLOW_SEARCH_MAX_RESULTS, STACKOVERFLOW_SEARCH_TIMEOUT,
        STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION, STACKOVERFLOW_ANSWERS_TIMEOUT,
        GITHUB_README_MAX_REPOS, GITHUB_README_SNIPPETS_PER_REPO,
        GITHUB_FILES_MAX_REPOS, GITHUB_FILES_PER_REPO_TARGET,
        GITHUB_FILES_CANDIDATE_MULTIPLIER, GITHUB_MAX_FILE_SIZE_KB,
        GITHUB_FILE_DOWNLOAD_TIMEOUT, USER_AGENT,
        EXTRACT_WHOLE_SMALL_PY_FILES, MAX_LINES_FOR_WHOLE_FILE_EXTRACTION,
        # Enhanced settings
        QUALITY_FILTER_ENABLED, SMART_DEDUPLICATION_ENABLED,
        ADDITIONAL_SOURCES_ENABLED, FREELANCER_SOURCES_ENABLED,
        CODE_CATEGORIZATION_ENABLED, FREELANCE_MODE,
        SOURCE_PRIORITY_WEIGHTS,
        # NEW: Pinescript settings
        PINESCRIPT_ENABLED, PINESCRIPT_SOURCES_ENABLED,
        CONTENT_TYPES, DEFAULT_CONTENT_TYPE,
        LANGUAGE_DETECTION_ENABLED, CONTENT_TYPE_KEYWORDS,
        PINESCRIPT_SOURCE_WEIGHTS, TRADING_MODE,
        LANGUAGE_SPECIFIC_QUALITY_FILTERS, QUALITY_SCORE_WEIGHTS,
        LANGUAGE_SPECIFIC_CATEGORIZATION, CATEGORIZATION_SETTINGS
    )
except ImportError:
    logging.critical("Config not found, using fallbacks")
    # Existing fallbacks...
    STDLIB_DOCS_BASE_URL = "https://docs.python.org/3/library/{module_name}.html"
    STDLIB_DOCS_TIMEOUT = 10
    # ... (all existing fallbacks)

    # NEW: Pinescript fallbacks
    PINESCRIPT_ENABLED = False
    PINESCRIPT_SOURCES_ENABLED = False
    CONTENT_TYPES = {'python': True, 'pinescript': False}
    DEFAULT_CONTENT_TYPE = 'python'
    LANGUAGE_DETECTION_ENABLED = False
    CONTENT_TYPE_KEYWORDS = {'python': ['import ', 'def '], 'pinescript': ['//@version']}
    PINESCRIPT_SOURCE_WEIGHTS = {}
    TRADING_MODE = False
    LANGUAGE_SPECIFIC_QUALITY_FILTERS = False
    QUALITY_SCORE_WEIGHTS = {}
    LANGUAGE_SPECIFIC_CATEGORIZATION = False
    CATEGORIZATION_SETTINGS = {}


def detect_content_type(query: str, logger) -> str:
    """Detect the content type based on query content."""
    if not LANGUAGE_DETECTION_ENABLED:
        return DEFAULT_CONTENT_TYPE

    query_lower = query.lower()

    # Check for explicit language indicators
    if any(keyword in query_lower for keyword in ['pinescript', 'tradingview', 'trading', 'indicator', 'strategy']):
        # Further check for Pinescript-specific terms
        pinescript_terms = CONTENT_TYPE_KEYWORDS.get('pinescript', [])
        if any(term in query_lower for term in
               ['pinescript', 'tradingview', '//@version', 'ta.', 'strategy(', 'indicator(']):
            logger.info(f"Detected Pinescript content type for query: {query}")
            return 'pinescript'

    if any(keyword in query_lower for keyword in ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy']):
        # Further check for Python-specific terms
        python_terms = CONTENT_TYPE_KEYWORDS.get('python', [])
        if any(term in query_lower for term in python_terms[:5]):  # Check first 5 terms
            logger.info(f"Detected Python content type for query: {query}")
            return 'python'

    # Check for trading/finance terms that might indicate Pinescript
    trading_terms = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'trading', 'backtest', 'strategy', 'alert']
    if any(term in query_lower for term in trading_terms):
        logger.info(f"Detected potential Pinescript content (trading terms) for query: {query}")
        return 'pinescript'

    # Default behavior
    if DEFAULT_CONTENT_TYPE == 'auto':
        # If auto, prefer Python for backward compatibility
        logger.info(f"Auto-detecting content type, defaulting to Python for query: {query}")
        return 'python'

    logger.info(f"Using default content type '{DEFAULT_CONTENT_TYPE}' for query: {query}")
    return DEFAULT_CONTENT_TYPE


def search_and_fetch(query, logger, progress_callback=None, content_type=None):
    """Enhanced search with multi-language support including Pinescript."""

    global ENHANCED_FEATURES_AVAILABLE, PINESCRIPT_FEATURES_AVAILABLE

    if logger is None:
        logger = logging.getLogger(USER_AGENT)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

    # Detect content type if not specified
    if content_type is None:
        content_type = detect_content_type(query, logger)

    logger.info(f"Processing query '{query}' as {content_type} content")

    # Initialize enhanced components based on content type
    enhanced_features_working = ENHANCED_FEATURES_AVAILABLE
    quality_filter = None
    deduplicator = None
    categorizer = None

    if ENHANCED_FEATURES_AVAILABLE:
        try:
            if QUALITY_FILTER_ENABLED:
                quality_filter = CodeQualityFilter()
            if SMART_DEDUPLICATION_ENABLED:
                deduplicator = SmartDeduplicator()
            if CODE_CATEGORIZATION_ENABLED:
                if content_type == 'pinescript' and PINESCRIPT_FEATURES_AVAILABLE:
                    categorizer = PinescriptCategorizer()
                else:
                    categorizer = CodeCategorizer()
        except Exception as e:
            logger.warning(f"Could not initialize enhanced features: {e}")
            enhanced_features_working = False

    all_snippets = []
    all_sources = []
    discovered_queries = set()

    # Calculate total steps for progress based on content type
    if content_type == 'pinescript':
        total_configured_steps = 1  # Just Pinescript sources for now
        if enhanced_features_working and PINESCRIPT_SOURCES_ENABLED and PINESCRIPT_FEATURES_AVAILABLE:
            total_configured_steps += 1
    else:
        # Original Python sources
        total_configured_steps = 4  # stdlib, stackoverflow, github_readme, github_files
        if enhanced_features_working and ADDITIONAL_SOURCES_ENABLED:
            total_configured_steps += 1
        if enhanced_features_working and FREELANCER_FEATURES_AVAILABLE and FREELANCER_SOURCES_ENABLED:
            total_configured_steps += 1

    current_step_for_progress = 0

    def _do_progress_update(source_name):
        nonlocal current_step_for_progress
        current_step_for_progress += 1
        if progress_callback:
            percentage = int((current_step_for_progress / total_configured_steps) * 85)
            progress_callback(f"Query '{query}': Fetching from {source_name}...", percentage)

    # === CONTENT-TYPE SPECIFIC SOURCES ===

    if content_type == 'pinescript':
        # === PINESCRIPT SOURCES ===
        if enhanced_features_working and PINESCRIPT_SOURCES_ENABLED and PINESCRIPT_FEATURES_AVAILABLE:
            _do_progress_update("Pinescript Sources")
            try:
                pinescript_snippets = search_pinescript_sources(query, logger)
                all_snippets.extend(pinescript_snippets)
                all_sources.extend(['pinescript'] * len(pinescript_snippets))
                logger.info(f"Found {len(pinescript_snippets)} Pinescript snippets")
            except Exception as e:
                logger.error(f"Error in pinescript sources for '{query}': {e}", exc_info=True)

        # Additional Pinescript-specific sources could be added here
        # For example, TradingView-specific scraping, Pine Script documentation, etc.

    else:
        # === PYTHON SOURCES (EXISTING) ===

        # Python Standard Library
        _do_progress_update("Python Standard Library")
        try:
            stdlib_snippets = fetch_stdlib_docs(query, logger)
            all_snippets.extend(stdlib_snippets)
            all_sources.extend(['stdlib'] * len(stdlib_snippets))
        except Exception as e:
            logger.error(f"Error in fetch_stdlib_docs for '{query}': {e}", exc_info=True)

        # Stack Overflow
        _do_progress_update("Stack Overflow")
        try:
            so_snippets, so_tags = fetch_stackoverflow_snippets(query, logger)
            all_snippets.extend(so_snippets)
            all_sources.extend(['stackoverflow'] * len(so_snippets))
            discovered_queries.update(so_tags)
        except Exception as e:
            logger.error(f"Error in fetch_stackoverflow_snippets for '{query}': {e}", exc_info=True)

        # GitHub READMEs
        _do_progress_update("GitHub READMEs")
        try:
            readme_snippets = fetch_github_readme_snippets(query, logger)
            all_snippets.extend(readme_snippets)
            all_sources.extend(['github_readme'] * len(readme_snippets))
        except Exception as e:
            logger.error(f"Error in fetch_github_readme_snippets for '{query}': {e}", exc_info=True)

        # GitHub Files
        _do_progress_update("GitHub Files")
        try:
            gh_file_snippets, gh_deps = fetch_github_file_snippets(query, logger)
            all_snippets.extend(gh_file_snippets)
            all_sources.extend(['github_files'] * len(gh_file_snippets))
            discovered_queries.update(gh_deps)
        except Exception as e:
            logger.error(f"Error in fetch_github_file_snippets for '{query}': {e}", exc_info=True)

        # Additional Python sources
        if enhanced_features_working and ADDITIONAL_SOURCES_ENABLED:
            _do_progress_update("Additional Sources")
            try:
                additional_snippets = search_additional_sources(query, logger)
                all_snippets.extend(additional_snippets)
                all_sources.extend(['additional'] * len(additional_snippets))
            except Exception as e:
                logger.error(f"Error in additional sources for '{query}': {e}", exc_info=True)

        # Freelancer-specific Python sources
        if enhanced_features_working and FREELANCER_FEATURES_AVAILABLE and FREELANCER_SOURCES_ENABLED:
            _do_progress_update("Freelancer Sources")
            try:
                freelancer_snippets = search_freelancer_sources(query, logger)
                all_snippets.extend(freelancer_snippets)
                all_sources.extend(['freelancer'] * len(freelancer_snippets))
            except Exception as e:
                logger.error(f"Error in freelancer sources for '{query}': {e}", exc_info=True)

    logger.info(
        f"RAW TOTAL: {len(all_snippets)} snippets gathered for query '{query}' (content_type: {content_type}) before processing.")

    # === ENHANCED PROCESSING WITH LANGUAGE-SPECIFIC HANDLING ===
    if enhanced_features_working and quality_filter and deduplicator:
        # Apply quality filtering and categorization
        if progress_callback:
            progress_callback("Applying quality filters and categorization...", 87)

        if QUALITY_FILTER_ENABLED:
            try:
                enhanced_snippets = []

                # Get language-specific quality settings
                if LANGUAGE_SPECIFIC_QUALITY_FILTERS and content_type in QUALITY_SCORE_WEIGHTS:
                    quality_settings = QUALITY_SCORE_WEIGHTS[content_type]
                    min_score = quality_settings.get('min_score', 3)
                else:
                    min_score = 3

                for i, (snippet, source) in enumerate(zip(all_snippets, all_sources)):
                    # Apply categorization if available
                    if categorizer:
                        if content_type == 'pinescript':
                            categorization = categorizer.categorize_pinescript(snippet)
                            # Apply Pinescript-specific scoring
                            base_score = categorization.get('trading_value_score', 0) + 5
                        else:
                            categorization = categorizer.categorize_snippet(snippet)
                            # Apply Python-specific scoring
                            base_score = categorization.get('freelance_score', 0) + 5

                        # Apply source priority weighting
                        if content_type == 'pinescript' and source in PINESCRIPT_SOURCE_WEIGHTS:
                            source_weight = PINESCRIPT_SOURCE_WEIGHTS[source]
                        else:
                            source_weight = SOURCE_PRIORITY_WEIGHTS.get(source, 1.0)

                        weighted_score = base_score * source_weight

                        enhanced_snippets.append({
                            'code': snippet,
                            'score': weighted_score,
                            'metadata': categorization,
                            'content_type': content_type
                        })
                    else:
                        # Fallback to basic quality scoring
                        result = quality_filter.score_snippet(snippet, source)
                        enhanced_snippets.append({
                            'code': snippet,
                            'score': result['score'],
                            'metadata': result['metadata'],
                            'content_type': content_type
                        })

                # Filter by minimum quality score
                scored_snippets = [s for s in enhanced_snippets if s['score'] >= min_score]

                logger.info(
                    f"Quality filtering ({content_type}): {len(all_snippets)} -> {len(scored_snippets)} snippets (min score: {min_score})")
            except Exception as e:
                logger.error(f"Quality filtering failed: {e}")
                # Fallback without scoring
                scored_snippets = [
                    {'code': snippet, 'score': 5, 'metadata': {'source': src}, 'content_type': content_type}
                    for snippet, src in zip(all_snippets, all_sources)
                ]
        else:
            # Convert to expected format without scoring
            scored_snippets = [
                {'code': snippet, 'score': 5, 'metadata': {'source': src}, 'content_type': content_type}
                for snippet, src in zip(all_snippets, all_sources)
            ]

        # Apply smart deduplication
        if progress_callback:
            progress_callback("Removing duplicates...", 92)

        if SMART_DEDUPLICATION_ENABLED:
            try:
                final_snippets_data = []
                for snippet_data in scored_snippets:
                    if deduplicator.add_snippet(snippet_data['code'], snippet_data.get('metadata', {})):
                        final_snippets_data.append(snippet_data)

                logger.info(
                    f"Smart deduplication ({content_type}): {len(scored_snippets)} -> {len(final_snippets_data)} snippets")
                dedup_stats = deduplicator.get_stats()
                logger.info(f"Deduplication stats: {dedup_stats}")
            except Exception as e:
                logger.error(f"Smart deduplication failed: {e}")
                final_snippets_data = scored_snippets
        else:
            final_snippets_data = scored_snippets

        # Sort by score (highest first) for better results
        final_snippets_data.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Extract just the code for backward compatibility
        unique_snippets_for_query = [item['code'] for item in final_snippets_data]

        # Store the enhanced data for potential RAG export
        logger.enhanced_snippet_data = final_snippets_data

    else:
        # Fallback to original simple deduplication
        unique_snippets_for_query = list(dict.fromkeys(all_snippets))
        logger.info(
            f"Basic deduplication ({content_type}): {len(all_snippets)} -> {len(unique_snippets_for_query)} snippets")

    if progress_callback:
        progress_callback("Search complete!", 100)

    logger.info(
        f"FINAL: {len(unique_snippets_for_query)} unique snippets for query '{query}' (content_type: {content_type}).")
    logger.info(f"Discovered {len(discovered_queries)} potential new search terms.")

    # Log content-type specific insights
    if enhanced_features_working and hasattr(logger, 'enhanced_snippet_data'):
        if content_type == 'pinescript':
            trading_relevant = sum(1 for item in logger.enhanced_snippet_data
                                   if item['metadata'].get('trading_relevant', False))
            high_value = sum(1 for item in logger.enhanced_snippet_data
                             if 'high_value' in item['metadata'].get('client_value', ''))
            logger.info(
                f"Pinescript insights: {trading_relevant} trading-relevant snippets, {high_value} high-value for trading")
        else:
            freelance_relevant = sum(1 for item in logger.enhanced_snippet_data
                                     if item['metadata'].get('freelance_relevant', False))
            high_value = sum(1 for item in logger.enhanced_snippet_data
                             if 'high_value' in item['metadata'].get('client_value', ''))
            logger.info(
                f"Python insights: {freelance_relevant} freelance-relevant snippets, {high_value} high-value for clients")

    return unique_snippets_for_query


# === EXISTING PYTHON-SPECIFIC FUNCTIONS REMAIN UNCHANGED ===
# (All the existing fetch_stdlib_docs, fetch_stackoverflow_snippets, etc. functions remain exactly the same)

def fetch_stdlib_docs(module_name, logger):
    logger.info(f"Fetching Python stdlib docs for: {module_name}")
    url = STDLIB_DOCS_BASE_URL.format(module_name=module_name)
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, timeout=STDLIB_DOCS_TIMEOUT, headers=headers)
        resp.raise_for_status()
        html_content = resp.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch stdlib docs for {module_name}: {e}")
        return []

    snippets = extract_code_from_html_markdown(html_content)
    logger.info(f"Found {len(snippets)} potential snippets from stdlib docs for {module_name}.")
    return snippets


def fetch_stackoverflow_snippets(query, logger, top_n=None):
    # ... (existing function remains unchanged)
    actual_top_n = top_n if top_n is not None else STACKOVERFLOW_SEARCH_MAX_RESULTS
    logger.info(f"Fetching Stack Overflow snippets for query: {query} (top {actual_top_n} questions)")
    discovered_tags = set()

    search_api_url = f"{STACKEXCHANGE_API_BASE_URL}{STACKOVERFLOW_SEARCH_ENDPOINT}"
    headers = {"User-Agent": USER_AGENT}
    params = {
        "order": "desc", "sort": "votes", "tagged": query, "q": query,
        "site": STACKOVERFLOW_SITE_NAME, "pagesize": actual_top_n, "filter": "withbody"
    }
    try:
        resp = requests.get(search_api_url, params=params, timeout=STACKOVERFLOW_SEARCH_TIMEOUT, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Stack Overflow search results for '{query}': {e}")
        return [], []

    data = resp.json()
    snippets = []
    items = data.get("items", [])
    if not items:
        logger.info(f"No questions found on Stack Overflow for query: {query}")
        return snippets, list(discovered_tags)

    for item_idx, item in enumerate(items):
        logger.debug(f"Processing SO Question {item_idx + 1}/{len(items)}: {item.get('title', 'N/A')}")

        if item.get("body"):
            question_page_snippets = extract_code_from_html_markdown(item.get("body", ""))
            for snippet_text in question_page_snippets:
                if snippet_text: snippets.append(snippet_text)

        qid = item.get("question_id")
        if not qid: continue

        ans_api_url = f"{STACKEXCHANGE_API_BASE_URL}{STACKOVERFLOW_ANSWERS_ENDPOINT.format(qid=qid)}"
        ans_params = {
            "order": "desc", "sort": "votes", "site": STACKOVERFLOW_SITE_NAME,
            "filter": "withbody", "pagesize": STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION
        }
        try:
            ans_resp = requests.get(ans_api_url, params=ans_params, timeout=STACKOVERFLOW_ANSWERS_TIMEOUT,
                                    headers=headers)
            ans_resp.raise_for_status()
            ans_data = ans_resp.json()
            for ans_idx, ans in enumerate(ans_data.get("items", [])):
                logger.debug(f"Processing SO Answer {ans_idx + 1}/{len(ans_data.get('items', []))} for QID {qid}")
                if ans.get("body"):
                    answer_page_snippets = extract_code_from_html_markdown(ans.get("body", ""))
                    for snippet_text in answer_page_snippets:
                        if snippet_text: snippets.append(snippet_text)
        except requests.RequestException as e:
            logger.warning(f"Could not fetch answers for Stack Overflow qid {qid}: {e}")
            continue

    logger.info(f"Found {len(snippets)} potential snippets from Stack Overflow for {query}.")
    return snippets, list(discovered_tags)


def fetch_github_readme_snippets(query, logger, max_repos=None, snippets_per_repo=None):
    # ... (existing function remains unchanged)
    actual_max_repos = max_repos if max_repos is not None else GITHUB_README_MAX_REPOS
    actual_snippets_per_repo = snippets_per_repo if snippets_per_repo is not None else GITHUB_README_SNIPPETS_PER_REPO

    logger.info(
        f"Fetching GitHub README snippets for query: {query} (max {actual_max_repos} repos, target {actual_snippets_per_repo} snippets/repo)")
    try:
        token = os.getenv('GITHUB_TOKEN')
        gh = Github(login_or_token=token, timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT) if token else Github(
            timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT)
    except Exception as e:
        logger.error(f"Failed to initialize GitHub API for READMEs: {e}. Using unauthenticated.", exc_info=True)
        gh = Github(timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT)

    snippets = []
    try:
        repositories = gh.search_repositories(query=f"{query} language:python", sort="stars", order="desc")
        repo_count = 0
        for repo in repositories:
            if repo_count >= actual_max_repos:
                logger.info(f"Reached GITHUB_README_MAX_REPOS limit ({actual_max_repos}).")
                break
            try:
                logger.debug(f"Processing README for repo: {repo.full_name} (Stars: {repo.stargazers_count})")
                content_file = repo.get_readme()
                content = content_file.decoded_content.decode("utf-8", errors="ignore")
                readme_page_snippets = extract_code_from_html_markdown(content)

                extracted_from_repo = 0
                for snippet_text in readme_page_snippets:
                    if snippet_text:
                        snippets.append(snippet_text)
                        extracted_from_repo += 1
                        if extracted_from_repo >= actual_snippets_per_repo:
                            break
                logger.info(f"Extracted {extracted_from_repo} snippets from README of {repo.full_name}")
                repo_count += 1
            except GithubException as e:
                if e.status == 404:
                    logger.info(f"README not found for {repo.full_name}. Skipping.")
                elif e.status == 403:
                    logger.error(
                        f"GitHub API error (403 Forbidden) processing README for {repo.full_name}. Check GITHUB_TOKEN and rate limits. Halting README search for this query.")
                    break
                else:
                    logger.warning(f"GitHub API error processing README for {repo.full_name}: {e.status} {e.data}")
            except Exception as e:
                logger.error(f"Error processing README for {repo.full_name}: {e}", exc_info=True)
                continue
    except GithubException as e:
        logger.error(f"GitHub API error during repository search for READMEs for '{query}': {e.status} {e.data}")
        if e.status == 403: logger.error("Ensure GITHUB_TOKEN is set and valid. Rate limit likely hit.")
    except Exception as e:
        logger.error(f"Unexpected error during GitHub README search for '{query}': {e}", exc_info=True)

    logger.info(f"Found {len(snippets)} potential snippets from GitHub READMEs for {query}.")
    return snippets


def fetch_github_file_snippets(query, logger, max_repos=None, files_per_repo_target=None):
    # ... (existing function remains unchanged - this is quite long so I'll note it's preserved)
    # The complete existing implementation remains exactly the same
    pass  # Placeholder - in real implementation, the full existing function would be here

# All other existing helper functions remain unchanged:
# - _get_source_segment
# - _extract_python_constructs
# - _parse_requirements_txt
# etc.