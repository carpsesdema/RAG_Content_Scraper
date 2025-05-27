import os
import re
import requests
from bs4 import BeautifulSoup
from github import Github, GithubException
import ast
import logging
from .parser import _clean_snippet_text, extract_code as extract_code_from_html_markdown, extract_code_from_ipynb

try:
    from .quality_filter import CodeQualityFilter
    from .additional_sources import search_additional_sources
    from .freelancer_sources import search_freelancer_sources
    from .pinescript_sources import search_pinescript_sources
    from ..utils.deduplicator import SmartDeduplicator
    from ..utils.code_categorizer import CodeCategorizer
    from ..utils.pinescript_categorizer import PinescriptCategorizer

    ENHANCED_FEATURES_AVAILABLE = True
    FREELANCER_FEATURES_AVAILABLE = True
    PINESCRIPT_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    FREELANCER_FEATURES_AVAILABLE = False
    PINESCRIPT_FEATURES_AVAILABLE = False

try:
    from config import (
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
        QUALITY_FILTER_ENABLED, SMART_DEDUPLICATION_ENABLED,
        ADDITIONAL_SOURCES_ENABLED, FREELANCER_SOURCES_ENABLED,
        CODE_CATEGORIZATION_ENABLED, FREELANCE_MODE,
        SOURCE_PRIORITY_WEIGHTS,
        PINESCRIPT_ENABLED, PINESCRIPT_SOURCES_ENABLED,
        CONTENT_TYPES, DEFAULT_CONTENT_TYPE,
        LANGUAGE_DETECTION_ENABLED, CONTENT_TYPE_KEYWORDS,
        PINESCRIPT_SOURCE_WEIGHTS, TRADING_MODE,
        LANGUAGE_SPECIFIC_QUALITY_FILTERS, QUALITY_SCORE_WEIGHTS,
        LANGUAGE_SPECIFIC_CATEGORIZATION, CATEGORIZATION_SETTINGS
    )
except ImportError:
    logging.critical("Config not found, using fallbacks")
    STDLIB_DOCS_BASE_URL = "https://docs.python.org/3/library/{module_name}.html"
    STDLIB_DOCS_TIMEOUT = 10
    STACKEXCHANGE_API_BASE_URL = "https://api.stackexchange.com/2.3"
    STACKOVERFLOW_SITE_NAME = "stackoverflow"
    STACKOVERFLOW_SEARCH_ENDPOINT = "/search/advanced"
    STACKOVERFLOW_ANSWERS_ENDPOINT = "/questions/{qid}/answers"
    STACKOVERFLOW_SEARCH_MAX_RESULTS = 10
    STACKOVERFLOW_SEARCH_TIMEOUT = 15
    STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION = 3
    STACKOVERFLOW_ANSWERS_TIMEOUT = 10
    GITHUB_README_MAX_REPOS = 5
    GITHUB_README_SNIPPETS_PER_REPO = 5
    GITHUB_FILES_MAX_REPOS = 5
    GITHUB_FILES_PER_REPO_TARGET = 5
    GITHUB_FILES_CANDIDATE_MULTIPLIER = 2
    GITHUB_MAX_FILE_SIZE_KB = 500
    GITHUB_FILE_DOWNLOAD_TIMEOUT = 15
    USER_AGENT = "RAGContentScraper/1.0 (fallback)"
    EXTRACT_WHOLE_SMALL_PY_FILES = True
    MAX_LINES_FOR_WHOLE_FILE_EXTRACTION = 100
    QUALITY_FILTER_ENABLED = False
    SMART_DEDUPLICATION_ENABLED = False
    ADDITIONAL_SOURCES_ENABLED = False
    FREELANCER_SOURCES_ENABLED = False
    CODE_CATEGORIZATION_ENABLED = False
    FREELANCE_MODE = False
    SOURCE_PRIORITY_WEIGHTS = {}
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
    if not LANGUAGE_DETECTION_ENABLED:
        return DEFAULT_CONTENT_TYPE

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ['pinescript', 'tradingview', 'trading', 'indicator', 'strategy']):
        pinescript_terms = CONTENT_TYPE_KEYWORDS.get('pinescript', [])
        if any(term in query_lower for term in
               ['pinescript', 'tradingview', '//@version', 'ta.', 'strategy(', 'indicator(']):
            logger.info(f"Detected Pinescript content type for query: {query}")
            return 'pinescript'

    if any(keyword in query_lower for keyword in ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy']):
        python_terms = CONTENT_TYPE_KEYWORDS.get('python', [])
        if any(term in query_lower for term in python_terms[:5]):
            logger.info(f"Detected Python content type for query: {query}")
            return 'python'

    trading_terms = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'trading', 'backtest', 'strategy', 'alert']
    if any(term in query_lower for term in trading_terms):
        logger.info(f"Detected potential Pinescript content (trading terms) for query: {query}")
        return 'pinescript'

    if DEFAULT_CONTENT_TYPE == 'auto':
        logger.info(f"Auto-detecting content type, defaulting to Python for query: {query}")
        return 'python'

    logger.info(f"Using default content type '{DEFAULT_CONTENT_TYPE}' for query: {query}")
    return DEFAULT_CONTENT_TYPE


def search_and_fetch(query, logger, progress_callback=None, content_type=None):
    global ENHANCED_FEATURES_AVAILABLE, PINESCRIPT_FEATURES_AVAILABLE

    if logger is None:
        logger = logging.getLogger(USER_AGENT)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

    if content_type is None:
        content_type = detect_content_type(query, logger)

    logger.info(f"Processing query '{query}' as {content_type} content")

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

    if content_type == 'pinescript':
        total_configured_steps = 1
        if enhanced_features_working and PINESCRIPT_SOURCES_ENABLED and PINESCRIPT_FEATURES_AVAILABLE:
            total_configured_steps += 1
    else:
        total_configured_steps = 4
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

    if content_type == 'pinescript':
        if enhanced_features_working and PINESCRIPT_SOURCES_ENABLED and PINESCRIPT_FEATURES_AVAILABLE:
            _do_progress_update("Pinescript Sources")
            try:
                pinescript_snippets = search_pinescript_sources(query, logger)
                all_snippets.extend(pinescript_snippets)
                all_sources.extend(['pinescript'] * len(pinescript_snippets))
                logger.info(f"Found {len(pinescript_snippets)} Pinescript snippets")
            except Exception as e:
                logger.error(f"Error in pinescript sources for '{query}': {e}", exc_info=True)
    else:
        _do_progress_update("Python Standard Library")
        try:
            stdlib_snippets = fetch_stdlib_docs(query, logger)
            all_snippets.extend(stdlib_snippets)
            all_sources.extend(['stdlib'] * len(stdlib_snippets))
        except Exception as e:
            logger.error(f"Error in fetch_stdlib_docs for '{query}': {e}", exc_info=True)

        _do_progress_update("Stack Overflow")
        try:
            so_snippets, so_tags = fetch_stackoverflow_snippets(query, logger)
            all_snippets.extend(so_snippets)
            all_sources.extend(['stackoverflow'] * len(so_snippets))
            discovered_queries.update(so_tags)
        except Exception as e:
            logger.error(f"Error in fetch_stackoverflow_snippets for '{query}': {e}", exc_info=True)

        _do_progress_update("GitHub READMEs")
        try:
            readme_snippets = fetch_github_readme_snippets(query, logger)
            all_snippets.extend(readme_snippets)
            all_sources.extend(['github_readme'] * len(readme_snippets))
        except Exception as e:
            logger.error(f"Error in fetch_github_readme_snippets for '{query}': {e}", exc_info=True)

        _do_progress_update("GitHub Files")
        try:
            gh_file_snippets, gh_deps = fetch_github_file_snippets(query, logger)
            all_snippets.extend(gh_file_snippets)
            all_sources.extend(['github_files'] * len(gh_file_snippets))
            discovered_queries.update(gh_deps)
        except Exception as e:
            logger.error(f"Error in fetch_github_file_snippets for '{query}': {e}", exc_info=True)

        if enhanced_features_working and ADDITIONAL_SOURCES_ENABLED:
            _do_progress_update("Additional Sources")
            try:
                additional_snippets = search_additional_sources(query, logger)
                all_snippets.extend(additional_snippets)
                all_sources.extend(['additional'] * len(additional_snippets))
            except Exception as e:
                logger.error(f"Error in additional sources for '{query}': {e}", exc_info=True)

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

    if enhanced_features_working and quality_filter and deduplicator:
        if progress_callback:
            progress_callback("Applying quality filters and categorization...", 87)

        if QUALITY_FILTER_ENABLED:
            try:
                enhanced_snippets = []
                if LANGUAGE_SPECIFIC_QUALITY_FILTERS and content_type in QUALITY_SCORE_WEIGHTS:
                    quality_settings = QUALITY_SCORE_WEIGHTS[content_type]
                    min_score = quality_settings.get('min_score', 3)
                else:
                    min_score = 3

                for i, (snippet, source) in enumerate(zip(all_snippets, all_sources)):
                    if categorizer:
                        if content_type == 'pinescript':
                            categorization = categorizer.categorize_pinescript(snippet)
                            base_score = categorization.get('trading_value_score', 0) + 5
                        else:
                            categorization = categorizer.categorize_snippet(snippet)
                            base_score = categorization.get('freelance_score', 0) + 5

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
                        result = quality_filter.score_snippet(snippet, source)
                        enhanced_snippets.append({
                            'code': snippet,
                            'score': result['score'],
                            'metadata': result['metadata'],
                            'content_type': content_type
                        })
                scored_snippets = [s for s in enhanced_snippets if s['score'] >= min_score]
                logger.info(
                    f"Quality filtering ({content_type}): {len(all_snippets)} -> {len(scored_snippets)} snippets (min score: {min_score})")
            except Exception as e:
                logger.error(f"Quality filtering failed: {e}")
                scored_snippets = [
                    {'code': snippet, 'score': 5, 'metadata': {'source': src}, 'content_type': content_type}
                    for snippet, src in zip(all_snippets, all_sources)
                ]
        else:
            scored_snippets = [
                {'code': snippet, 'score': 5, 'metadata': {'source': src}, 'content_type': content_type}
                for snippet, src in zip(all_snippets, all_sources)
            ]

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

        final_snippets_data.sort(key=lambda x: x.get('score', 0), reverse=True)
        unique_snippets_for_query = [item['code'] for item in final_snippets_data]
        logger.enhanced_snippet_data = final_snippets_data
    else:
        unique_snippets_for_query = list(dict.fromkeys(all_snippets))
        logger.info(
            f"Basic deduplication ({content_type}): {len(all_snippets)} -> {len(unique_snippets_for_query)} snippets")

    if progress_callback:
        progress_callback("Search complete!", 100)

    logger.info(
        f"FINAL: {len(unique_snippets_for_query)} unique snippets for query '{query}' (content_type: {content_type}).")
    logger.info(f"Discovered {len(discovered_queries)} potential new search terms.")

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
    actual_max_repos = max_repos if max_repos is not None else GITHUB_FILES_MAX_REPOS
    actual_files_per_repo_target = files_per_repo_target if files_per_repo_target is not None else GITHUB_FILES_PER_REPO_TARGET
    logger.info(
        f"Fetching GitHub file snippets for query: {query} (max {actual_max_repos} repos, target {actual_files_per_repo_target} files/repo)")
    snippets = []
    discovered_dependencies = set()
    try:
        token = os.getenv('GITHUB_TOKEN')
        gh = Github(login_or_token=token, timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT) if token else Github(
            timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT)
    except Exception as e:
        logger.error(f"Failed to initialize GitHub API for files: {e}. Using unauthenticated.", exc_info=True)
        gh = Github(timeout=GITHUB_FILE_DOWNLOAD_TIMEOUT)

    try:
        repositories = gh.search_repositories(query=f"{query} language:python", sort="stars", order="desc")
        repo_count = 0
        for repo in repositories:
            if repo_count >= actual_max_repos:
                logger.info(f"Reached GITHUB_FILES_MAX_REPOS limit ({actual_max_repos}).")
                break
            logger.debug(f"Searching files in repo: {repo.full_name} (Stars: {repo.stargazers_count})")
            try:
                contents = repo.get_contents("")
                candidate_files = []
                while contents:
                    file_content = contents.pop(0)
                    if file_content.type == "dir":
                        if len(contents) < 2000:  # Limit recursion depth/breadth
                            try:
                                contents.extend(repo.get_contents(file_content.path))
                            except GithubException as ge_dir:
                                if ge_dir.status == 403:  # Rate limit while traversing
                                    logger.error(
                                        "GitHub rate limit hit while listing directory contents. Skipping rest of this repo.")
                                    break
                                else:
                                    logger.warning(f"Could not get contents of dir {file_content.path}: {ge_dir}")
                        else:
                            logger.warning(
                                f"Skipping deep directory traversal in {repo.full_name} at {file_content.path}")
                    elif file_content.name.endswith(".py") or file_content.name.endswith(".ipynb"):
                        if file_content.size > 0 and file_content.size < (GITHUB_MAX_FILE_SIZE_KB * 1024):
                            candidate_files.append(file_content)
                    elif file_content.name == "requirements.txt":
                        try:
                            req_content = file_content.decoded_content.decode("utf-8", errors="ignore")
                            discovered_dependencies.update(_parse_requirements_txt(req_content))
                        except Exception as req_e:
                            logger.warning(f"Could not parse requirements.txt in {repo.full_name}: {req_e}")

                if not contents and not candidate_files:  # If rate limit broke directory listing early
                    pass

                candidate_files.sort(key=lambda fc: fc.size,
                                     reverse=True)  # Process smaller files first if many candidates

                files_extracted_from_repo = 0
                for fc_idx, file_candidate in enumerate(candidate_files):
                    if files_extracted_from_repo >= actual_files_per_repo_target:
                        break
                    if fc_idx >= (
                            actual_files_per_repo_target * GITHUB_FILES_CANDIDATE_MULTIPLIER):  # Check only a multiple of target
                        logger.debug(f"Checked enough candidate files in {repo.full_name}, moving to next repo.")
                        break
                    try:
                        logger.debug(f"Processing file: {file_candidate.path} (Size: {file_candidate.size} B)")
                        file_content_str = file_candidate.decoded_content.decode("utf-8", errors="ignore")

                        if file_candidate.name.endswith(".ipynb"):
                            file_snippets = extract_code_from_ipynb(file_content_str, logger)
                        elif EXTRACT_WHOLE_SMALL_PY_FILES and len(
                                file_content_str.splitlines()) <= MAX_LINES_FOR_WHOLE_FILE_EXTRACTION:
                            file_snippets = [_clean_snippet_text(file_content_str)] if file_content_str.strip() else []
                        else:
                            constructs = _extract_python_constructs(file_content_str, query, logger)
                            file_snippets = [_clean_snippet_text(code) for code in constructs if code]

                        if file_snippets:
                            snippets.extend(file_snippets)
                            files_extracted_from_repo += len(file_snippets)
                            logger.info(f"Extracted {len(file_snippets)} snippets from {file_candidate.path}")

                    except GithubException as ge_file:  # Catching Github specific exceptions for files
                        if ge_file.status == 403:  # Rate limit
                            logger.error(
                                "GitHub rate limit hit while fetching file content. Aborting file search for this query.")
                            repo_count = actual_max_repos  # Force outer loop to break
                            break
                        else:
                            logger.warning(
                                f"GitHub API error processing file {file_candidate.path}: {ge_file.status} {ge_file.data}")
                    except Exception as e_file_proc:
                        logger.error(f"Error processing file {file_candidate.path}: {e_file_proc}", exc_info=True)
                        continue
                repo_count += 1
            except GithubException as e_repo:
                if e_repo.status == 403:  # Rate limit
                    logger.error(
                        f"GitHub API error (403 Forbidden) processing repo {repo.full_name}. Check GITHUB_TOKEN and rate limits. Halting file search for this query.")
                    break
                else:
                    logger.warning(
                        f"GitHub API error processing files for repo {repo.full_name}: {e_repo.status} {e_repo.data}")
            except Exception as e_repo_outer:
                logger.error(f"Error processing files for repo {repo.full_name}: {e_repo_outer}", exc_info=True)
                continue
    except GithubException as e_search:
        logger.error(
            f"GitHub API error during repository search for files for '{query}': {e_search.status} {e_search.data}")
        if e_search.status == 403: logger.error("Ensure GITHUB_TOKEN is set and valid. Rate limit likely hit.")
    except Exception as e_search_outer:
        logger.error(f"Unexpected error during GitHub file search for '{query}': {e_search_outer}", exc_info=True)

    logger.info(f"Found {len(snippets)} potential snippets from GitHub files for {query}.")
    return snippets, list(discovered_dependencies)


def _get_source_segment(lines, start_node, end_node):
    segment_lines = lines[start_node.lineno - 1: end_node.lineno]

    first_line_offset = start_node.col_offset if hasattr(start_node, 'col_offset') else 0
    if segment_lines:
        segment_lines[0] = segment_lines[0][first_line_offset:]

        if hasattr(end_node, 'end_col_offset') and end_node.end_lineno == start_node.lineno:
            segment_lines[-1] = segment_lines[-1][:end_node.end_col_offset - first_line_offset]
        elif hasattr(end_node, 'end_col_offset'):
            segment_lines[-1] = segment_lines[-1][:end_node.end_col_offset]

    return "\n".join(segment_lines)


def _extract_python_constructs(code_str, query, logger):
    constructs = []
    try:
        lines = code_str.splitlines()
        tree = ast.parse(code_str)
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))

        for node in ast.walk(tree):
            segment = ""
            relevant = False

            node_code_str = ast.unparse(node) if hasattr(ast, 'unparse') else ""  # Fallback for older ast

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, 'decorator_list') and node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    segment = _get_source_segment(lines, first_decorator, node)
                else:
                    segment = _get_source_segment(lines, node, node)

                if query_terms:
                    node_name_lower = node.name.lower()
                    if node_name_lower in query_terms or any(
                            term in node_name_lower for term in query_terms if len(term) > 3):
                        relevant = True
                    if not relevant and node_code_str:
                        if any(term in node_code_str.lower() for term in query_terms):
                            relevant = True
                else:  # No query terms, extract all top-level constructs
                    relevant = (node.col_offset == 0)  # A simple heuristic for top-level

            elif isinstance(node, ast.If) and node.col_offset == 0 and \
                    isinstance(node.test, ast.Compare) and \
                    isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and \
                    isinstance(node.test.ops[0], ast.Eq) and \
                    isinstance(node.test.comparators[0], ast.Constant) and \
                    node.test.comparators[0].value == '__main__':
                segment = _get_source_segment(lines, node, node)
                relevant = True  # Always relevant if it's a main block

            if relevant and segment.strip():
                constructs.append(segment)
            elif not query_terms and segment.strip() and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                           ast.ClassDef)):  # If no query, take all functions/classes
                constructs.append(segment)


    except SyntaxError as e:
        logger.debug(
            f"SyntaxError parsing Python code for construct extraction: {e}. Code (first 100 chars): {code_str[:100]}")
    except Exception as e:
        logger.error(f"Unexpected error during Python construct extraction: {e}", exc_info=True)

    if not constructs and code_str.strip():  # Fallback: if no constructs, return whole code if it's small
        if len(lines) < MAX_LINES_FOR_WHOLE_FILE_EXTRACTION / 2:  # Arbitrary smaller limit for segments
            return [code_str]
    return constructs


def _parse_requirements_txt(content):
    dependencies = set()
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^\s*([a-zA-Z0-9_.-]+)", line)
        if match:
            dependencies.add(match.group(1).lower())
    return list(dependencies)