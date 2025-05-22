# scraper/searcher.py - Enhanced Max Power Edition!
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
    from ..utils.deduplicator import SmartDeduplicator

    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

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
        # Crawler settings
        GITHUB_CRAWL_PARSE_DEPENDENCIES, STACKOVERFLOW_CRAWL_EXPLORE_TAGS,
        CRAWLER_ENABLED
    )
except ImportError:
    logging.critical(
        "CRITICAL: config.py not found or essential settings are missing. Scraper may not function correctly. Using emergency fallbacks.")
    # Emergency fallbacks - these are conservative to avoid overwhelming APIs if config is gone
    STDLIB_DOCS_BASE_URL = "https://docs.python.org/3/library/{module_name}.html"
    STDLIB_DOCS_TIMEOUT = 10
    STACKEXCHANGE_API_BASE_URL = "https://api.stackexchange.com/2.3"
    STACKOVERFLOW_SITE_NAME = "stackoverflow"
    STACKOVERFLOW_SEARCH_ENDPOINT = "/search/advanced"
    STACKOVERFLOW_ANSWERS_ENDPOINT = "/questions/{qid}/answers"
    STACKOVERFLOW_SEARCH_MAX_RESULTS = 3
    STACKOVERFLOW_SEARCH_TIMEOUT = 15
    STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION = 2
    STACKOVERFLOW_ANSWERS_TIMEOUT = 10
    GITHUB_README_MAX_REPOS = 2
    GITHUB_README_SNIPPETS_PER_REPO = 3
    GITHUB_FILES_MAX_REPOS = 1
    GITHUB_FILES_PER_REPO_TARGET = 2
    GITHUB_FILES_CANDIDATE_MULTIPLIER = 1.5
    GITHUB_MAX_FILE_SIZE_KB = 500
    GITHUB_FILE_DOWNLOAD_TIMEOUT = 15
    USER_AGENT = "RAGContentScraper/1.1-emergency-fallback"
    EXTRACT_WHOLE_SMALL_PY_FILES = False
    MAX_LINES_FOR_WHOLE_FILE_EXTRACTION = 50
    # Crawler fallbacks
    CRAWLER_ENABLED = False
    GITHUB_CRAWL_PARSE_DEPENDENCIES = False
    STACKOVERFLOW_CRAWL_EXPLORE_TAGS = False


def _get_source_segment(source_lines, node):
    """Extracts the source code segment for an AST node."""
    if not hasattr(node, 'end_lineno') or node.end_lineno is None:
        end_line_fallback = node.lineno + 5
        if hasattr(node, 'body') and isinstance(node.body, list) and node.body:
            last_body_item = node.body[-1]
            if hasattr(last_body_item, 'end_lineno') and last_body_item.end_lineno is not None:
                end_line_fallback = last_body_item.end_lineno
            elif hasattr(last_body_item, 'lineno') and last_body_item.lineno is not None:
                end_line_fallback = last_body_item.lineno + 2
        end_line_fallback = min(end_line_fallback, len(source_lines))
        return "\n".join(source_lines[node.lineno - 1: end_line_fallback])

    start_line_idx = node.lineno - 1
    end_line_idx = node.end_lineno

    if start_line_idx < 0 or end_line_idx > len(source_lines) or start_line_idx >= end_line_idx:
        problematic_segment = "\n".join(source_lines[max(0, node.lineno - 1): min(len(source_lines), node.lineno + 2)])
        return f"# Could not determine full segment (L{node.lineno}), partial extract:\n{problematic_segment}"
    return "\n".join(source_lines[start_line_idx:end_line_idx])


def _extract_python_constructs(source_code, logger):
    """
    Parses Python source code using AST and extracts functions/classes.
    Includes constructs even if they don't have docstrings.
    """
    constructs = []
    try:
        tree = ast.parse(source_code)
        source_lines = source_code.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                snippet = _get_source_segment(source_lines, node)
                if snippet:
                    constructs.append(snippet)
    except SyntaxError as e:
        logger.warning(f"AST parsing failed for a Python file (likely not valid Python or partial code): {e}")
    except Exception as e:
        logger.error(f"Unexpected error during AST processing: {e}", exc_info=True)
    return constructs


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
        if STACKOVERFLOW_CRAWL_EXPLORE_TAGS and CRAWLER_ENABLED:
            for tag in item.get("tags", []):
                if tag.lower() != query.lower() and len(tag) > 1:
                    discovered_tags.add(tag)

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


def _parse_requirements_txt(content, logger):
    """Rudimentary parsing of requirements.txt content."""
    dependencies = set()
    if not content:
        return list(dependencies)
    try:
        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if not line or line.startswith('-'):
                continue
            match = re.match(r"^[a-zA-Z0-9_.-]+", line)
            if match:
                dep_name = match.group(0)
                if len(dep_name) > 2:
                    dependencies.add(dep_name)
    except Exception as e:
        logger.warning(f"Error parsing requirements.txt content: {e}")
    return list(dependencies)


def fetch_github_file_snippets(query, logger, max_repos=None, files_per_repo_target=None):
    """
    Search GitHub repos for `query` and extract code constructs from top .py files.
    Now more aggressive, can extract whole small files, and discover dependencies.
    """
    actual_max_repos = max_repos if max_repos is not None else GITHUB_FILES_MAX_REPOS
    actual_files_per_repo_target = files_per_repo_target if files_per_repo_target is not None else GITHUB_FILES_PER_REPO_TARGET
    discovered_dependencies = set()

    logger.info(
        f"Fetching GitHub file snippets for query: '{query}' (max_repos={actual_max_repos}, target_constructs/repo={actual_files_per_repo_target})")
    try:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            logger.warning("GITHUB_TOKEN not set. GitHub API requests will be severely rate-limited.")
        gh = Github(login_or_token=token, timeout=max(15, GITHUB_FILE_DOWNLOAD_TIMEOUT)) if token else Github(
            timeout=max(15, GITHUB_FILE_DOWNLOAD_TIMEOUT))
    except Exception as e:
        logger.error(f"Failed to initialize GitHub API for file search: {e}. Using unauthenticated.", exc_info=True)
        gh = Github(timeout=max(15, GITHUB_FILE_DOWNLOAD_TIMEOUT))

    snippets = []
    try:
        search_query_gh = f"{query} language:python"
        logger.debug(f"GitHub repository search query: '{search_query_gh}'")
        repositories = gh.search_repositories(query=search_query_gh, sort="stars", order="desc")

        repo_count = 0
        total_snippets_from_files = 0

        for repo in repositories:
            if repo_count >= actual_max_repos:
                logger.info(f"Reached GITHUB_FILES_MAX_REPOS limit ({actual_max_repos}).")
                break

            logger.info(
                f"Processing repo {repo_count + 1}/{actual_max_repos}: {repo.full_name} (Stars: {repo.stargazers_count})")
            try:
                if GITHUB_CRAWL_PARSE_DEPENDENCIES and CRAWLER_ENABLED:
                    try:
                        req_file_content = repo.get_contents("requirements.txt")
                        if req_file_content and not isinstance(req_file_content, list):
                            req_text = req_file_content.decoded_content.decode("utf-8", errors="ignore")
                            deps = _parse_requirements_txt(req_text, logger)
                            if deps:
                                logger.info(f"Discovered dependencies from {repo.full_name}/requirements.txt: {deps}")
                                discovered_dependencies.update(deps)
                    except GithubException as e_req:
                        if e_req.status == 404:
                            logger.debug(f"No requirements.txt found in root of {repo.full_name}.")
                        else:
                            logger.warning(f"Could not fetch requirements.txt from {repo.full_name}: {e_req.status}")
                    except Exception as e_parse_req:
                        logger.warning(f"Error processing requirements.txt from {repo.full_name}: {e_parse_req}")

                root_contents = repo.get_contents("")
                py_files_candidates = []
                candidate_search_limit_per_repo = actual_files_per_repo_target * GITHUB_FILES_CANDIDATE_MULTIPLIER * 5
                common_code_dirs = [
                    'example', 'examples', 'sample', 'samples', 'demo', 'demos', 'notebook', 'notebooks',
                    'tutorial', 'tutorials', 'guide', 'guides', 'howto',
                    'test', 'tests', 'tests_integration', 'tests_unit', 'test_cases',
                    'doc', 'docs',
                    'src', 'lib', repo.name.lower().replace('-', '_').replace(' ', '_'),
                    'apps', 'app', 'application', 'cmd', 'cli', 'script', 'scripts', 'tool', 'tools', 'util', 'utils',
                    'utility', 'core', 'pkg'
                ]
                queue = [("", root_contents)]
                visited_dirs = set()
                MAX_DIRS_TO_SCAN_PER_REPO = 40
                MAX_FILES_TO_CONSIDER_PER_REPO = 250
                MAX_SCAN_DEPTH = 3
                dirs_scanned = 0
                files_considered = 0

                while queue and dirs_scanned < MAX_DIRS_TO_SCAN_PER_REPO and files_considered < candidate_search_limit_per_repo:
                    current_path_prefix, current_contents = queue.pop(0)
                    current_depth = current_path_prefix.count('/') if current_path_prefix else 0
                    if current_depth > MAX_SCAN_DEPTH:
                        logger.debug(f"Max scan depth ({MAX_SCAN_DEPTH}) reached for path: {current_path_prefix}")
                        continue
                    if current_path_prefix in visited_dirs and current_path_prefix != "":
                        continue
                    visited_dirs.add(current_path_prefix)
                    dirs_scanned += 1

                    for content_file in current_contents:
                        if files_considered >= candidate_search_limit_per_repo: break
                        files_considered += 1
                        if content_file.type == 'dir':
                            dir_name_lower = content_file.name.lower()
                            is_common_code_dir = any(common_dir in dir_name_lower for common_dir in common_code_dirs)
                            is_short_generic_name = len(dir_name_lower) <= 4
                            if is_common_code_dir or is_short_generic_name or not current_path_prefix or current_depth < MAX_SCAN_DEPTH - 1:
                                if content_file.path not in visited_dirs and dirs_scanned < MAX_DIRS_TO_SCAN_PER_REPO:
                                    try:
                                        logger.debug(
                                            f"Queueing dir for scan: {content_file.path} (depth {current_depth + 1})")
                                        queue.append((content_file.path, repo.get_contents(content_file.path)))
                                    except GithubException as e_gh_dir:
                                        logger.warning(
                                            f"Could not get contents for dir {content_file.path}: {e_gh_dir.status}")
                                    except Exception as e_dir_generic:
                                        logger.warning(
                                            f"Error getting contents for dir {content_file.path}: {e_dir_generic}")
                        elif content_file.type == 'file' and content_file.path.endswith((".py", ".ipynb")):
                            priority = 5
                            path_lower = content_file.path.lower()
                            if any(d in path_lower for d in
                                   ['example', 'examples', 'sample', 'samples', 'demo', 'demos', 'tutorial', 'notebook',
                                    'notebooks']):
                                priority = 0
                            elif any(d in path_lower for d in ['test', 'tests']):
                                priority = 1
                            elif any(d in path_lower for d in
                                     ['src', 'lib', repo.name.lower().replace('-', '_').replace(' ', '_'), 'core',
                                      'pkg']):
                                priority = 2
                            elif any(d in path_lower for d in ['script', 'scripts', 'tool', 'tools', 'util', 'utils']):
                                priority = 3
                            elif '__init__.py' in path_lower:
                                priority = 6 if content_file.size > 1024 else 4

                            if content_file.size > (GITHUB_MAX_FILE_SIZE_KB * 1024):
                                logger.debug(
                                    f"Skipping large file: {content_file.path} ({content_file.size} bytes > {GITHUB_MAX_FILE_SIZE_KB}KB)")
                                continue
                            if content_file.size == 0 and not content_file.path.endswith(".ipynb"):
                                logger.debug(f"Skipping empty .py file: {content_file.path}")
                                continue
                            py_files_candidates.append({
                                'path': content_file.path, 'priority': priority,
                                'url': content_file.download_url, 'size': content_file.size,
                                'is_notebook': content_file.path.endswith(".ipynb")
                            })
                    if files_considered >= candidate_search_limit_per_repo: break

                logger.debug(
                    f"Found {len(py_files_candidates)} candidate files in {repo.full_name} after scanning {dirs_scanned} dirs.")
                py_files_candidates.sort(key=lambda x: (x['priority'], -x['size']))

                extracted_constructs_from_repo = 0
                files_processed_this_repo = 0
                MAX_FILES_TO_DOWNLOAD_PER_REPO = max(25, int(actual_files_per_repo_target * 1.5))

                for file_info in py_files_candidates:
                    if files_processed_this_repo >= MAX_FILES_TO_DOWNLOAD_PER_REPO:
                        logger.info(
                            f"Reached download limit ({MAX_FILES_TO_DOWNLOAD_PER_REPO}) for repo {repo.full_name}.")
                        break
                    if extracted_constructs_from_repo >= actual_files_per_repo_target * 2 and actual_files_per_repo_target > 0:
                        logger.info(
                            f"Well exceeded file target ({actual_files_per_repo_target * 2}) for repo {repo.full_name}. Moving on.")
                        break

                    files_processed_this_repo += 1
                    try:
                        logger.debug(
                            f"Processing file {files_processed_this_repo}/{MAX_FILES_TO_DOWNLOAD_PER_REPO}: {file_info['path']} (Prio: {file_info['priority']}, Size: {file_info.get('size', 'N/A')})")
                        if not file_info['url']:
                            logger.warning(f"No download_url for {file_info['path']}, skipping.")
                            continue

                        headers = {"User-Agent": USER_AGENT}
                        file_timeout = max(10, GITHUB_FILE_DOWNLOAD_TIMEOUT)
                        raw_resp = requests.get(file_info['url'], timeout=file_timeout, headers=headers)
                        raw_resp.raise_for_status()
                        raw_code = raw_resp.text

                        file_specific_snippets = []
                        if file_info['is_notebook']:
                            notebook_codes = extract_code_from_ipynb(raw_code, logger)
                            for code_cell in notebook_codes:
                                cleaned_cell = _clean_snippet_text(code_cell)
                                if cleaned_cell: file_specific_snippets.append(cleaned_cell)
                        else:
                            ast_constructs = _extract_python_constructs(raw_code, logger)
                            if ast_constructs:
                                for construct_code in ast_constructs:
                                    cleaned_construct = _clean_snippet_text(construct_code)
                                    if cleaned_construct: file_specific_snippets.append(cleaned_construct)
                            if not ast_constructs and EXTRACT_WHOLE_SMALL_PY_FILES:
                                num_lines = len(raw_code.splitlines())
                                if num_lines > 0 and num_lines <= MAX_LINES_FOR_WHOLE_FILE_EXTRACTION:
                                    logger.info(
                                        f"No AST constructs in {file_info['path']} ({num_lines} lines), extracting whole file content.")
                                    cleaned_whole_file = _clean_snippet_text(raw_code)
                                    if cleaned_whole_file: file_specific_snippets.append(cleaned_whole_file)
                                elif num_lines == 0:
                                    logger.debug(f"File {file_info['path']} is empty, skipping whole file extraction.")
                                else:
                                    logger.debug(
                                        f"File {file_info['path']} ({num_lines} lines) is too large for whole file extraction, and no AST constructs found.")

                        if file_specific_snippets:
                            snippets.extend(file_specific_snippets)
                            extracted_constructs_from_repo += len(file_specific_snippets)
                            total_snippets_from_files += len(file_specific_snippets)
                            logger.debug(
                                f"Added {len(file_specific_snippets)} snippets from {file_info['path']}. Repo total: {extracted_constructs_from_repo}.")
                        else:
                            logger.debug(f"No snippets extracted from {file_info['path']}.")
                    except requests.RequestException as e:
                        logger.warning(f"Could not download/process file {file_info['path']}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_info['path']} from {repo.full_name}: {e}",
                                     exc_info=True)

                logger.info(
                    f"Extracted {extracted_constructs_from_repo} total code constructs/snippets from repo {repo.full_name}.")
                repo_count += 1
            except GithubException as e_gh_repo:
                logger.warning(f"GitHub API error for repo {repo.full_name}: {e_gh_repo.status} {e_gh_repo.data}")
                if e_gh_repo.status == 403:
                    logger.error(
                        "GitHub API rate limit hit (403 Forbidden) while processing repo. Halting GitHub file search for this query.")
                    break
            except Exception as e_repo_generic:
                logger.error(f"Error processing repo {repo.full_name}: {e_repo_generic}", exc_info=True)
                continue
    except GithubException as e_gh_search:
        logger.error(
            f"GitHub API error during initial repository search for '{query}': {e_gh_search.status} {e_gh_search.data}")
        if e_gh_search.status == 403: logger.error(
            "Ensure GITHUB_TOKEN is set and valid. Rate limit may have been hit.")
    except Exception as e_search_generic:
        logger.error(f"Unexpected error during GitHub file search for '{query}': {e_search_generic}", exc_info=True)

    logger.info(
        f"AGGREGATE: Found {total_snippets_from_files} total snippets from all GitHub files for query '{query}'.")
    return snippets, list(discovered_dependencies)


def search_and_fetch(query, logger, progress_callback=None):
    """Enhanced search with quality filtering and smart deduplication."""

    global ENHANCED_FEATURES_AVAILABLE

    if logger is None:
        logger = logging.getLogger(USER_AGENT)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

    # Initialize enhanced components if available
    enhanced_features_working = ENHANCED_FEATURES_AVAILABLE
    quality_filter = None
    deduplicator = None

    if ENHANCED_FEATURES_AVAILABLE:
        try:
            quality_filter = CodeQualityFilter()
            deduplicator = SmartDeduplicator()
        except Exception as e:
            logger.warning(f"Could not initialize enhanced features: {e}")
            enhanced_features_working = False

    all_snippets = []
    all_sources = []
    discovered_queries = set()

    # Calculate total steps for progress
    total_configured_steps = 4  # Your existing 4 sources

    # Check if additional sources are enabled
    additional_sources_enabled = False
    try:
        from config import ADDITIONAL_SOURCES_ENABLED
        additional_sources_enabled = ADDITIONAL_SOURCES_ENABLED
    except ImportError:
        additional_sources_enabled = False

    if enhanced_features_working and additional_sources_enabled:
        total_configured_steps += 1  # Add step for additional sources

    current_step_for_progress = 0

    def _do_progress_update(source_name):
        nonlocal current_step_for_progress
        current_step_for_progress += 1
        if progress_callback:
            percentage = int((current_step_for_progress / total_configured_steps) * 85)  # Leave 15% for post-processing
            progress_callback(f"Query '{query}': Fetching from {source_name}...", percentage)

    # === EXISTING SOURCES ===

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
        if STACKOVERFLOW_CRAWL_EXPLORE_TAGS and CRAWLER_ENABLED:
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
        if GITHUB_CRAWL_PARSE_DEPENDENCIES and CRAWLER_ENABLED:
            discovered_queries.update(gh_deps)
    except Exception as e:
        logger.error(f"Error in fetch_github_file_snippets for '{query}': {e}", exc_info=True)

    # === NEW ENHANCED SOURCES ===
    if enhanced_features_working and additional_sources_enabled:
        _do_progress_update("Additional Sources")
        try:
            additional_snippets = search_additional_sources(query, logger)
            all_snippets.extend(additional_snippets)
            all_sources.extend(['additional'] * len(additional_snippets))
        except Exception as e:
            logger.error(f"Error in additional sources for '{query}': {e}", exc_info=True)

    logger.info(f"RAW TOTAL: {len(all_snippets)} snippets gathered for query '{query}' before processing.")

    # === ENHANCED PROCESSING ===
    if enhanced_features_working and quality_filter and deduplicator:
        # Apply quality filtering
        if progress_callback:
            progress_callback("Applying quality filters...", 87)

        # Check if quality filtering is enabled
        quality_filter_enabled = True
        try:
            from config import QUALITY_FILTER_ENABLED
            quality_filter_enabled = QUALITY_FILTER_ENABLED
        except ImportError:
            quality_filter_enabled = True

        if quality_filter_enabled:
            try:
                scored_snippets = quality_filter.filter_snippets(all_snippets, all_sources)
                logger.info(f"Quality filtering: {len(all_snippets)} -> {len(scored_snippets)} snippets")
            except Exception as e:
                logger.error(f"Quality filtering failed: {e}")
                # Fallback without scoring
                scored_snippets = [
                    {'code': snippet, 'score': 5, 'metadata': {'source': src}}
                    for snippet, src in zip(all_snippets, all_sources)
                ]
        else:
            # Convert to expected format without scoring
            scored_snippets = [
                {'code': snippet, 'score': 5, 'metadata': {'source': src}}
                for snippet, src in zip(all_snippets, all_sources)
            ]

        # Apply smart deduplication
        if progress_callback:
            progress_callback("Removing duplicates...", 92)

        # Check if smart deduplication is enabled
        smart_dedup_enabled = True
        try:
            from config import SMART_DEDUPLICATION_ENABLED
            smart_dedup_enabled = SMART_DEDUPLICATION_ENABLED
        except ImportError:
            smart_dedup_enabled = True

        if smart_dedup_enabled:
            try:
                final_snippets_data = []
                for snippet_data in scored_snippets:
                    if deduplicator.add_snippet(snippet_data['code'], snippet_data.get('metadata', {})):
                        final_snippets_data.append(snippet_data)

                logger.info(f"Smart deduplication: {len(scored_snippets)} -> {len(final_snippets_data)} snippets")
                dedup_stats = deduplicator.get_stats()
                logger.info(f"Deduplication stats: {dedup_stats}")
            except Exception as e:
                logger.error(f"Smart deduplication failed: {e}")
                final_snippets_data = scored_snippets
        else:
            final_snippets_data = scored_snippets

        # Extract just the code for backward compatibility
        unique_snippets_for_query = [item['code'] for item in final_snippets_data]

        # Store the enhanced data for potential RAG export
        logger.enhanced_snippet_data = final_snippets_data

    else:
        # Fallback to original simple deduplication
        unique_snippets_for_query = list(dict.fromkeys(all_snippets))
        logger.info(f"Basic deduplication: {len(all_snippets)} -> {len(unique_snippets_for_query)} snippets")

    if progress_callback:
        progress_callback("Search complete!", 100)

    logger.info(f"FINAL: {len(unique_snippets_for_query)} unique snippets for query '{query}'.")
    logger.info(f"Discovered {len(discovered_queries)} potential new search terms.")

    return unique_snippets_for_query