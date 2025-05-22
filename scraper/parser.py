# scraper/parser.py

import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse # For extract_relevant_links


def _clean_snippet_text(snippet_text):  # Renamed to avoid clash if searcher is imported elsewhere
    """Basic cleaning of common artifacts from snippets."""
    if not isinstance(snippet_text, str):
        return ""
    lines = snippet_text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.startswith(">>> "):
            cleaned_lines.append(line[4:])
        elif line.startswith("... "):
            cleaned_lines.append(line[4:])
        else:
            cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'\bCopy code\b', '', cleaned_text, flags=re.IGNORECASE).strip()
    return cleaned_text


def extract_code(html):
    """
    Extracts code snippets from HTML content. Returns a list of code snippet strings.
    """
    snippets = []
    # Markdown fenced code blocks
    # This regex tries to capture content within ```, optionally with a language hint
    fence_pattern = re.compile(r'```(?:[a-zA-Z0-9_+-]*\n)?(.*?)```', re.S)
    for match in fence_pattern.findall(html):
        snippet = match.strip()
        if snippet:
            snippets.append(_clean_snippet_text(snippet))

    # HTML <pre> tags
    soup = BeautifulSoup(html, "html.parser")
    for pre_tag in soup.find_all("pre"):
        # Check if this <pre> tag is inside a Markdown-style block already processed
        # This is a simple check; more robust might involve tracking exact source positions
        is_inside_fence = False
        parent = pre_tag.parent
        while parent:
            if parent.name == 'div' and parent.attrs.get('class') and \
                 any('highlight' in c for c in
                        parent.attrs.get('class', [])):  # Common for syntax highlighted Markdown
                is_inside_fence = True
                break
            parent = parent.parent

        if is_inside_fence:
            continue

        text = pre_tag.get_text().strip()
        cleaned_text = _clean_snippet_text(text)
        if cleaned_text and cleaned_text not in snippets:  # Avoid adding if already found (e.g. from fence)
            snippets.append(cleaned_text)

    # Look for <code> tags not inside <pre> as they might be inline code,
    # but only if they are multi-line or sufficiently long to be considered a "snippet"
    for code_tag in soup.find_all("code"):
        if code_tag.find_parent("pre"):  # Already handled by <pre> tag processing
            continue

        text = code_tag.get_text().strip()
        if '\n' in text or len(text) > 40:  # If it's multi-line or a substantial single line
            cleaned_text = _clean_snippet_text(text)
            if cleaned_text and cleaned_text not in snippets:
                snippets.append(cleaned_text)

    return list(dict.fromkeys(snippets))  # Final deduplication


def extract_code_from_ipynb(json_string, logger):
    """
    Placeholder function to extract code from Jupyter Notebook (.ipynb) files.
    Actual implementation would parse the JSON, find code cells, and extract their source.
    """
    logger.info("extract_code_from_ipynb is a placeholder and needs to be implemented.")
    # Example structure (needs robust implementation):
    # import json
    # codes = []
    # try:
    #     notebook = json.loads(json_string)
    #     for cell in notebook.get('cells', []):
    #         if cell.get('cell_type') == 'code':
    #             source = "".join(cell.get('source', []))
    #             if source.strip():
    #                 codes.append(source.strip())
    # except Exception as e:
    #     logger.error(f"Error parsing IPYNB content: {e}")
    # return codes
    return []


def extract_relevant_links(html_content, base_url, allowed_domains=None, logger=None):
    """
    Extracts relevant hyperlinks from HTML content.

    Args:
        html_content (str): The HTML content to parse.
        base_url (str): The base URL of the page, used to resolve relative links
                        and determine the primary domain.
        allowed_domains (list, optional): A list of domains to restrict links to.
                                           If None or empty, defaults to the domain of base_url.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        list: A list of unique, absolute URLs.
    """
    links = set()
    if not html_content or not base_url:
        return []

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        parsed_base_url = urlparse(base_url)
        base_domain = parsed_base_url.netloc

        if not allowed_domains:
            effective_allowed_domains = [base_domain]
        else:
            effective_allowed_domains = allowed_domains + [base_domain] # Ensure base domain is always allowed

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            try:
                # Join the (potentially relative) href with the base_url to get an absolute URL
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)

                # Basic filter: ensure it's http/https and has a network location (domain)
                if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                    continue

                # Filter by domain
                if parsed_url.netloc in effective_allowed_domains:
                    # Avoid #-fragments that point to the same page content
                    links.add(absolute_url.split('#')[0])

            except Exception as e:
                if logger:
                    logger.debug(f"Could not process link '{href}' from base '{base_url}': {e}")
                continue
        return list(links)
    except Exception as e:
        if logger:
            logger.error(f"Error parsing HTML for links (base URL: {base_url}): {e}")
        return []