# scraper/additional_sources.py

import requests
import json
from bs4 import BeautifulSoup
from typing import List, Tuple
import re
from .parser import extract_code as extract_code_from_html_markdown


class AdditionalPythonSources:
    """Scrape additional Python-specific sources."""

    def __init__(self, user_agent: str, timeout: int = 15):
        self.user_agent = user_agent
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

    def fetch_real_python_snippets(self, query: str, logger) -> List[str]:
        """Fetch code snippets from Real Python articles."""
        logger.info(f"Searching Real Python for: {query}")
        snippets = []

        try:
            # Real Python search
            search_url = f"https://realpython.com/search/"
            params = {'q': query}

            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/lessons/' in href or '/tutorials/' in href:
                    if not href.startswith('http'):
                        href = 'https://realpython.com' + href
                    article_links.append(href)

            # Limit to first 3 articles to avoid overwhelming
            for article_url in article_links[:3]:
                try:
                    article_response = self.session.get(article_url, timeout=self.timeout)
                    article_response.raise_for_status()

                    article_soup = BeautifulSoup(article_response.text, 'html.parser')

                    # Extract code blocks
                    for code_block in article_soup.find_all(['pre', 'code']):
                        code_text = code_block.get_text().strip()
                        if len(code_text) > 50 and '\n' in code_text:  # Multi-line code
                            snippets.append(code_text)

                except Exception as e:
                    logger.warning(f"Error fetching Real Python article {article_url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching Real Python: {e}")

        logger.info(f"Found {len(snippets)} snippets from Real Python")
        return snippets

    def fetch_pypi_examples(self, package_name: str, logger) -> List[str]:
        """Fetch usage examples from PyPI package pages."""
        logger.info(f"Fetching PyPI examples for: {package_name}")
        snippets = []

        try:
            # PyPI API
            api_url = f"https://pypi.org/pypi/{package_name}/json"
            response = self.session.get(api_url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Extract description which often contains usage examples
            description = data.get('info', {}).get('description', '')
            if description:
                # Look for code blocks in markdown
                code_blocks = re.findall(r'```python\n(.*?)\n```', description, re.DOTALL)
                code_blocks.extend(re.findall(r'```\n(.*?)\n```', description, re.DOTALL))

                for block in code_blocks:
                    if block.strip() and len(block.strip()) > 20:
                        snippets.append(block.strip())

            # Also check project URLs for documentation
            project_urls = data.get('info', {}).get('project_urls', {})
            docs_url = project_urls.get('Documentation') or project_urls.get('Homepage')

            if docs_url and 'readthedocs' in docs_url:
                try:
                    docs_response = self.session.get(docs_url, timeout=self.timeout)
                    docs_response.raise_for_status()
                    docs_soup = BeautifulSoup(docs_response.text, 'html.parser')

                    # Look for code examples in documentation
                    for code_elem in docs_soup.find_all(['pre', 'code']):
                        code_text = code_elem.get_text().strip()
                        if 'import' in code_text and len(code_text) > 30:
                            snippets.append(code_text)

                except Exception as e:
                    logger.debug(f"Could not fetch docs from {docs_url}: {e}")

        except Exception as e:
            logger.error(f"Error fetching PyPI info for {package_name}: {e}")

        logger.info(f"Found {len(snippets)} snippets from PyPI for {package_name}")
        return snippets

    def fetch_python_org_examples(self, query: str, logger) -> List[str]:
        """Fetch examples from python.org documentation and PEPs."""
        logger.info(f"Searching python.org for: {query}")
        snippets = []

        try:
            # Search Python.org
            search_url = "https://docs.python.org/3/search.html"
            params = {'q': query}

            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find documentation links
            doc_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/library/' in href or '/tutorial/' in href:
                    if not href.startswith('http'):
                        href = 'https://docs.python.org' + href
                    doc_links.append(href)

            # Process first few documentation pages
            for doc_url in doc_links[:2]:
                try:
                    doc_response = self.session.get(doc_url, timeout=self.timeout)
                    doc_response.raise_for_status()

                    doc_soup = BeautifulSoup(doc_response.text, 'html.parser')

                    # Extract code examples
                    for pre_tag in doc_soup.find_all('pre'):
                        code_text = pre_tag.get_text().strip()
                        # Filter for Python code (contains common Python patterns)
                        if any(pattern in code_text for pattern in ['import ', 'def ', 'class ', 'if __name__']):
                            if len(code_text) > 30:
                                snippets.append(code_text)

                except Exception as e:
                    logger.warning(f"Error fetching python.org doc {doc_url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching python.org: {e}")

        logger.info(f"Found {len(snippets)} snippets from python.org")
        return snippets

    def fetch_awesome_python_repos(self, category: str, logger) -> List[str]:
        """Fetch examples from Awesome Python list repositories."""
        logger.info(f"Fetching from Awesome Python category: {category}")
        snippets = []

        # This would require GitHub API integration
        # Placeholder for now - you could integrate with your existing GitHub functionality
        logger.info("Awesome Python integration - placeholder")

        return snippets


# Integration function to add to searcher.py
def search_additional_sources(query: str, logger) -> List[str]:
    """Search additional Python-specific sources."""
    try:
        from config import USER_AGENT, DEFAULT_REQUEST_TIMEOUT
    except ImportError:
        USER_AGENT = "RAGContentScraper/1.0"
        DEFAULT_REQUEST_TIMEOUT = 15

    additional_sources = AdditionalPythonSources(USER_AGENT, DEFAULT_REQUEST_TIMEOUT)
    all_snippets = []

    # Real Python
    try:
        real_python_snippets = additional_sources.fetch_real_python_snippets(query, logger)
        all_snippets.extend(real_python_snippets)
    except Exception as e:
        logger.error(f"Error in Real Python search: {e}")

    # PyPI (if query looks like a package name)
    if query.replace('-', '').replace('_', '').isalnum() and len(query) > 2:
        try:
            pypi_snippets = additional_sources.fetch_pypi_examples(query, logger)
            all_snippets.extend(pypi_snippets)
        except Exception as e:
            logger.error(f"Error in PyPI search: {e}")

    # Python.org docs
    try:
        python_org_snippets = additional_sources.fetch_python_org_examples(query, logger)
        all_snippets.extend(python_org_snippets)
    except Exception as e:
        logger.error(f"Error in python.org search: {e}")

    return all_snippets