# scraper/documentation_sources.py

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from urllib.parse import urljoin, urlparse
from .parser import extract_code as extract_code_from_html_markdown


class DocumentationSources:
    """Scrape documentation sites for code examples and patterns."""

    def __init__(self, user_agent: str, timeout: int = 15):
        self.user_agent = user_agent
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

        # Common documentation patterns
        self.doc_domains = {
            'readthedocs.io': self._scrape_readthedocs,
            'docs.python.org': self._scrape_python_docs,
            'github.io': self._scrape_github_pages,
            'sphinx': self._scrape_sphinx_docs,
        }

    def fetch_package_documentation(self, package_name: str, logger) -> List[str]:
        """Try to find and scrape documentation for a package."""
        logger.info(f"Searching documentation for: {package_name}")
        snippets = []

        # Common documentation URL patterns
        doc_urls = self._generate_doc_urls(package_name)

        for url in doc_urls:
            try:
                snippets.extend(self._scrape_doc_site(url, logger))
                if len(snippets) > 50:  # Prevent overwhelming results
                    break
            except Exception as e:
                logger.debug(f"Could not scrape {url}: {e}")

        return snippets[:50]  # Cap total results

    def _generate_doc_urls(self, package_name: str) -> List[str]:
        """Generate likely documentation URLs for a package."""
        urls = []

        # Official sites (for major packages) - prioritize these
        official_sites = {
            'django': 'https://docs.djangoproject.com/en/stable/',
            'flask': 'https://flask.palletsprojects.com/en/latest/',
            'fastapi': 'https://fastapi.tiangolo.com/',
            'requests': 'https://docs.python-requests.org/en/latest/',
            'numpy': 'https://numpy.org/doc/stable/',
            'pandas': 'https://pandas.pydata.org/docs/',
            'matplotlib': 'https://matplotlib.org/stable/',
            'opencv': 'https://docs.opencv.org/4.x/',
            'pyqt6': 'https://doc.qt.io/qtforpython/',
            'selenium': 'https://selenium-python.readthedocs.io/',
            'beautifulsoup4': 'https://www.crummy.com/software/BeautifulSoup/bs4/doc/',
            'sqlalchemy': 'https://docs.sqlalchemy.org/en/20/',
            'pytest': 'https://docs.pytest.org/en/stable/',
            'click': 'https://click.palletsprojects.com/en/8.1.x/',
            'pydantic': 'https://docs.pydantic.dev/latest/',
            'asyncio': 'https://docs.python.org/3/library/asyncio.html',
        }

        package_lower = package_name.lower().replace('_', '').replace('-', '')

        # Check for exact matches first
        for package, url in official_sites.items():
            if package_lower == package.replace('_', '').replace('-', ''):
                urls.insert(0, url)
                break

        # ReadTheDocs patterns
        urls.extend([
            f"https://{package_name}.readthedocs.io/en/latest/",
            f"https://{package_name.replace('_', '-')}.readthedocs.io/en/latest/",
            f"https://{package_name.replace('-', '')}.readthedocs.io/en/stable/",
        ])

        # GitHub Pages patterns
        urls.extend([
            f"https://{package_name}.github.io/",
            f"https://{package_name.replace('_', '-')}.github.io/",
        ])

        return urls

    def _scrape_doc_site(self, url: str, logger) -> List[str]:
        """Scrape a documentation site for code examples."""
        snippets = []

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract code blocks using existing parser
            html_snippets = extract_code_from_html_markdown(response.text)

            # Filter for Python code
            for snippet in html_snippets:
                if self._looks_like_python(snippet) and len(snippet.strip()) > 30:
                    snippets.append(snippet)

            # Look for tutorial/example pages (but limit crawling)
            if len(snippets) < 10:  # Only crawl deeper if we didn't find much
                tutorial_links = self._find_tutorial_links(soup, url)
                for link in tutorial_links[:2]:  # Very limited to avoid overwhelming
                    try:
                        sub_snippets = self._scrape_doc_site(link, logger)
                        snippets.extend(sub_snippets[:3])  # Very limited per page
                        if len(snippets) > 20:  # Stop if we have enough
                            break
                    except:
                        continue

        except Exception as e:
            logger.debug(f"Error scraping {url}: {e}")

        return snippets

    def _looks_like_python(self, code: str) -> bool:
        """Quick heuristic to identify Python code."""
        code_lower = code.lower()

        # Strong Python indicators
        strong_indicators = [
            'import ', 'from ', 'def ', 'class ', 'if __name__',
            'print(', '>>>', 'python'
        ]

        # Weak indicators (need multiple)
        weak_indicators = [
            '.py', 'self.', '__init__', 'elif', 'lambda:', 'yield'
        ]

        strong_count = sum(1 for indicator in strong_indicators if indicator in code_lower)
        weak_count = sum(1 for indicator in weak_indicators if indicator in code_lower)

        return strong_count > 0 or weak_count >= 2

    def _find_tutorial_links(self, soup, base_url: str) -> List[str]:
        """Find links to tutorials, examples, guides."""
        tutorial_keywords = [
            'tutorial', 'example', 'guide', 'quickstart',
            'getting-started', 'howto', 'cookbook', 'user-guide'
        ]

        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text().lower()

            if any(keyword in text or keyword in href.lower()
                   for keyword in tutorial_keywords):
                full_url = urljoin(base_url, href)
                # Avoid external links and duplicates
                if urlparse(full_url).netloc == urlparse(base_url).netloc and full_url not in links:
                    links.append(full_url)

        return links[:5]  # Limit links to explore

    def _scrape_readthedocs(self, url: str, logger) -> List[str]:
        """Specialized scraping for ReadTheDocs sites."""
        return self._scrape_doc_site(url, logger)

    def _scrape_python_docs(self, url: str, logger) -> List[str]:
        """Specialized scraping for Python official docs."""
        return self._scrape_doc_site(url, logger)

    def _scrape_github_pages(self, url: str, logger) -> List[str]:
        """Specialized scraping for GitHub Pages docs."""
        return self._scrape_doc_site(url, logger)

    def _scrape_sphinx_docs(self, url: str, logger) -> List[str]:
        """Specialized scraping for Sphinx-generated docs."""
        return self._scrape_doc_site(url, logger)


def search_documentation_sources(query: str, logger) -> List[str]:
    """Main interface for documentation searching."""
    try:
        from config import USER_AGENT, DEFAULT_REQUEST_TIMEOUT, DOC_SCRAPING_TIMEOUT
    except ImportError:
        USER_AGENT = "RAGContentScraper/1.0"
        DEFAULT_REQUEST_TIMEOUT = 15
        DOC_SCRAPING_TIMEOUT = 20

    doc_sources = DocumentationSources(USER_AGENT, DOC_SCRAPING_TIMEOUT)
    return doc_sources.fetch_package_documentation(query, logger)