# scraper/fetcher.py

import requests
import urllib.robotparser
from urllib.parse import urlparse

try:
    from config import DEFAULT_REQUEST_TIMEOUT, USER_AGENT, DOCS_CRAWL_RESPECT_ROBOTS_TXT
except ImportError:
    # Fallback for testing or if config is not on path
    DEFAULT_REQUEST_TIMEOUT = 10
    USER_AGENT = "RAGContentScraper/1.0 (fallback)"
    DOCS_CRAWL_RESPECT_ROBOTS_TXT = False # Fallback

# Simple cache for robots.txt parsers
_robot_parsers = {}

def _get_robot_parser(url):
    """
    Retrieves or creates a RobotFileParser for the given URL's site.
    """
    if not DOCS_CRAWL_RESPECT_ROBOTS_TXT:
        return None

    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    if robots_url in _robot_parsers:
        return _robot_parsers[robots_url]

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        _robot_parsers[robots_url] = rp
        return rp
    except Exception: # Broad exception for network issues, etc.
        _robot_parsers[robots_url] = None # Cache failure to avoid re-fetching constantly
        return None

def fetch_url(url, timeout=None, logger=None):
    """
    Fetches the content of the given URL.
    Respects robots.txt if DOCS_CRAWL_RESPECT_ROBOTS_TXT is True.
    Raises HTTPError on bad responses.
    """
    if DOCS_CRAWL_RESPECT_ROBOTS_TXT:
        rp = _get_robot_parser(url)
        if rp and not rp.can_fetch(USER_AGENT, url):
            if logger:
                logger.info(f"Skipping URL due to robots.txt: {url}")
            raise requests.exceptions.RequestException(f"Fetching disallowed by robots.txt: {url}")

    actual_timeout = timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
    headers = {
        "User-Agent": USER_AGENT
    }
    response = requests.get(url, headers=headers, timeout=actual_timeout)
    response.raise_for_status()
    return response.text