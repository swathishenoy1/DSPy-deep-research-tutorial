from tavily import TavilyClient

from config import TAVILY_API_KEY


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def internet_search(query: str, max_results: int = 5, include_raw_content: bool = False):
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic="general",
    )


def read_webpage(url: str) -> dict:
    """Read the content of a URL."""
    return tavily_client.extract(url)
