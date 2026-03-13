import os
import dspy
from tavily import TavilyClient
from typing import Literal

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your environment")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Add it to your environment or .env file.")

lm = dspy.LM('openai/gpt-5', api_key=OPENAI_API_KEY, max_tokens=64000)
dspy.configure(lm=lm)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def internet_search(query: str, max_results: int = 5, include_raw_content: bool = False):
	"""Run a web search"""
	return tavily_client.search(
		query, max_results=max_results,
		include_raw_content=include_raw_content,
		topic="general",
	)

def read_webpage(url: str) -> dict:
	"""Read the content of a URL."""
	response = tavily_client.extract(url)
	return response


researcher_signature = "research_request: str -> report: str"
researcher = dspy.ReAct(researcher_signature, tools=[internet_search, read_webpage])


result = researcher(research_request="Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California.")

with open("output/react.txt", "w") as f:
    f.write(result.report)
