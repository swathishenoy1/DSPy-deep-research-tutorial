import os

import dspy

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Add it to your environment")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Add it to your environment or .env file.")

# LLM config
LM_MODEL = "gemini/gemini-2.5-flash"
LM_MAX_TOKENS = 64000

# Budgets
BUDGETS = {
    "light": {"max_subtopics": 3, "num_sources": 2, "max_search_results": 3},
    "medium": {"max_subtopics": 5, "num_sources": 2, "max_search_results": 5},
    "deep": {"max_subtopics": 7, "num_sources": 5, "max_search_results": 10},
}
DEFAULT_BUDGET = "medium"

# Output destinations (set to None to disable)
ARTIFACT_OUTPUT = "output/Gemini/research_artifact.json"
REPORT_OUTPUT = "output/Gemini/workflow.txt"

# Default research request
RESEARCH_REQUEST = (
    "Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California."
)


def configure_dspy() -> dspy.LM:
    lm = dspy.LM(LM_MODEL, api_key=GEMINI_API_KEY, max_tokens=LM_MAX_TOKENS)
    dspy.configure(lm=lm)
    return lm
