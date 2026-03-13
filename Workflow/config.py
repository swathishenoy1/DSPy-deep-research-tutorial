import os

import dspy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your environment")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Add it to your environment or .env file.")

# LLM config
LM_MODEL = "openai/gpt-5"
LM_MAX_TOKENS = 64000

# Budgets
BUDGETS = {
    "light": {"max_subtopics": 3, "num_sources": 2, "max_search_results": 3},
    "medium": {"max_subtopics": 5, "num_sources": 2, "max_search_results": 5},
    "deep": {"max_subtopics": 7, "num_sources": 5, "max_search_results": 10},
}
DEFAULT_BUDGET = "medium"

# Output destinations (set to None to disable)
ARTIFACT_OUTPUT = "output/GPT-5/research_artifact.json"
REPORT_OUTPUT = "output/GPT-5/workflow.txt"

# Default research request
RESEARCH_REQUEST = (
    "Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California."
)


def configure_dspy() -> dspy.LM:
    lm = dspy.LM(LM_MODEL, api_key=OPENAI_API_KEY, max_tokens=LM_MAX_TOKENS)
    dspy.configure(lm=lm)
    return lm
