import os
import dspy

from tavily import TavilyClient
from typing import Literal

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Add it to your environment")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Add it to your environment or .env file.")

lm = dspy.LM('gemini/gemini-2.5-flash-lite', api_key=GEMINI_API_KEY, max_tokens=64000)
dspy.configure(lm=lm)
lm("Say hello")

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


# Create our clarifier program
class ClarifyingSignature(dspy.Signature):
    "Generate clarifying questions to inform research work in response to the request."
    research_request: str = dspy.InputField()
    number_of_questions: int = dspy.InputField(desc="The number of clarifying questions to generate")
    clarifying_questions: list[str] = dspy.OutputField()

clarifier = dspy.Predict(ClarifyingSignature)


# Create our researcher program, with a clarifying_questions_and_answers input
researcher_signature = "research_request: str, clarifying_questions_and_answers -> report: str"
researcher = dspy.ReAct(researcher_signature, tools=[internet_search, read_webpage])

# Our research request
research_request = "Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California."

# Generate our questions
results = clarifier(research_request=research_request, number_of_questions=3)

# Get our answers
q_and_a = []
for question in results.clarifying_questions:
    answer = input(f"{question}\n")
    q_and_a.append({"clarifying_question": question, "user_guidance": answer})


result = researcher(
    research_request="Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California.",
    clarifying_questions_and_answers=q_and_a
)

with open("output/Gemini/clarifier.txt", "w") as f:
    f.write(result.report)
