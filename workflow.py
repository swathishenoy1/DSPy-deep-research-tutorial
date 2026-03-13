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

BUDGETS = {
    "light":  {"max_subtopics": 3, "num_sources": 2, "max_search_results": 3},
    "medium": {"max_subtopics": 5, "num_sources": 2, "max_search_results": 5},
    "deep":   {"max_subtopics": 7, "num_sources": 5, "max_search_results": 10},
}
budget = BUDGETS["medium"]

# Output destinations
ARTIFACT_OUTPUT = "output/Gemini/research_artifact.json"  # Set to None to skip writing artifacts to disk
REPORT_OUTPUT = "output/Gemini/workflow.txt"  # Set to None to print to console instead


# Our research request
research_request = "Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California."

class ClarifyingSignature(dspy.Signature):
    "Generate clarifying questions to inform research work in response to the request."
    research_request: str = dspy.InputField()
    number_of_questions: int = dspy.InputField(desc="The number of clarifying questions to generate")
    clarifying_questions: list[str] = dspy.OutputField()

clarifier = dspy.Predict(ClarifyingSignature)

results = clarifier(research_request=research_request, number_of_questions=3)

# Get our answers
q_and_a = []
for question in results.clarifying_questions:
    answer = input(f"{question}\n")
    q_and_a.append({"clarifying_question": question, "user_guidance": answer})

planner = dspy.Predict("research_request: str, clarifying_questions_and_answers, max_number_of_sub_topics: int -> topics_to_research: list[str]")

topics_to_research = planner(
    research_request=research_request,
    clarifying_questions_and_answers=q_and_a,
    max_number_of_sub_topics=budget["max_subtopics"],
).topics_to_research

print("\nSubtopics to research:")
for topic in topics_to_research:
    print(f"- {topic}")

gatherer_sig = "research_request: str, subtopic_to_research: str, num_sources: int -> relevant_urls_to_investigate: list[str]"
gatherer = dspy.ReAct(gatherer_sig, tools=[internet_search], max_iters=budget["max_search_results"])


class Subtopic(TypedDict):
    subtopic: str
    urls: list[str]

subtopics_with_sources: list[Subtopic] = []
for subtopic in topics_to_research:
    urls = gatherer(
        research_request=research_request,
        subtopic_to_research=subtopic,
        num_sources=budget["num_sources"],
    ).relevant_urls_to_investigate
    subtopics_with_sources.append(Subtopic(subtopic=subtopic, urls=urls))

print("\nSources by subtopic:")
for subtopic_sources in subtopics_with_sources:
    print(f"Subtopic: {subtopic_sources['subtopic']}")
    for url in (subtopic_sources.get("urls") or []):
        print(f"- {url}")
    print()

class ProcessSource(dspy.Signature):
    """
    Process a source (e.g. a webpage) to extract relevant information that helps answer the research question.
    """
    research_task: str = dspy.InputField(desc="The overall research task or question that we are trying to answer. Use this as context when determining what information is relevant in the source.")
    research_subtask: str = dspy.InputField(desc="The subtopic that this source is meant to inform.")
    url: str = dspy.InputField()
    page_content: dict = dspy.InputField(desc="The content of the webpage; may also include metadata such as the title, author, publication date, etc.")
    summary: str = dspy.OutputField(descriptions="A concise summary of the source, focusing on information relevant to the research question. Aim for 1-2 paragraphs that capture the essence of the source's content in relation to the research question.")
    relevant_facts: list[str] = dspy.OutputField(descriptions="A list of relevant facts extracted from the source that help answer the research question.")
    interesting_annecdotes: list[str] = dspy.OutputField(description="Interesting or unique pieces of information that could make the report more engaging, relatable, or compelling.")
    additional_topics_to_explore: list[str] = dspy.OutputField(description="Additional items or questions to explore that are identified while processing the source.")

source_processing_agent = dspy.Predict(ProcessSource)


processed_sources = []
for subtopic_source in subtopics_with_sources:
    subtopic_findings = {"subtopic": subtopic_source['subtopic'], "findings": []}
    urls = subtopic_source.get("urls") or []
    for url in urls:
        if url.endswith('.pdf'):
            print(f"  Skipping PDF: {url}")
            continue
        page_content = read_webpage(url)
        result = source_processing_agent(
            research_task=research_request,
            research_subtask=subtopic_source["subtopic"],
            url=url,
            page_content=page_content,
        )
        subtopic_findings["findings"].append({
            "url": url,
            "summary": result.summary,
            "relevant_facts": result.relevant_facts,
            "interesting_annecdotes": result.interesting_annecdotes,
            "additional_topics_to_explore": result.additional_topics_to_explore,
        })
        print(f"Processed: {url}")
    processed_sources.append(subtopic_findings)


if ARTIFACT_OUTPUT:
    os.makedirs(os.path.dirname(ARTIFACT_OUTPUT) or ".", exist_ok=True)
    with open(ARTIFACT_OUTPUT, "w") as f:
        json.dump(processed_sources, f, indent=4)
    print(f"\nResearch artifacts saved to {ARTIFACT_OUTPUT}")


synthesizer = dspy.ChainOfThought(
    "research_request: str, clarifying_questions_and_answers, subtopic_research: list[dict] -> final_report: str"
)


final_report = synthesizer(
    research_request=research_request,
    clarifying_questions_and_answers=q_and_a,
    subtopic_research=processed_sources,
).final_report

print(final_report)