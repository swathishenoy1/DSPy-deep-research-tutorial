import json
import os
from typing import TypedDict

import dspy

from config import ARTIFACT_OUTPUT, BUDGETS, DEFAULT_BUDGET, REPORT_OUTPUT
from signatures import ClarifyingSignature, ProcessSource
from tools import internet_search, read_webpage


class Subtopic(TypedDict):
    subtopic: str
    urls: list[str]


def run_pipeline(
    research_request: str,
    budget_name: str = DEFAULT_BUDGET,
    num_clarifying_questions: int = 3,
) -> str:
    budget = BUDGETS[budget_name]

    clarifier = dspy.Predict(ClarifyingSignature)
    clarifying = clarifier(
        research_request=research_request,
        number_of_questions=num_clarifying_questions,
    )

    q_and_a = []
    for question in clarifying.clarifying_questions:
        answer = input(f"{question}\n")
        q_and_a.append({"clarifying_question": question, "user_guidance": answer})

    planner = dspy.Predict(
        "research_request: str, clarifying_questions_and_answers, "
        "max_number_of_sub_topics: int -> topics_to_research: list[str]"
    )
    topics_to_research = planner(
        research_request=research_request,
        clarifying_questions_and_answers=q_and_a,
        max_number_of_sub_topics=budget["max_subtopics"],
    ).topics_to_research

    print("\nSubtopics to research:")
    for topic in topics_to_research:
        print(f"- {topic}")

    gatherer_sig = (
        "research_request: str, subtopic_to_research: str, num_sources: int -> "
        "relevant_urls_to_investigate: list[str]"
    )
    gatherer = dspy.ReAct(
        gatherer_sig, tools=[internet_search], max_iters=budget["max_search_results"]
    )

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

    source_processing_agent = dspy.Predict(ProcessSource)

    processed_sources = []
    for subtopic_source in subtopics_with_sources:
        subtopic_findings = {
            "subtopic": subtopic_source["subtopic"],
            "findings": [],
        }
        urls = subtopic_source.get("urls") or []
        for url in urls:
            if url.endswith(".pdf"):
                print(f"  Skipping PDF: {url}")
                continue
            page_content = read_webpage(url)
            result = source_processing_agent(
                research_task=research_request,
                research_subtask=subtopic_source["subtopic"],
                url=url,
                page_content=page_content,
            )
            subtopic_findings["findings"].append(
                {
                    "url": url,
                    "summary": result.summary,
                    "relevant_facts": result.relevant_facts,
                    "interesting_annecdotes": result.interesting_annecdotes,
                    "additional_topics_to_explore": result.additional_topics_to_explore,
                }
            )
            print(f"Processed: {url}")
        processed_sources.append(subtopic_findings)

    if ARTIFACT_OUTPUT:
        os.makedirs(os.path.dirname(ARTIFACT_OUTPUT) or ".", exist_ok=True)
        with open(ARTIFACT_OUTPUT, "w") as f:
            json.dump(processed_sources, f, indent=4)
        print(f"\nResearch artifacts saved to {ARTIFACT_OUTPUT}")

    synthesizer = dspy.ChainOfThought(
        "research_request: str, clarifying_questions_and_answers, "
        "subtopic_research: list[dict] -> final_report: str"
    )
    final_report = synthesizer(
        research_request=research_request,
        clarifying_questions_and_answers=q_and_a,
        subtopic_research=processed_sources,
    ).final_report

    if REPORT_OUTPUT:
        os.makedirs(os.path.dirname(REPORT_OUTPUT) or ".", exist_ok=True)
        with open(REPORT_OUTPUT, "w") as f:
            f.write(final_report)
        print(f"\nReport saved to {REPORT_OUTPUT}")
    else:
        print(final_report)

    return final_report
