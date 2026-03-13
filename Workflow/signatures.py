import dspy


class ClarifyingSignature(dspy.Signature):
    """Generate clarifying questions to inform research work in response to the request."""

    research_request: str = dspy.InputField()
    number_of_questions: int = dspy.InputField(
        desc="The number of clarifying questions to generate"
    )
    clarifying_questions: list[str] = dspy.OutputField()


class ProcessSource(dspy.Signature):
    """Process a source (e.g. a webpage) to extract relevant information."""

    research_task: str = dspy.InputField(
        desc=(
            "The overall research task or question that we are trying to answer. "
            "Use this as context when determining what information is relevant in the source."
        )
    )
    research_subtask: str = dspy.InputField(
        desc="The subtopic that this source is meant to inform."
    )
    url: str = dspy.InputField()
    page_content: dict = dspy.InputField(
        desc="The content of the webpage; may also include metadata such as title/author/date."
    )
    summary: str = dspy.OutputField(
        descriptions=(
            "A concise summary of the source, focusing on information relevant to the research question. "
            "Aim for 1-2 paragraphs."
        )
    )
    relevant_facts: list[str] = dspy.OutputField(
        descriptions="Facts extracted from the source that help answer the research question."
    )
    interesting_annecdotes: list[str] = dspy.OutputField(
        description="Unique or engaging details to enrich the report."
    )
    additional_topics_to_explore: list[str] = dspy.OutputField(
        description="Follow-on items or questions identified while processing the source."
    )
