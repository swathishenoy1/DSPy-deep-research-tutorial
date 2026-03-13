import os
import dspy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your environment")

lm = dspy.LM('openai/gpt-5', api_key=OPENAI_API_KEY, max_tokens=64000)
dspy.configure(lm=lm)
lm("Say hello")

researcher_signature = "research_request: str -> report: str"
researcher = dspy.Predict(researcher_signature)

result = researcher(research_request="Write a history of Coyote Hills, a park in the East Bay Regional Parks District in California.")

with open("output/predict.txt", "w") as f:
    f.write(result.report)
