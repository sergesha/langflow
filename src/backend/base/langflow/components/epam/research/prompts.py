import datetime


# --- Prompt Templates (adapted from prompts.py) ---
def get_current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")


QUERY_WRITER_INSTRUCTIONS = """\
You are an AI research assistant.
Your task is to generate a list of {number_queries} search queries based on the provided research topic:
{research_topic}.
Today's date is {current_date}.
Ensure the queries are diverse and cover different aspects of the topic.
Output ONLY a JSON object with a single key "query" whose value is a list of strings (the search queries).
"""

WEB_SEARCHER_INSTRUCTIONS = """\
You are an AI research assistant.
Research the following topic using your search capabilities:
{research_topic}

Today's date is {current_date}.

Please search for relevant information and synthesize what you find into a coherent response.
Make sure to:
1. Be factual and objective
2. Include specific details and examples
3. Cite your sources using [X] format markers

Response:
"""

REFLECTION_INSTRUCTIONS = """\
You are an AI research assistant. You have been tasked with researching the topic: {research_topic}.
You have gathered the following information so far:
---
{summaries}
---
Today's date is {current_date}.
Based on the information gathered, reflect on its sufficiency.
1. Is the current information sufficient to provide a comprehensive answer to the research topic?
2. If not, what specific knowledge gaps exist?
3. Generate a list of follow-up search queries to address these gaps.
If the information is sufficient, provide an empty list.

Output ONLY a JSON object with the following keys: "is_sufficient" (boolean),
"knowledge_gap" (string, or empty if sufficient), "follow_up_queries" (list of strings).
"""

ANSWER_INSTRUCTIONS = """\
You are an AI research assistant.
Your goal is to provide a comprehensive answer to the research topic: {research_topic},
based on the gathered information.
Today's date is {current_date}.
Ensure your answer is well-structured, informative, and directly addresses the research topic.
Incorporate information from the following summaries, and cite them appropriately using the [Source X] format,
where X is the number of the source.
Summaries:
{summaries_with_sources}

Final Answer:
"""
