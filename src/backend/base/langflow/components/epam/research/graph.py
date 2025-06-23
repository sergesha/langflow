from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from langflow.components.epam.research.prompts import (
    ANSWER_INSTRUCTIONS,
    QUERY_WRITER_INSTRUCTIONS,
    REFLECTION_INSTRUCTIONS,
    get_current_date,
)
from langflow.components.epam.research.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from langflow.components.epam.research.tools_and_schemas import Reflection, SearchQueryList


async def generate_query_node(state: OverallState, config: dict[str, Any]) -> QueryGenerationState:
    """Generate initial search queries based on the research topic."""
    # Get model from configurable section
    configurable = config.get("configurable", {})
    llm = configurable.get("query_generator_model")

    if not llm:
        msg = "Query generator model not provided in configuration"
        raise ValueError(msg)

    prompt = QUERY_WRITER_INSTRUCTIONS.format(
        number_queries=state["initial_search_query_count"],
        research_topic=state["research_topic"],
        current_date=get_current_date(),
    )

    structured_llm = llm.with_structured_output(SearchQueryList)
    result = await structured_llm.ainvoke(prompt)

    return {"query_list": result.query}


def continue_to_web_research_router(state: QueryGenerationState) -> list[Send]:
    """Create web research branches for each generated query."""
    return [
        Send(
            "web_research",
            {
                "search_query": search_query,
                "id": idx,
                "research_topic": state.get("research_topic", ""),
                "max_research_loops": state.get("max_research_loops", 2),
                "sources_gathered": [],
                "web_research_result": [],
            },
        )
        for idx, search_query in enumerate(state["query_list"])
    ]


async def web_research_node(state: WebSearchState, config: dict[str, Any]) -> OverallState:
    """Execute web search and process results."""
    configurable = config.get("configurable", {})
    search_tool_config_value = configurable.get("search_tool")
    search_tool = (
        search_tool_config_value[0]
        if isinstance(search_tool_config_value, list) and search_tool_config_value
        else search_tool_config_value
    )

    if not search_tool or not isinstance(search_tool, StructuredTool):
        msg = f"Search tool must be a StructuredTool, but got: {type(search_tool).__name__ if search_tool else 'None'}"
        raise ValueError(msg)

    query = state["search_query"]
    query_id = state["id"]

    results_text = []
    gathered_sources = []

    try:
        # Tools typically expect input_value as the standard parameter name
        docs = await search_tool.ainvoke({"input_value": query, "query": query})

        if isinstance(docs, list) and all(hasattr(d, "page_content") for d in docs):
            for i, doc in enumerate(docs):
                url = doc.metadata.get("source", f"http://source-{query_id}-{i}.com")
                title = doc.metadata.get("title", f"Result {i+1} for '{query}'")
                content = doc.page_content

                citation = f"[{query_id}.{i+1}]"
                results_text.append(f"{citation} {content}")
                gathered_sources.append(
                    {"value": url, "short_url": citation, "title": title, "snippets": [content], "id": query_id}
                )
        elif isinstance(docs, str):
            content = docs
            citation = f"[{query_id}.0]"
            results_text.append(f"{citation} {content}")
            gathered_sources.append(
                {
                    "value": f"query-{query_id}",
                    "short_url": citation,
                    "title": f"Results for '{query}'",
                    "snippets": [content],
                    "id": query_id,
                }
            )
        else:
            content = str(docs) if docs else "No results found."
            citation = f"[{query_id}.0]"
            results_text.append(f"{citation} {content}")
            gathered_sources.append(
                {
                    "value": f"query-{query_id}",
                    "short_url": citation,
                    "title": f"Results for '{query}'",
                    "snippets": [content],
                    "id": query_id,
                }
            )

    except AttributeError as ae:  # Catch if ainvoke is not found on the tool
        msg = (
            f"Configured search tool (type: {type(search_tool).__name__}) "
            f"is not invokable as expected (missing 'ainvoke' method). Original error: {ae}"
        )
        raise ValueError(msg)
    except Exception as e:  # General error handling for the search process
        error_citation = f"[ERROR-{query_id}]"
        error_text = f"Error during search with tool {type(search_tool).__name__}: {e!s}"
        results_text.append(f"{error_citation} {error_text}")
        gathered_sources.append(
            {"value": "error", "short_url": error_citation, "title": "Error", "snippets": [error_text], "id": query_id}
        )

    return {
        "web_research_result": ["\n".join(results_text)],
        "sources_gathered": gathered_sources,
        "search_query": [query],
    }


async def reflection_node(state: OverallState, config: dict[str, Any]) -> ReflectionState:
    """Analyze results and identify knowledge gaps."""
    configurable = config.get("configurable", {})
    llm = configurable.get("reflection_model")

    if not llm:
        msg = "Reflection model not provided in configuration"
        raise ValueError(msg)

    current_summaries = "\n\n---\n\n".join(state.get("web_research_result", ["No information gathered yet."]))

    prompt = REFLECTION_INSTRUCTIONS.format(
        research_topic=state["research_topic"], summaries=current_summaries, current_date=get_current_date()
    )

    structured_llm = llm.with_structured_output(Reflection)
    result = await structured_llm.ainvoke(prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state.get("research_loop_count", 0) + 1,
        "number_of_ran_queries": len(state.get("search_query", [])),
    }


def evaluate_research_router(state: ReflectionState) -> str | list[Send]:
    """Determine whether to continue research or finalize answer."""
    if state.get("is_sufficient") or state.get("research_loop_count", 0) >= state.get("max_research_loops", 2):
        return "finalize_answer"

    follow_ups = state.get("follow_up_queries", [])
    if not follow_ups:
        return "finalize_answer"

    base_id = state.get("number_of_ran_queries", len(state.get("query_list", [])))

    return [
        Send(
            "web_research",
            {
                "search_query": query,
                "id": base_id + idx,
                "research_topic": state["research_topic"],
                "max_research_loops": state["max_research_loops"],
                "sources_gathered": state.get("sources_gathered", []),
                "web_research_result": state.get("web_research_result", []),
            },
        )
        for idx, query in enumerate(follow_ups)
    ]


async def finalize_answer_node(state: OverallState, config: dict[str, Any]) -> dict[str, Any]:
    """Generate final comprehensive answer with citations."""
    configurable = config.get("configurable", {})
    llm = configurable.get("answer_model")

    if not llm:
        msg = "Answer model not provided in configuration"
        raise ValueError(msg)

    summaries = "\n\n---\n\n".join(state.get("web_research_result", ["No information to finalize."]))

    prompt = ANSWER_INSTRUCTIONS.format(
        research_topic=state["research_topic"], summaries_with_sources=summaries, current_date=get_current_date()
    )

    raw_response = await llm.ainvoke(prompt)
    answer_text = raw_response.content

    cited_sources = []
    citation_index = 1
    final_text = answer_text

    valid_sources = [s for s in state.get("sources_gathered", []) if isinstance(s.get("short_url"), str)]

    # Sort by marker length to avoid partial replacements
    for source in sorted(valid_sources, key=lambda s: len(s["short_url"]), reverse=True):
        marker = source["short_url"]
        if marker in final_text:
            final_text = final_text.replace(marker, f"[{citation_index}]")
            cited_sources.append({"number": citation_index, "title": source["title"], "url": source["value"]})
            citation_index += 1

    return {"messages": [AIMessage(content=final_text)], "sources_gathered": cited_sources}


def compile_research_graph():
    """Compile the research graph with sequential and parallel execution paths."""
    graph = StateGraph(OverallState)

    # Add core nodes
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("finalize_answer", finalize_answer_node)

    # Define flow
    graph.add_edge(START, "generate_query")

    graph.add_conditional_edges("generate_query", continue_to_web_research_router, ["web_research"])

    graph.add_edge("web_research", "reflection")

    graph.add_conditional_edges("reflection", evaluate_research_router, ["web_research", "finalize_answer"])

    graph.add_edge("finalize_answer", END)

    return graph.compile()
