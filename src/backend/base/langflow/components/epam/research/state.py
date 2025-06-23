import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class Query(TypedDict):
    """Represents a single search query and its rationale."""

    query: str
    rationale: str


class OverallState(TypedDict):
    """Defines the complete state of the research graph.

    This TypedDict includes all fields that are part of the graph's state,
    whether initialized at the start or populated by nodes during execution.
    Fields intended for aggregation across parallel branches use Annotated with operator.add.
    """

    # Initialized at the start of the flow
    messages: list[BaseMessage]
    research_topic: str
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int  # Initialized to 0, incremented by reflection_node
    number_of_ran_queries: int  # Initialized to 0, updated by reflection_node

    # Populated by generate_query_node
    query_list: list[Query]  # Initialized as [], populated by generate_query_node

    # Aggregated from web_research_node outputs
    search_query: Annotated[list[str], operator.add]  # Initialized as [], appended to by web_search_node
    web_research_result: Annotated[list[str], operator.add]  # Initialized as [], appended to by web_search_node
    sources_gathered: Annotated[list[dict[str, Any]], operator.add]  # Initialized as [], appended to by web_search_node

    # Populated by reflection_node
    is_sufficient: bool | None  # Initialized as None
    knowledge_gap: str | None  # Initialized as None
    follow_up_queries: Annotated[list[str], operator.add]  # Initialized as [], updated by reflection_node


class ReflectionState(TypedDict):
    """Output of the reflection step. These fields are merged into OverallState."""

    is_sufficient: bool
    knowledge_gap: str | None
    follow_up_queries: Annotated[
        list[str], operator.add
    ]  # This specific annotation is key for how reflection_node returns it
    research_loop_count: int
    number_of_ran_queries: int


class QueryGenerationState(TypedDict):
    """Output of the query generation step. Used as input for continue_to_web_research_router."""

    query_list: list[Query]
    # Context fields from OverallState needed by the router
    research_topic: str
    max_research_loops: int


class WebSearchState(TypedDict):
    """Input state for an individual web search branch.

    This defines the dictionary passed via Send to each web_research_node instance.
    """

    search_query: str  # The specific query for this branch
    id: str  # Identifier for this branch
    # Context fields passed from the router
    research_topic: str
    max_research_loops: int
    # Initial empty lists for this branch to populate; will be aggregated into OverallState
    sources_gathered: list[dict[str, Any]]
    web_research_result: list[str]
