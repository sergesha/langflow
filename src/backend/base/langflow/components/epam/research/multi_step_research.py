from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from langflow.components.epam.research.graph import compile_research_graph
from langflow.components.epam.research.state import OverallState
from langflow.custom.custom_component.component import Component
from langflow.inputs import BoolInput, HandleInput, IntInput
from langflow.schema.message import ErrorMessage, Message
from langflow.schema.properties import Source
from langflow.template import Output

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class MultiStepResearch(Component):
    display_name = "Multi-Step Research"
    description = "A component that uses LangGraph to perform multi-step research."
    documentation: str = "https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart"
    icon = "search"
    name = "MultiStepResearch"

    inputs = [
        HandleInput(
            name="input_value",
            display_name="User Query",
            info="The question or topic to research.",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="query_generator_model",
            display_name="Query Generator Model",
            info="Optional LLM for generating search queries. Defaults to Answer Model.",
            input_types=["LanguageModel"],
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="reflection_model",
            display_name="Reflection Model",
            info="Optional LLM for reflecting on search results. Defaults to Answer Model.",
            input_types=["LanguageModel"],
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="answer_model",
            display_name="Answer Model",
            info="LLM for final answer generation and default for other steps.",
            input_types=["LanguageModel"],
            required=True,
        ),
        HandleInput(
            name="search_tool",
            display_name="Search Tool",
            info="A LangFlow tool that performs web searches. Must have ainvoke or _arun method.",
            input_types=["Tool"],
            required=True,
        ),
        IntInput(
            name="number_of_initial_queries",
            display_name="Number of Initial Queries",
            info="How many search queries to generate and run in parallel initially.",
            value=3,
        ),
        IntInput(
            name="max_research_loops",
            display_name="Max Research Loops",
            info="Maximum iterations of reflection and follow-up search.",
            value=2,
        ),
        BoolInput(
            name="verbose_logging",
            display_name="Verbose Logging",
            info="Enable detailed logging of graph execution steps.",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def _get_session_id(self) -> str:
        """Safely get session ID with fallback to new UUID."""
        session_id = getattr(self, "session_id", None)
        return str(session_id if session_id is not None else uuid4())

    def _get_component_id(self) -> str:
        """Safely get component ID with fallback to display name."""
        component_id = getattr(self, "_id", None)
        return str(component_id if component_id is not None else self.display_name)

    async def build_output(self) -> Message:
        """Build and execute the research graph.

        Returns:
            Message: A Message object containing research results or an ErrorMessage if failed.
        """
        self.status = "Starting research process..."
        self.log("Initializing multi-step research process")

        try:
            # Use default models if no special models are specified
            _query_generator_model = self.query_generator_model or self.answer_model
            _reflection_model = self.reflection_model or self.answer_model

            self.log(
                f"Models configuration - Query Generator: {_query_generator_model}, Reflection: {_reflection_model}, Answer: {self.answer_model}"
            )

            if self.verbose_logging:
                self.status = "Configuring research models and parameters..."
                self.log("Verbose logging enabled - configuring research parameters")

            # Configuration for graph nodes
            node_config: RunnableConfig = {
                "configurable": {
                    "query_generator_model": _query_generator_model,
                    "reflection_model": _reflection_model,
                    "answer_model": self.answer_model,
                    "search_tool": self.search_tool,
                    "verbose_logging": self.verbose_logging,
                }
            }

            self.status = "Compiling research graph..."
            self.log("Compiling LangGraph research pipeline")
            research_graph = compile_research_graph()

            # Create initial state
            research_topic = str(self.input_value)
            self.log(f"Starting research on topic: {research_topic}")

            initial_state = OverallState(
                messages=[HumanMessage(content=research_topic)],
                research_topic=research_topic,
                initial_search_query_count=self.number_of_initial_queries,
                max_research_loops=self.max_research_loops,
                research_loop_count=0,
                number_of_ran_queries=0,
                query_list=[],
                search_query=[],
                web_research_result=[],
                sources_gathered=[],
                follow_up_queries=[],
                is_sufficient=None,
                knowledge_gap=None,
            )

            self.log(
                f"Research parameters - Initial queries: {self.number_of_initial_queries}, Max loops: {self.max_research_loops}"
            )

            self.status = "Executing research process..."
            self.log("Starting research graph execution")

            final_state: OverallState = await research_graph.ainvoke(initial_state, config=node_config)

            # Extract results
            final_message = final_state.get("messages", [AIMessage(content="No answer generated.")])[-1]
            final_content = final_message.content if isinstance(final_message, AIMessage) else str(final_message)
            final_sources = final_state.get("sources_gathered", [])

            # Create stats for result metadata
            research_stats = {
                "research_loop_count": final_state.get("research_loop_count", 0),
                "number_of_ran_queries": final_state.get("number_of_ran_queries", 0),
                "sources_count": len(final_sources),
            }

            self.log(
                f"Research completed - Stats: Queries: {research_stats['number_of_ran_queries']}, Loops: {research_stats['research_loop_count']}, Sources: {research_stats['sources_count']}"
            )

            status_msg = f"Research completed: {research_stats['number_of_ran_queries']} queries executed across {research_stats['research_loop_count']} research loops"
            self.status = status_msg
            self.log(status_msg)

            # Create and return final message
            message = await Message.create(
                text=final_content,
                sender=self.display_name,
                sender_name=self.display_name,
                session_id=self._get_session_id(),
                files=[],
                properties={"research_stats": research_stats, "sources": final_sources},
            )

            self.status = message
            self.log("Generated response message with metadata")
            return message

        except Exception as e:
            error_msg = f"Research failed: {e!s}"
            self.status = error_msg
            self.log(f"Error during research - Type: {type(e).__name__}, Message: {e!s}")

            return ErrorMessage(
                exception=e,
                session_id=self._get_session_id(),
                source=Source(id=self._get_component_id(), display_name=self.display_name, source=self.display_name),
                trace_name=getattr(self, "trace_name", None),
                flow_id=None,
            )
