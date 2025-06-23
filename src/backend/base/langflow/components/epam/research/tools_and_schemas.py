from pydantic import BaseModel, Field


# Pydantic models for structured LLM output validation
class SearchQueryList(BaseModel):
    query: list[str] = Field(description="List of search queries to investigate the topic")


class Reflection(BaseModel):
    is_sufficient: bool = Field(description="Whether current information is sufficient")
    knowledge_gap: str = Field(description="Description of what information is missing")
    follow_up_queries: list[str] = Field(description="List of follow-up queries to fill knowledge gaps")
