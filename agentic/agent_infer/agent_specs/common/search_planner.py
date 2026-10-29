from pydantic import BaseModel

from ..base_agent import base_agent
PROMPT = (
    "produce a set of web searches to gather the context needed. Aim for recent "
    "headlines, earnings calls or 10‑K snippets, analyst commentary, and industry background. "
    "Output between 5 and 15 search terms to query for."
)


class SearchItem(BaseModel):
    reason: str
    """Your reasoning for why this search is relevant."""

    query: str
    """The search term to feed into a web (or file) search."""


class SearchPlan(BaseModel):
    searches: list[SearchItem]
    """A list of searches to perform."""


search_planner_agent = base_agent.clone(
    name="SearchPlanAgent",
    instructions=PROMPT,
    output_type=SearchPlan
)
