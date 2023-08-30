import os
from typing import List

from pydantic import BaseModel, Field

from pydatic_prompter import Prompter


class RecommendedEntry(BaseModel):
    id: str
    name: str
    reason: str = Field(description="Why this entry fits the query", default="")


class RecommendationResults(BaseModel):
    title: str
    entries: List[RecommendedEntry]


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def rank_recommendation_entries(json_entries, user_query) -> RecommendationResults:
    """
    >> system: You are a movie ranking expert
    >> user: >
        Which of the following JSON entries fit best to the query. order by best fit descending
        Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries

    >> user: >
        The JSON entries:
        {{ json_entries }}

    >> user: "query: {{ user_query }}"

    """


os.environ["OPENAI_API_KEY"] = "sk-..."
my_entries = []
print(rank_recommendation_entries(json_entries=my_entries, user_query="Romantic comedy"))
print(rank_recommendation_entries.build_string(json_entries=my_entries, user_query="Romantic comedy"))
