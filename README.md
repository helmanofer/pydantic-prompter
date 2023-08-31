# Pydantic Prompter

### Installation

`pip install pydantic-prompter`

### setup

`export OPENAI_API_KEY=<your openai token>
`

### Basic usage

```py
from pydantic_prompter import Prompter
from pydantic import BaseModel, Field
from typing import List
import os


class RecommendedEntry(BaseModel):
    id: str
    name: str
    reason: str = Field(
        description="Why this entry fits the query", default=""
    )


class RecommendationResults(BaseModel):
    title: str
    entries: List[RecommendedEntry]


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def rank_recommendation_entries(
        json_entries, user_query
) -> RecommendationResults:
    """
    >> system: You are a movie ranking expert
    >> user: >
        Which of the following JSON entries fit best to the query. 
        order by best fit descending
        Base your answer ONLY on the given JSON entries, 
        if you are not sure, or there are no entries

    >> user: >
        The JSON entries:
        {{ json_entries }}

    >> user: "query: {{ user_query }}"

    """


my_entries = "[{\"text\": \"Description: Four everyday suburban guys come together as a ...."
print(rank_recommendation_entries(json_entries=my_entries, user_query="Romantic comedy"))

```
