from pydantic_prompter import Prompter
from pydantic import BaseModel, Field
from typing import List
import os

os.environ["OPENAI_API_KEY"] = "sk-...."


class RecommendedEntry(BaseModel):
    id: str
    name: str
    reason: str = Field(description="Why this entry fits the query", default="")


class RecommendationResults(BaseModel):
    title: str
    entries: List[RecommendedEntry]


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def rank_recommendation(entries, query) -> RecommendationResults:
    """
    - system: You are a movie ranking expert
    - user: |
        Which of the following JSON entries fit best to the query.
        order by best fit descending
        Base your answer ONLY on the given JSON entries

    - user: >
        The JSON entries:
        {{ entries }}

    - user: "query: {{ query }}"

    """


my_entries = (
    '[{"text": "Description: Four everyday suburban guys come together as a ....'
)
print(rank_recommendation(entries=my_entries, query="Romantic comedy"))

# >>> title='Romantic Comedy' entries=[RecommendedEntry(id='2312973', \
#       name='The Ugly Truth', reason='Romantic comedy genre')]


print(rank_recommendation.build_string(entries=my_entries, query="Romantic comedy"))

# >>> system: You are a movie ranking expert
#     user: Which of the following JSON entries fit best to the query.
#     order by best fit descending
#     Base your answer ONLY on the given JSON entries
#     user: The JSON entries: [{"text": "Description: Four everyday suburban guys come together as a ....
#     user: query: Romantic comedy
