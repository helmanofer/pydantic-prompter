# Pydantic Prompter
A lightweight tool that lets you simply build prompts and get Pydantic objects as outputs 

Documentation https://helmanofer.github.io/pydantic-prompter

### Installation

`pip install pydantic-prompter`

### Setup

`export OPENAI_API_KEY=<your openai token>`

### Basic usage

Create you output model with Pydantic

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
```

Create a Prompter function as a YML string
```py
@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def rank_recommendation(entries, query) -> RecommendationResults:
    """
    - system: You are a movie ranking expert
    - user: >
        Which of the following JSON entries fit best to the query. 
        order by best fit descending
        Base your answer ONLY on the given JSON entries, 
        if you are not sure, or there are no entries

    - user: >
        The JSON entries:
        {{ entries }}

    - user: "query: {{ query }}"

    """
```
Run you function
```py
my_entries = "[{\"text\": \"Description: Four everyday suburban guys come together as a ...."
print(rank_recommendation(entries=my_entries, query="Romantic comedy"))

```
Debug your prompt
```py
print(rank_recommendation.build_string(entries=my_entries, query="Romantic comedy"))

```
See the [Documentation](https://helmanofer.github.io/pydantic-prompter) for more details
