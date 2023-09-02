# Pydantic Prompter
Pydantic Prompter is a lightweight utility designed to facilitate the construction of prompts using YAML and generate Pydantic objects as outputs.

Documentation https://helmanofer.github.io/pydantic-prompter

### Installation
To install Pydantic Prompter, use the following command:



```bash
pip install pydantic-prompter
```

### Setup
Before using Pydantic Prompter, ensure that you set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=<your openai token>
```

### Basic usage

Begin by defining your output model using Pydantic:


```py
from pydantic import BaseModel, Field
from typing import List


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

Next, create a Prompter function, which is defined as a YAML string with Jinja2 templating or simple string formatting:

```py
from pydantic_prompter import Prompter


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
Execute your function as follows:

```py
my_entries = "[{\"text\": \"Description: Four everyday suburban guys come together as a ...."
print(rank_recommendation(entries=my_entries, query="Romantic comedy"))

```
For debugging purposes, inspect your prompt with:

```py
print(rank_recommendation.build_string(entries=my_entries, query="Romantic comedy"))

```
For additional details, refer to the [Documentation](https://helmanofer.github.io/pydantic-prompter)
