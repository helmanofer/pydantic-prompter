# Pydantic Prompter

Pydantic Prompter is a lightweight tool designed for effortlessly constructing prompts and obtaining Pydantic objects as outputs.


Seamlessly call LLMs like functions in Python with Pydantic Prompter. 
It handles prompt creation and output parsing to custom models for providers like Cohere, 
Bedrock, and OpenAI. Get [OpenAi function calling API](https://platform.openai.com/docs/guides/gpt/function-calling) capabilities for any LLM. 
Structured text generation with less code.

The design of the library's API draws inspiration by [DeclarAI](https://github.com/vendi-ai/declarai).
Other alternatives [Outlines](https://github.com/outlines-dev/outlines) and [Jsonformer](https://github.com/1rgs/jsonformer)

📄 Documentation https://helmanofer.github.io/pydantic-prompter

### Installation
To install Pydantic Prompter, use the following command:



```bash
pip install 'pydantic-prompter[openai]'
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
