# Welcome to Pydantic Prompter

This library helps you build prompts easily using Pydantic

This library is using [OpenAi function calling API](https://platform.openai.com/docs/guides/gpt/function-calling)

The library's API was inspired by [DeclarAI](https://github.com/vendi-ai/declarai)


## Usage
```py
--8<-- "docs/example1.py"
```

## Best practices

Explicitly state the parameter name you want to get, in this example, `title`

```py hl_lines="2"
class RecommendationTitleResponse(BaseModel):
    title: str = Field(description="4 to 6 words title")


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def recommendation_title(json_entries) -> RecommendationTitleResponse:
    """
    - user: >
        Based on the JSON entries, suggest a minimum 4 words and maximum 6 words title

    - user: >
        The JSON entries:
        {{ json_entries }}

    """

```
Don't do this

```py hl_lines="2"
class BaseResponse(BaseModel):
    text: str = Field(description="4 to 6 words text")


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def recommendation_title(json_entries) -> BaseResponse:
    """
    ...
    """

```
