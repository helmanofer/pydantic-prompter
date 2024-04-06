# Welcome to Pydantic Prompter

Pydantic Prompter is a lightweight tool designed for effortlessly constructing prompts and obtaining Pydantic objects as outputs.


Seamlessly call LLMs like functions in Python with Pydantic Prompter. 
It handles prompt creation and output parsing to custom models for providers like Cohere, 
Bedrock, and OpenAI. Get [OpenAi function calling API](https://platform.openai.com/docs/guides/gpt/function-calling) capabilities for any LLM. 
Structured text generation with less code.

The design of the library's API draws inspiration by [DeclarAI](https://github.com/vendi-ai/declarai).
Other alternatives [Outlines](https://github.com/outlines-dev/outlines) and [Jsonformer](https://github.com/1rgs/jsonformer)

### Why should you use Pydantic Prompter
💻 **Seamless LLM Integration**: Pydantic Prompter supported multiple LLM providers, including Cohere, Bedrock, and OpenAI, right out of the box. This meant we could easily switch between providers without modifying our code, ensuring flexibility and portability.

📦 **Structured Outputs**: By leveraging Pydantic models, Pydantic Prompter automatically parsed the LLM's output into structured Python objects. Manual parsing became a thing of the past, and we enjoyed consistently formatted data that was a breeze to work with.

✍️ **Easy Prompt Engineering**: Crafting effective prompts is an art, and Pydantic Prompter made us all masters. By defining prompts using Python classes and string interpolation, we created readable, maintainable, and reusable prompts.

🔧 **Reusable Components**: Pydantic Prompter encouraged a modular approach, allowing us to define reusable prompt components such as instructions, examples, and constraints. This promoted code reuse and made maintaining our code effortless.

🐛 **Logging and Debugging**: Built-in logging and debugging features meant we could quickly identify and resolve any issues, ensuring a smooth and efficient development process, free of bugs and errors.


## Install
#### OpenAI
```python
pip install 'pydantic-prompter[openai]'
```

#### Bedrock
```python
pip install 'pydantic-prompter[bedrock]'
```

#### Cohere
```python
pip install 'pydantic-prompter[cohere]'
```



## Usage
#### Basic usage
To utilize Pydantic Prompter with Jinja2 templates, follow the example below:

```py
--8<-- "examples/rank_movies.py"
```
#### Simple string formatting
For injecting conversation history through straightforward string formatting, refer to this example:

```py
--8<-- "examples/history_injection.py"
```
#### Jinja2 advance usage
For more advanced usage involving Jinja2 loops to inject conversation history, consider the following code snippet:

```py hl_lines="13-15"
--8<-- "examples/history_injection_jinja.py"
```

#### Simple typings
Use `int`, `float`, `bool` or `str`
```py hl_lines="11"
--8<-- "examples/simple_typings.py"
```

## Best practices

When using Pydantic Prompter, it is recommended to explicitly specify the parameter name you wish to retrieve, as demonstrated in the example below, where title is explicitly mentioned:

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
Avoid the following practice where the parameter name is not explicitly stated:


```py hl_lines="2"
class BaseResponse(BaseModel):
    text: str = Field(description="4 to 6 words text")


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def recommendation_title(json_entries) -> BaseResponse:
    """
    ...
    """

```
Adhering to these best practices will ensure clarity and precision when using Pydantic Prompter.

## Debugging and logging

You can view info and/or debugging logging using the following snippet:

```py
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("pydantic_prompter").setLevel(logging.DEBUG)
```
Resulting
```console
DEBUG:pydantic_prompter:Using OpenAI provider with model gpt-3.5-turbo
DEBUG:pydantic_prompter:Using PydanticParser
DEBUG:pydantic_prompter:Calling with prompt: 
 {'role': 'user', 'content': 'say hi'}
DEBUG:pydantic_prompter:Response from llm: 
 {
  "response": "Hi there!"
 }
```

