# Pydantic Prompter
Crafting Prompts, Unlocking Structured Outputs
Pydantic Prompter is a lightweight tool designed for effortlessly constructing prompts and obtaining Pydantic objects as outputs.

Pydantic Prompter simplifies the art of prompt engineering, offering a seamless experience as you call LLMs like regular functions. 
With Pydantic Prompter, you unlock structured text generation and effortlessly obtain Pydantic objects as outputs.

The design of the library's API draws inspiration by [DeclarAI](https://github.com/vendi-ai/declarai).
Other alternatives [Outlines](https://github.com/outlines-dev/outlines) and [Jsonformer](https://github.com/1rgs/jsonformer)

📄 Documentation https://helmanofer.github.io/pydantic-prompter

### Why should you use Pydantic Prompter
💻 **Seamless LLM Integration**: Pydantic Prompter supported multiple LLM providers, including Cohere, Bedrock, and OpenAI, right out of the box. This meant we could easily switch between providers without modifying our code, ensuring flexibility and portability.

📦 **Structured Outputs**: By leveraging Pydantic models, Pydantic Prompter automatically parsed the LLM's output into structured Python objects. Manual parsing became a thing of the past, and we enjoyed consistently formatted data that was a breeze to work with.

✍️ **Easy Prompt Engineering**: Crafting effective prompts is an art, and Pydantic Prompter made us all masters. By defining prompts using Python classes and string interpolation, we created readable, maintainable, and reusable prompts.

🔧 **Reusable Components**: Pydantic Prompter encouraged a modular approach, allowing us to define reusable prompt components such as instructions, examples, and constraints. This promoted code reuse and made maintaining our code effortless.

🐛 **Logging and Debugging**: Built-in logging and debugging features meant we could quickly identify and resolve any issues, ensuring a smooth and efficient development process, free of bugs and errors.

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

### Diving into Pydantic Prompter

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

#### Creating a Prompt Function

Now, let's create a Pydantic Prompter function using the `@Prompter` decorator. 
You can define your prompt as a YAML string with Jinja2 templating or simple string formatting:

```py
from pydantic_prompter import Prompter


@Prompter(ai_provider="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
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
#### Executing Your Prompt
Execute your prompt function just like a regular Python function:

```py
my_entries = "[{\"text\": \"Description: Four everyday suburban guys come together as a ...."
print(rank_recommendation(entries=my_entries, query="Romantic comedy"))

```
For debugging, you can inspect the generated prompt using:

```py
print(rank_recommendation.build_string(entries=my_entries, query="Romantic comedy"))

```

### Explore the Possibilities

With Pydantic Prompter, you're equipped to seamlessly integrate LLMs into your projects, 
whether it's generating content, answering queries, or building creative applications.
Dive into the [Documentation](https://helmanofer.github.io/pydantic-prompter) to explore more examples, 
learn about advanced features, and discover the full potential of Pydantic Prompter.

Contribute to the open-source project, share your feedback, and join the community to shape the future of prompt engineering!

Happy Prompting! 🌟💻🤖