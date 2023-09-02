from pydantic import BaseModel

from pydantic_prompter import Prompter


class QueryGPTResponse(BaseModel):
    google_like_search_term: str


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo")
def search_query(history) -> QueryGPTResponse:
    """
    {%- for line in history %}
     {{ line }}
    {% endfor %}

    - user: |
        Generate a Google-like search query text
        encompassing all previous chat questions and answers
    """


history = [
    "- assistant: what genre do you want to watch?",
    "- user: Comedy",
    "- assistant: do you want a movie or series?",
    "- user: Movie",
]
res = search_query.build_string(history=history)
print(res)

# >>> assistant: what genre do you want to watch?
#     user: Comedy
#     assistant: do you want a movie or series?
#     user: Movie
#     user: Generate a Google-like search query text
#     encompassing all previous chat questions and answers
