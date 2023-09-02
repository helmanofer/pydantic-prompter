import logging
import textwrap
from typing import List

from pydantic import BaseModel, Field

from pydantic_prompter import Prompter
from pydantic_prompter.exceptions import BedRockAuthenticationError
from pydantic_prompter.prompter import Message

logging.basicConfig()


def test_anthropic_generic():
    class MyChildren(BaseModel):
        num_of_children: int
        children_names: List[str] = Field(description="The names of my children")

    @Prompter(llm="bedrock", model_name="anthropic.claude-instant-v1")
    def aaa(name) -> MyChildren:
        """
        - user: hi, my name is {name} and my children are called, aa, bb, cc
        - user: |
            how many children do I have and what's their names?
        """

    expected = [
        Message(
            role="user",
            content="hi, my name is Ofer and my children are called, aa, bb, cc",
        ),
        Message(
            role="user", content="how many children do I have and what's their names?"
        ),
    ]
    res = aaa.build_prompt(name="Ofer")
    assert res == expected
    if run_bedrock:
        res: MyChildren = aaa(name="Ofer")
        assert isinstance(res, MyChildren)
        assert res.num_of_children == 3
        assert res.children_names == ["aa", "bb", "cc"]


def test_anthropic_non_yaml():
    class QueryGPTResponse(BaseModel):
        google_like_search_term: str

    @Prompter(llm="bedrock", model_name="anthropic.claude-instant-v1", jinja=True)
    def search_query(history) -> QueryGPTResponse:
        """
        {{ history }}

        - user: |
            Generate a Google-like search query text encompassing all previous chat questions and answers
        """

    history = [
        "- assistant: what genre do you want to watch?",
        "- user: Comedy",
        "- assistant: do you want a movie or series?",
        "- user: Movie",
    ]
    res = search_query.build_prompt(history="\n".join(history))
    expected = [
        Message(role="assistant", content="what genre do you want to watch?"),
        Message(role="user", content="Comedy"),
        Message(role="assistant", content="do you want a movie or series?"),
        Message(role="user", content="Movie"),
        Message(
            role="user",
            content="Generate a Google-like search query "
            "text encompassing all previous chat questions and answers",
        ),
    ]
    assert res == expected
    qstr = search_query.build_string()
    qstr_expected = textwrap.dedent(
        """Human: You are a REST API that answers the question contained in <question> tags.
Your response should be in the JSON which it's schema is specified in the <json> tags.

<json>
{
    "google_like_search_term": {
        "title": "Google Like Search Term",
        "type": "string"
    }
}
</json>
<question>
Human: Generate a Google-like search query text encompassing all previous chat questions and answers
</question>

Assistant:"""
    )
    assert qstr == qstr_expected
    try:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)
    except BedRockAuthenticationError:
        print("\nSkipped actual run")
