import logging
import os
from typing import List
import textwrap
from pydantic import BaseModel, Field

from pydantic_prompter import Prompter
from pydantic_prompter.prompter import Message
from tests.settings import Settings

logging.basicConfig()
run_gpt = Settings().openai_api_key is not None
if run_gpt:
    os.environ["OPENAI_API_KEY"] = Settings().openai_api_key

run_bedrock = True
try:
    import boto3

    session = boto3.Session(profile_name="okta_prod", region_name="us-east-1")
    client = session.client("bedrock")
except:
    run_bedrock = False


def test_pydantic():
    class Hey(BaseModel):
        name: str = Field(description="the name")
        children: List[str] = Field(description="list of my children")

    @Prompter(jinja=True, llm="openai", model_name="gpt-3.5-turbo")
    def bbb(name) -> Hey:
        """
        - system: you are a writer
        - user: hi, my name is {{ name }} and my children are called, aa, bb, cc
        - user: |
            what is my name and my children name
        """

    expected = [
        Message(role="system", content="you are a writer"),
        Message(
            role="user",
            content="hi, my name is Ofer and my children are called, aa, bb, cc",
        ),
        Message(role="user", content="what is my name and my children name"),
    ]
    res = bbb.build_prompt(name="Ofer")
    assert res == expected
    if run_gpt:
        res: Hey = bbb(name="Ofer")
        assert isinstance(res, Hey)
        assert res.name == "Ofer"


def test_generic():
    class MyChildren(BaseModel):
        num_of_children: int
        children_names: List[str]

    @Prompter(llm="openai", model_name="gpt-3.5-turbo")
    def aaa(name) -> MyChildren:
        """
        - user: hi, my name is {name}  and my children are called, aa, bb, cc
        - user: |
            how many children do I have?
        """

    expected = [
        Message(
            role="user",
            content="hi, my name is Ofer  and my children are called, aa, bb, cc",
        ),
        Message(role="user", content="how many children do I have?"),
    ]
    res = aaa.build_prompt(name="Ofer")
    assert res == expected
    if run_gpt:
        res: MyChildren = aaa(name="Ofer")
        assert isinstance(res, MyChildren)
        assert res.num_of_children == 3
        assert res.children_names == ["aa", "bb", "cc"]


def test_non_yaml():
    class QueryGPTResponse(BaseModel):
        google_like_search_term: str

    @Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo")
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
    if run_gpt:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)


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
    if run_bedrock:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)
