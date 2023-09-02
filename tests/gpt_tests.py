import logging
from typing import List

import pytest
from pydantic import BaseModel, Field

from pydantic_prompter import Prompter
from pydantic_prompter.exceptions import OpenAiAuthenticationError
from pydantic_prompter.prompter import Message

logging.basicConfig()


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


def test_pydantic():
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


def test_pydantic_result():
    try:
        res: Hey = bbb(name="Ofer")
        assert isinstance(res, Hey)
        assert res.name == "Ofer"
    except OpenAiAuthenticationError:
        pytest.skip("unsupported configuration")


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


def test_generic():
    expected = [
        Message(
            role="user",
            content="hi, my name is Ofer  and my children are called, aa, bb, cc",
        ),
        Message(role="user", content="how many children do I have?"),
    ]
    res = aaa.build_prompt(name="Ofer")
    assert res == expected


def test_generic_result():
    try:
        res: MyChildren = aaa(name="Ofer")
        assert isinstance(res, MyChildren)
        assert res.num_of_children == 3
        assert res.children_names == ["aa", "bb", "cc"]
    except OpenAiAuthenticationError:
        pytest.skip("unsupported configuration")


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


def test_non_yaml():
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


def test_non_yaml_result():
    try:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)
    except OpenAiAuthenticationError as e:
        pytest.skip("unsupported configuration")
