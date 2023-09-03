import logging
from pprint import pprint

import pytest

from pydantic_prompter import Prompter
from pydantic_prompter.exceptions import (
    OpenAiAuthenticationError,
    BedRockAuthenticationError,
)
from tests.data_for_tests import *

logging.basicConfig()


@pytest.mark.parametrize(
    "llm,model",
    [("openai", "gpt-3.5-turbo"), ("bedrock", "anthropic.claude-instant-v1")],
)
def test_pydantic_result(llm, model):
    @Prompter(jinja=True, llm=llm, model_name=model)
    def bbb(name) -> Hey:
        """
        - system: you are a writer
        - user: hi, my name is {{ name }} and my children are called, aa, bb, cc
        - user: |
            what is my name and my children name
        """

    try:
        res: Hey = bbb(name="Ofer")
        assert isinstance(res, Hey)
        assert res.name == "Ofer"
    except (OpenAiAuthenticationError, BedRockAuthenticationError):
        pytest.skip("unsupported configuration")


@pytest.mark.parametrize(
    "llm,model",
    [("openai", "gpt-3.5-turbo"), ("bedrock", "anthropic.claude-instant-v1")],
)
def test_generic_result(llm, model):
    @Prompter(llm=llm, model_name=model)
    def aaa(name) -> MyChildren:
        """
        - user: hi, my name is {name} and my children are called, aa, bb, cc
        - user: |
            how many children do I have and what's their names?
        """

    try:
        res: MyChildren = aaa(name="Ofer")
        pprint(res)
        assert isinstance(res, MyChildren)
        assert res.num_of_children == 3
        assert res.children_names == ["aa", "bb", "cc"]
    except (OpenAiAuthenticationError, BedRockAuthenticationError):
        pytest.skip("unsupported configuration")


@pytest.mark.parametrize(
    "llm,model",
    [("openai", "gpt-3.5-turbo"), ("bedrock", "anthropic.claude-instant-v1")],
)
def test_non_yaml_result(llm, model):
    @Prompter(llm=llm, model_name=model, jinja=True)
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

    try:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)
    except (OpenAiAuthenticationError, BedRockAuthenticationError):
        pytest.skip("unsupported configuration")


@pytest.mark.parametrize(
    "llm,model",
    [("openai", "gpt-3.5-turbo"), ("bedrock", "anthropic.claude-instant-v1")],
)
def test_complex_question_result(llm, model):
    @Prompter(llm=llm, jinja=True, model_name=model)
    def rank_recommendation(json_entries, query) -> RecommendationResults:
        """
        - user: >
            Which of the following JSON entries fit best to the query. order by best fit descending
            Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries

        - user: >
            The JSON entries:
            {{ json_entries }}

        - user: Query - {{ query }}

        """

    try:
        res = rank_recommendation(json_entries=entries, query=query)
        assert isinstance(res, RecommendationResults)
        pprint(res)
    except (OpenAiAuthenticationError, BedRockAuthenticationError):
        pytest.skip("unsupported configuration")
