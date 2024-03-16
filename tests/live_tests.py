import logging

import pytest

from pydantic_prompter import Prompter
from pydantic_prompter.exceptions import (
    CohereAuthenticationError,
    OpenAiAuthenticationError,
    BedRockAuthenticationError,
)
from tests.data_for_tests import *

logger = logging.getLogger()

param_tests = pytest.mark.parametrize(
    "llm,model",
    [
        ("bedrock", "meta.llama2-70b-chat-v1"),
        # ("bedrock", "meta.llama2-13b-chat-v1"),
        ("bedrock", "cohere.command-text-v14"),
        ("openai", "gpt-3.5-turbo"),
        ("bedrock", "anthropic.claude-instant-v1"),
        ("bedrock", "anthropic.claude-v2"),
        ("cohere", "command"),
        ("cohere", "command-light"),
    ],
)


@pytest.mark.live
@param_tests
def test_pydantic_result(llm, model):
    @Prompter(jinja=True, llm=llm, model_name=model)
    def bbb(name) -> Hey:
        """
        - system: you are a writer
        - user:
            hi, my name is {{ name }} and my children are called, aa, bb, cc
            what is my name and my children name
        """

    try:
        res: Hey = bbb(name="Ofer")
        assert isinstance(res, Hey)
        assert res.name == "Ofer"
    except (
        OpenAiAuthenticationError,
        BedRockAuthenticationError,
        CohereAuthenticationError,
    ):
        print(bbb.build_string())
        logger.exception("")
        pytest.skip("unsupported configuration")


@pytest.mark.live
@param_tests
def test_non_yaml_result(llm, model):
    @Prompter(llm=llm, model_name=model, jinja=True)
    def search_query(history) -> QueryGPTResponse:
        """
        {{ history }}

        - user:
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
    except (
        OpenAiAuthenticationError,
        BedRockAuthenticationError,
        CohereAuthenticationError,
    ) as e:
        logger.warning(e)
        pytest.skip("unsupported configuration")


@pytest.mark.live
@param_tests
def test_complex_question_result(llm, model):
    @Prompter(llm=llm, jinja=True, model_name=model)
    def rank_recommendation(json_entries, query) -> RecommendationResults:
        """
        - user:
            Which of the following JSON entries fit best to the query. order by best fit descending
            Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries

            The JSON entries:
            {{ json_entries }}

            Query - {{ query }}

        """

    try:
        res = rank_recommendation(json_entries=entries, query=query)
        assert isinstance(res, RecommendationResults)
    except (
        OpenAiAuthenticationError,
        BedRockAuthenticationError,
        CohereAuthenticationError,
    ) as e:
        logger.warning(e)
        pytest.skip("unsupported configuration")


@pytest.mark.live
@param_tests
def test_str_result(llm, model):
    @Prompter(jinja=True, llm=llm, model_name=model)
    def bbb(name) -> str:
        """
        - system: you are a writer
        - user:
            hi, my name is {{ name }} and my children are called, aa, bb, cc
            what is my name and my children name
        """

    try:
        res = bbb(name="Ofer")
        print(res)
        assert isinstance(res, str)
        # assert res.name == "Ofer"
    except (
        OpenAiAuthenticationError,
        BedRockAuthenticationError,
        CohereAuthenticationError,
    ):
        print(bbb.build_string())
        logger.exception("")
        pytest.skip("unsupported configuration")


@pytest.mark.live
@param_tests
def test_int_result(llm, model):
    class Hey(BaseModel):
        res: int

    @Prompter(jinja=True, llm=llm, model_name=model)
    def bbb(name) -> int:
        """
        - system: you are a writer
        - user:
            hi, my name is {{ name }} and my children are called, aa, bb, cc
            how many children do I have?
        """

    try:
        res = bbb(name="Ofer")
        print(res)
        assert isinstance(res, int)
    except (
        OpenAiAuthenticationError,
        BedRockAuthenticationError,
        CohereAuthenticationError,
    ):
        print(bbb.build_string())
        logger.exception("")
        pytest.skip("unsupported configuration")


@pytest.mark.live
@param_tests
def test_bool_result(llm, model):
    @Prompter(jinja=True, llm=llm, model_name=model)
    def bbb(name) -> bool:
        """
        - system: you are a writer
        - user:
            hi, my name is {{ name }} and my children are called, aa, bb, cc
            do I have children?
        """

    try:
        res = bbb(name="Ofer")
        print(res)
        assert isinstance(res, bool)
    except (
        OpenAiAuthenticationError,
        BedRockAuthenticationError,
        CohereAuthenticationError,
    ):
        print(bbb.build_string())
        logger.exception("")
        pytest.skip("unsupported configuration")
