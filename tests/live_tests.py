import logging

import pytest
import yaml

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
        # ("bedrock", "meta.llama2-70b-chat-v1"),
        # ("bedrock", "meta.llama2-13b-chat-v1"),
        ("bedrock", "cohere.command-r-v1:0"),
        ("bedrock", "cohere.command-r-v1:0"),
        # ("openai", "gpt-3.5-turbo"),
        # ("openai", "gpt-35-turbo-16k"),
        # ("bedrock", "anthropic.claude-instant-v1"),
        # ("bedrock", "anthropic.claude-v2"),
        # ("bedrock", "anthropic.claude-3-sonnet-20240229-v1:0"),
        # ("cohere", "command"),
        # ("cohere", "command-light"),
        # ("ollama", "llama3"),
        # ("ollama", "llama2:13b"),
    ],
)


@pytest.mark.live
@param_tests
def test_pydantic_result(llm, model):
    @Prompter(jinja=True, provider=llm, model_name=model)
    def bbb(name) -> PersonalInfo:
        """
        - user:
            hi, my name is {{ name }} and my children are called, aa, bb, cc
            what is my name and my children name
        """

    try:
        res: PersonalInfo = bbb(name="Ofer")
        assert isinstance(res, PersonalInfo)
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
    @Prompter(provider=llm, model_name=model, jinja=True)
    def search_query(history) -> QueryGPTResponse:
        """
        {{ history }}

        - user:
            Generate a Google-like search query text encompassing all previous chat questions and answers
        """

    history = [
        "- user: Hi",
        "- assistant: what genre do you want to watch?",
        "- user: Comedy",
        "- assistant: do you want a movie or series?",
        "- user: Movie",
        "- assistant: OK",
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
    @Prompter(provider=llm, jinja=True, model_name=model)
    def rank_recommendation(json_entries, query) -> RecommendationResults:
        """
        - user:
            Which of the following YAML entries fit best to the query. order by best fit descending
            Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries

            The YAML entries:
            {{ json_entries }}

            Query - {{ query }}

        """

    try:
        res = rank_recommendation(json_entries=yaml.dump(entries), query=query)
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
    @Prompter(jinja=True, provider=llm, model_name=model)
    def bbb(name) -> str:
        """
        - user:
            hi, my name is {{ name }} and my children are called, aa, bb, cc
            what is my name and my children name
        """

    try:
        res = bbb(name="Ofer")
        assert isinstance(res, str)
        assert "Ofer" in res
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
    @Prompter(jinja=True, provider=llm, model_name=model)
    def bbb(name) -> int:
        """
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
    @Prompter(jinja=True, provider=llm, model_name=model)
    def bbb(name) -> bool:
        """
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
