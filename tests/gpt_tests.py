import logging
from pprint import pprint

import pytest

from pydantic_prompter.exceptions import OpenAiAuthenticationError
from tests.prompts_for_tests import *

logging.basicConfig()


def test_pydantic_result():
    try:
        res: Hey = bbb(name="Ofer")
        assert isinstance(res, Hey)
        assert res.name == "Ofer"
    except OpenAiAuthenticationError:
        pytest.skip("unsupported configuration")


def test_generic_result():
    try:
        res: MyChildren = aaa(name="Ofer")
        pprint(res)
        assert isinstance(res, MyChildren)
        assert res.num_of_children == 3
        assert res.children_names == ["aa", "bb", "cc"]
    except OpenAiAuthenticationError:
        pytest.skip("unsupported configuration")


def test_non_yaml_result():
    try:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)
    except OpenAiAuthenticationError as e:
        pytest.skip("unsupported configuration")


def test_complex_question_result():
    try:
        res = rank_recommendation(json_entries=entries, query=query)
        assert isinstance(res, RecommendationResults)
        pprint(res)
    except OpenAiAuthenticationError:
        pytest.skip("unsupported configuration")
