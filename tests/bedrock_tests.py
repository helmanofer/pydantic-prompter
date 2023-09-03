import logging
import textwrap
from pprint import pprint

import pytest

from pydantic_prompter.exceptions import BedRockAuthenticationError
from tests.prompts_for_tests import *

logging.basicConfig(level=logging.INFO)


def test_anthropic_generic_result():
    try:
        res: MyChildren = aaa(name="Ofer")
        assert isinstance(res, MyChildren)
        assert res.num_of_children == 3
        assert res.children_names == ["aa", "bb", "cc"]
    except BedRockAuthenticationError:
        pytest.skip("unsupported configuration")


def test_anthropic_non_yaml():
    qstr = search_query.build_string(history="\n".join(history))
    qstr_expected = textwrap.dedent(
        """Human: You are a REST API that answers the question contained in <question> tags.
Your response should be in a JSON format which it's schema is specified in the <json> tags. DO NOT add any other text other than the JSON

<json>
{
    "title": "QueryGPTResponse",
    "type": "object",
    "properties": {
        "google_like_search_term": {
            "title": "Google Like Search Term",
            "type": "string"
        }
    },
    "required": [
        "google_like_search_term"
    ]
}
</json>
<question>
Assistant: what genre do you want to watch?
Human: Comedy
Assistant: do you want a movie or series?
Human: Movie
Human: Generate a Google-like search query text encompassing all previous chat questions and answers
</question>

Assistant:"""
    )
    assert qstr == qstr_expected


def test_anthropic_non_yaml_result():
    try:
        res = search_query(history="\n".join(history))
        assert isinstance(res, QueryGPTResponse)
    except BedRockAuthenticationError:
        pytest.skip("unsupported configuration")


def test_anthropic_complex_question_result():
    try:
        res = rank_recommendation(json_entries=entries, query=query)
        assert isinstance(res, RecommendationResults)
        pprint(res)
    except BedRockAuthenticationError:
        pytest.skip("unsupported configuration")
