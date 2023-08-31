import logging
import os
from typing import List

from pydantic import BaseModel, Field

from pydatic_prompter import Prompter
from tests.settings import Settings

os.environ["OPENAI_API_KEY"] = Settings().openai_api_key
logging.basicConfig()


class Hey(BaseModel):
    name: str = Field(description="the name")
    children: List[str] = Field(description="list of my children")


def test_pydantic():
    @Prompter(jinja=True, llm="openai", model_name="gpt-3.5-turbo")
    def bbb(name) -> Hey:
        """
        >> system: you are a writer
        >> user: hi, my name is {{ name }} and my children are called, aa, bb, cc
        >> user: |
            what is my name and my children name
        """

    res: Hey = bbb(name="Ofer")
    assert isinstance(res, Hey)
    assert res.name == "Ofer"


class MyChildren(BaseModel):
    num_of_children: int
    children_names: List[str]


def test_generic():
    @Prompter(llm="openai", model_name="gpt-3.5-turbo")
    def aaa(name) -> MyChildren:
        """
        >> user: hi, my name is {name}  and my children are called, aa, bb, cc
        >> user: |
            how many children do I have
        """

    res: MyChildren = aaa(name="Ofer")
    assert isinstance(res, MyChildren)
    assert res.num_of_children == 3
    assert res.children_names == ["aa", "bb", "cc"]
