from typing import Generic, TypeVar

from pydantic import BaseModel

from openai.lib._parsing._completions import type_to_response_format_param
from openai.lib._parsing._responses import type_to_text_format_param
from openai.lib._tools import pydantic_function_tool


T = TypeVar("T")


class Item(BaseModel):
    title: str


class Page(BaseModel, Generic[T]):
    items: list[T]


def test_parameterized_generic_names_are_api_safe() -> None:
    model = Page[Item]

    assert model.__name__ == "Page[Item]"

    completion = type_to_response_format_param(model)
    assert completion["json_schema"]["name"] == "Page_Item_"

    response = type_to_text_format_param(model)
    assert response["name"] == "Page_Item_"

    tool = pydantic_function_tool(model)
    assert tool["function"]["name"] == "Page_Item_"


def test_derived_name_is_capped_at_64_characters() -> None:
    class VeryLongModelNameWithCharactersThatShouldNotReachTheRemoteApiBoundary(BaseModel):
        value: str

    tool = pydantic_function_tool(VeryLongModelNameWithCharactersThatShouldNotReachTheRemoteApiBoundary)
    assert len(tool["function"]["name"]) <= 64
    assert all(char.isalnum() or char in "_-" for char in tool["function"]["name"])


def test_explicit_name_is_not_normalized() -> None:
    class ModelWithExplicitName(BaseModel):
        value: str

    tool = pydantic_function_tool(ModelWithExplicitName, name="caller.chosen")
    assert tool["function"]["name"] == "caller.chosen"
