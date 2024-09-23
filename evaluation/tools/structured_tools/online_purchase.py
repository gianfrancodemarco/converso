from typing import Type

from langchain.tools import StructuredTool
from pydantic import BaseModel

from converso.conversational_engine.tools.online_purchase import \
    OnlinePurchasePayload


class OnlinePurchaseEvaluation(StructuredTool):
    name: str = "OnlinePurchase"
    description: str = """Purchase an item from an online store"""
    args_schema: Type[BaseModel] = OnlinePurchasePayload

    def _run(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"
