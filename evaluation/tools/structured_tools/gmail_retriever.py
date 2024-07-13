from typing import Optional, Type

from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from converso.clients import GetEmailsPayload


class GmailRetrieverEvaluation(StructuredTool):
    name = "GmailRetriever"
    description = """Useful to retrieve emails from Gmail"""
    args_schema: Type[BaseModel] = GetEmailsPayload

    return_direct = True
    chat_id: Optional[str] = None

    def _run(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"
