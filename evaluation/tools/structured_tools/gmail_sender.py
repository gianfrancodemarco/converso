from typing import Optional, Type

from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from converso.clients import SendEmailPayload


class GmailSenderEvaluation(StructuredTool):
    name: str = "GmailSender"
    description: str = """Useful to send emails from Gmail"""
    args_schema: Type[BaseModel] = SendEmailPayload
    chat_id: Optional[str] = None

    def _run(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"
