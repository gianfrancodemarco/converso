from typing import Dict, Optional, Type

import faker
from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from converso.clients import CreateCalendarEventPayload

fake = faker.Faker()


class GoogleCalendarCreatorEvaluation(StructuredTool):
    name = "GoogleCalendarCreator"
    description = """Useful to create events/memos/reminders on Google Calendar."""
    args_schema: Type[BaseModel] = CreateCalendarEventPayload
    chat_id: Optional[str] = None

    def _run(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"
