from typing import Any, Type

from pydantic import BaseModel

from converso.conversational_engine.form_agent.form_tool import (
    BaseTool, FormTool)


class MockBaseTool(BaseTool):
    name: str = "MockBaseTool"
    description: str = "MockBaseTool description"

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        pass


class _DummyPayload(BaseModel):
    pass


class _DummyPayloadWithFields(BaseModel):
    name: set
    age: int


class MockFormTool(FormTool):
    name: str = "MockFormTool"
    description: str = "MockFormTool description"
    args_schema: Type[BaseModel] = _DummyPayload

    def _run_when_complete(self) -> Any:
        pass


class MockFormToolWithFields(FormTool):
    name: str = "MockFormToolWithFields"
    description: str = "MockFormToolWithFields description"
    args_schema: Type[BaseModel] = _DummyPayloadWithFields

    def _run_when_complete(self) -> Any:
        pass
