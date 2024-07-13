from typing import Any
from unittest.mock import MagicMock

from langchain_core.exceptions import OutputParserException

from converso.conversational_engine.form_agent.form_tool import (
    AgentState, FormReset)
from converso.conversational_engine.form_agent.form_agent_executor import *

from .mocks import *


class MockFormAgentExecutorOkModel(FormAgentExecutor):
    def build_model(self, state):
        agent = MagicMock()
        agent.invoke = MagicMock(return_value="Mocked response")
        return agent


class MockFormAgentExecutorErrorModel(FormAgentExecutor):
    def build_model(self, state):
        agent = MagicMock()
        agent.invoke = MagicMock(
            side_effect=OutputParserException("Mocked error"))
        return agent


class MockToolError(MockBaseTool):
    name = "MockToolError"
    description = "Mock tool that raises an error"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise Exception("Mocked error")


class TestFormAgentExecutor:

    def test_get_tools_no_active_form_tool(self):
        graph = FormAgentExecutor()
        state = AgentState()
        tools = graph.get_tools(state)
        assert len(tools) == 0

    def test_get_tools_with_active_form_tool(self):
        graph = FormAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        tools = graph.get_tools(state)
        assert len(tools) == 2
        assert tools[0] == active_form_tool
        assert tools[1] == FormReset()

    def test_get_tool_by_name_existing_tool_inactive(self):
        graph = FormAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        tool = graph.get_tool_by_name("MockFormToolStart", state)
        assert tool == active_form_tool

    def test_get_tool_by_name_non_existing_tool(self):
        graph = FormAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        tool = graph.get_tool_by_name("NonExistingTool", state)
        assert tool is None

    def test_should_continue_error(self):
        graph = FormAgentExecutor()
        state = AgentState()
        state["error"] = True
        result = graph.should_continue_after_agent(state)
        assert result == "error"

    def test_should_continue_end(self):
        graph = FormAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = AgentFinish({}, "end")
        result = graph.should_continue_after_agent(state)
        assert result == "end"

    def test_should_continue_tool(self):
        graph = FormAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = [AgentAction(
            tool="MockBaseTool", tool_input={}, log="")]
        result = graph.should_continue_after_agent(state)
        assert result == "tool"

    def test_should_continue_after_tool_error(self):
        graph = FormAgentExecutor()
        state = AgentState()
        state["error"] = True
        result = graph.should_continue_after_tool(state)
        assert result == "error"

    def test_should_continue_after_tool_return_direct_end(self):
        graph = FormAgentExecutor()
        state = AgentState(tool_outcome=FormToolOutcome(
            return_direct=True, output=""))
        result = graph.should_continue_after_tool(state)
        assert result == "end"

    def test_should_continue_after_tool_continue(self):
        graph = FormAgentExecutor()
        state = AgentState(tool_outcome=FormToolOutcome(
            return_direct=False, output=""))
        result = graph.should_continue_after_tool(state)
        assert result == "continue"

    def test_call_agent(self):
        graph = MockFormAgentExecutorOkModel()
        state = AgentState()
        state = {
            "input": "Hello",
            "chat_history": [],
            "intermediate_steps": [],
            "error": None
        }
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is None
        assert "tool_outcome" in response
        assert response["tool_outcome"] is None
        assert "agent_outcome" in response
        assert response["agent_outcome"] == "Mocked response"

    def test_call_agent_error(self):
        graph = MockFormAgentExecutorErrorModel()
        state = AgentState()
        state = {
            "input": "Hello",
            "chat_history": [],
            "intermediate_steps": [],
            "error": None
        }
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is not None

    def test_call_tool(self):
        graph = FormAgentExecutor(
            tools=[MockFormTool()],
        )
        state = AgentState()
        state["agent_outcome"] = [AgentAction(
            tool="MockFormToolStart", tool_input={}, log="")]
        response = graph.call_tool(state)
        assert "intermediate_steps" in response
        assert "error" in response
        assert response["error"] is None

    def test_call_tool_error(self):
        graph = FormAgentExecutor(
            tools=[MockToolError()],
        )
        state = AgentState()
        state["agent_outcome"] = [AgentAction(
            tool="MockToolError", tool_input={}, log="")]
        response = graph.call_tool(state)
        assert "intermediate_steps" in response
        assert "error" in response
        assert response["error"] is not None

    def test_on_tool_start_on_tool_end(self):
        on_tool_start = MagicMock()
        on_tool_end = MagicMock()
        graph = FormAgentExecutor(
            tools=[MockFormTool()],
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        state = AgentState(
            agent_outcome=[AgentAction(
                tool="MockFormToolStart", tool_input={}, log="")]
        )
        graph.call_tool(state)
        on_tool_start.assert_called_once()
        on_tool_end.assert_called_once()

    def test_call_agent_more_intermediate_steps(self):

        graph = MockFormAgentExecutorOkModel(
            tools=[MockFormTool()],
        )
        state = AgentState(
            input="Hello",
            agent_outcome=[AgentAction(
                tool="MockFormTool", tool_input={}, log="")],
            intermediate_steps=(
                (
                    AgentAction(tool="MockFormTool", tool_input={}, log=""),
                    FunctionMessage(
                        content="Tool output",
                        name="MockFormTool"
                    )),
            ) * 10,
        )
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is None
        assert "tool_outcome" in response
        assert response["tool_outcome"] is None
        assert "agent_outcome" in response
        assert response["agent_outcome"] == "Mocked response"

    def test_parse_output_tool_outcome(self):
        graph = FormAgentExecutor()
        graph_output = {END: {"tool_outcome": FormToolOutcome(
            return_direct=True, output="Tool output")}}
        output = graph.parse_output(graph_output)
        assert output == "Tool output"

    def test_parse_output_agent_outcome(self):
        graph = FormAgentExecutor()
        graph_output = {END: {"agent_outcome": AgentFinish(
            {"output": "Agent output"}, "end")}}
        output = graph.parse_output(graph_output)
        assert output == "Agent output"

    def test_parse_output_no_outcome(self):
        graph = FormAgentExecutor()
        graph_output = {END: {}}
        output = graph.parse_output(graph_output)
        assert output is None

    def test_build_model(self):
        graph = FormAgentExecutor()
        state = AgentState()
        model = graph.build_model(state)
        assert model is not None
