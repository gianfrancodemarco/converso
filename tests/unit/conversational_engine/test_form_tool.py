from langchain_core.tools import BaseTool

from converso.conversational_engine import (AgentState, FormReset, FormTool,
                                            FormToolState, filter_active_tools)

from .mocks import MockBaseTool, MockFormTool


def test_filter_active_tools_no_active_form_tool():
    tools = [
        MockBaseTool(),
        MockBaseTool(),
        MockFormTool(),
        MockFormTool(),
    ]
    agent_state = AgentState()

    filtered_tools = filter_active_tools(tools, agent_state)

    assert len(filtered_tools) == 4
    assert isinstance(filtered_tools[0], BaseTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert isinstance(filtered_tools[2], FormTool)
    assert filtered_tools[2].state == FormToolState.INACTIVE
    assert isinstance(filtered_tools[3], FormTool)
    assert filtered_tools[3].state == FormToolState.INACTIVE


def test_filter_active_tools_with_active_form_tool():

    active_form_tool = MockFormTool()

    tools = [
        MockBaseTool(),
        MockBaseTool(),
        active_form_tool,
        MockFormTool(),
    ]
    agent_state = AgentState()
    agent_state["active_form_tool"] = active_form_tool

    filtered_tools = filter_active_tools(tools, agent_state)

    assert len(filtered_tools) == 4
    assert isinstance(filtered_tools[0], BaseTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert filtered_tools[2] == active_form_tool
    assert isinstance(filtered_tools[3], FormReset)
