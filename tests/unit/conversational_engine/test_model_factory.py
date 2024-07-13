from langchain_core.prompts.chat import ChatPromptTemplate

from converso.conversational_engine.form_agent import (AgentState,
                                                       FormAgentExecutor,
                                                       ModelFactory)

from .mocks import *


class TestModelFactory:
    def test_build_model_default_prompt(self):
        state = AgentState()
        model = ModelFactory.build_model(state=state)
        prompt_template = model.steps[1].messages[0].prompt.template
        basic_template = "You are a personal assistant trying to help the user. You always answer in English."
        assert prompt_template.startswith(basic_template)

    def test_build_model_active_form_tool(self):
        state = AgentState()
        active_form_tool = MockFormTool()
        active_form_tool.enter_active_state()
        state["active_form_tool"] = active_form_tool
        model = ModelFactory.build_model(state=state)
        assert isinstance(model.steps[1], ChatPromptTemplate)

    def test_build_model_active_form_tool_information_to_collect(self):
        state = AgentState()
        active_form_tool = MockFormToolWithFields()
        active_form_tool.enter_active_state()
        state["active_form_tool"] = active_form_tool
        model = ModelFactory.build_model(state=state)
        assert isinstance(model.steps[1], ChatPromptTemplate)

    def test_build_model_error(self):
        state = AgentState()
        active_form_tool = MockFormTool()
        active_form_tool.enter_active_state()
        state.update({
            "active_form_tool": active_form_tool,
            "error": "Mocked error"
        })
        model = ModelFactory.build_model(state=state)
        assert isinstance(model.steps[1], ChatPromptTemplate)
