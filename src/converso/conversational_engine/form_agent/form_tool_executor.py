from langgraph.prebuilt.tool_executor import *

from converso.conversational_engine.form_agent.form_tool import (
    AgentState, FormToolOutcome)


class FormToolExecutor(ToolExecutor):

    def _execute(
        self,
        tool_invocation: ToolInvocationInterface,
        agent_state: AgentState = None,
    ) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join(
                    [t.name for t in self.tools]),
            )

        tool = self.tool_map[tool_invocation.tool]
        output = tool.invoke(
            tool_invocation.tool_input,
            agent_state=agent_state)
        output = self._parse_tool_outcome(output, tool)
        return output

    def _parse_tool_outcome(
        self,
        output: Union[str, FormToolOutcome],
        tool: BaseTool
    ):
        if isinstance(output, str):
            return FormToolOutcome(
                state_update={},
                output=output,
                return_direct=tool.return_direct,
            )

        if isinstance(output, FormToolOutcome):
            return output

        raise ValueError(f"""Tool returned an invalid output: {
                         output}. Must return a string or a FormToolOutcome.""")
