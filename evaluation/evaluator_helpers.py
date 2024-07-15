import json
import os
from datetime import datetime
from typing import Any, Dict

from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from converso.conversational_engine.form_agent.form_agent_executor import \
    FormAgentExecutor

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")


class MaxIterationsReached(Exception):
    pass


class SuccessfulExecution(Exception):
    pass


class ConversationAborted(Exception):
    """
    This exception is raised when the conversation reaches an unrecoverable state.
    For example, when the agents are stuck in a loop saying Goodbye to each other.
    """
    pass


def normalize_json(json_data):
    """
    The LLM can call the correct tool with the correct output, but it may differ from the expected one by small details.
    For examples, the dates may be in different formats, or the whitespace may be different, there may be dots at the end of the sentences, etc.

    This function normalizes the JSON string so that it can be compared with the expected output.
    We assume that these small differences are not relevant for the evaluation.

    Also, remove attributes that end in _, as by convention they are not relevant for the evaluation.
    """

    json_data = {key: value for key,
                 value in json_data.items() if not key.endswith('_')}

    for key, value in json_data.items():
        if key in ['start', 'end']:
            json_data[key] = datetime.fromisoformat(
                value.replace('T', ' ').replace("Z", '')).strftime('%Y-%m-%d %H:%M:%S')

        if isinstance(value, str):
            json_data[key] = json_data[key].replace("\n", " ").replace(".", " ").replace(
                ",", " ").replace("  ", " ").lower().strip()

    # Normalize whitespace
    json_data = {key: value.strip() if isinstance(
        value, str) else value for key, value in json_data.items()}
    print(json_data)
    return json_data


class UserLLMForEvaluation:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            verbose=True
        )
        self.history = []

    def execute_first(self, message):
        response = self.llm([SystemMessage(content=message)])
        self.history.extend([
            SystemMessage(content=message),
            response
        ])
        return response.content

    def execute(self, message):
        response = self.llm([
            *self.history,
            HumanMessage(content=message)
        ])
        self.history.extend([
            HumanMessage(content=message),
            response
        ])
        return response.content


class ExecutorForEvaluation:
    def __init__(
        self,
        tools: list,
        target_tool_call: Dict[str, Any] = None,
    ):
        self.target_tool_call = target_tool_call
        self.max_iterations = 10
        self.current_iteration = 1
        self.tools = tools
        self.graph = FormAgentExecutor(tools=self.tools)
        self.state = {
            "input": "",
            "chat_history": [],
            "intermediate_steps": [],
            "active_form_tool": None
        }

    def execute(self, input_data):

        inputs = {
            "input": input_data,
            "chat_history": self.state["chat_history"],
            "intermediate_steps": self.state["intermediate_steps"],
            "active_form_tool": self.state["active_form_tool"]
        }

        for output in self.graph.app.stream(
                inputs, config={"recursion_limit": 25}):
            for key, value in output.items():
                self.check_successful_execution(key, value)
        output = self.graph.parse_output(output)

        # Update state
        self.state.update({
            "chat_history": [
                *self.state["chat_history"],
                HumanMessage(content=input_data),
                AIMessage(content=output)
            ],
            "active_form_tool": value["active_form_tool"]
        })

        if self.current_iteration >= self.max_iterations:
            raise MaxIterationsReached()

        self.current_iteration += 1

        return output

    def check_successful_execution(self, key, value):
        raise NotImplementedError()


class FormAgentExecutorForEvaluation(ExecutorForEvaluation):
    """
    This class is used to execute the form agent in the conversational engine.

    Args:
        target_tool_call (Dict[str, Any], optional): The target tool call that we want to reach.
        The executor will raise a SuccessfulExecution exception if the target tool call is reached, or
        a MaxIterationsReached exception if the maximum number of iterations is reached.
    """

    def check_successful_execution(self, key, value):
        """
        Check if the target tool call is reached. If it is, raise a SuccessfulExecution exception.
        This is done by checking that when the agent is confirming the tool call, the current form of the FormTool is the same as the expected one.
        When checking the form, the JSON is normalized string so that it can be compared with the expected output.
        """

        if key != "agent":
            return

        if not isinstance(value["agent_outcome"], list):
            return

        if not isinstance(value["agent_outcome"][0], OpenAIToolAgentAction):
            return

        agent_outcome = value["agent_outcome"][0]
        target_tool_name = f"{self.target_tool_call['tool']}Finalize"

        if not agent_outcome.tool == target_tool_name:
            return

        if not agent_outcome.tool_input == {'confirm': True}:
            return

        target_tool = next(filter(
            lambda tool: tool.name == target_tool_name,
            self.graph._tools
        ))

        normalized_expected_output = normalize_json(
            self.target_tool_call['payload'])
        normalized_actual_output = normalize_json(
            json.loads(target_tool.form.model_dump_json()))

        if normalized_expected_output == normalized_actual_output:
            raise SuccessfulExecution()


class BasicAgentExecutorForEvaluation(ExecutorForEvaluation):
    """
    This class is used to execute the basic agent in the conversational engine.

    Args:
        target_tool_call (Dict[str, Any], optional): The target tool call that we want to reach.
        The executor will raise a SuccessfulExecution exception if the target tool call is reached, or
        a MaxIterationsReached exception if the maximum number of iterations is reached.
    """

    def check_successful_execution(self, key, value):
        """
        Check if the target tool call is reached. If it is, raise a SuccessfulExecution exception.
        """

        if key != "agent":
            return

        if not isinstance(value["agent_outcome"], list):
            return

        if not isinstance(value["agent_outcome"][0], OpenAIToolAgentAction):
            return

        agent_outcome = value["agent_outcome"][0]
        target_tool_name = self.target_tool_call['tool']

        if not agent_outcome.tool == target_tool_name:
            return

        normalized_expected_output = normalize_json(
            self.target_tool_call['payload'])
        normalized_actual_output = normalize_json(agent_outcome.tool_input)

        if normalized_expected_output == normalized_actual_output:
            raise SuccessfulExecution()


class EvaluationLogger:

    def __init__(
        self,
        type: str
    ) -> None:
        self.type = type
        self.logfile = os.path.join(
            os.path.dirname(__file__), "logs", self.type, f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.json")

        self.log = {
            "id": None,
            "prompt": None,
            "use_case": None,
            "messages": [],
            "result": None
        }

    def start_new_log(self, id, prompt, use_case):
        self.log = {
            "id": id,
            "prompt": prompt,
            "use_case": use_case,
            "messages": [],
            "result": None
        }

    def log_ai_message(self, message):
        print(message)
        self.log["messages"].append({
            "AI": message
        })

    def log_user_message(self, message):
        print(message)
        self.log["messages"].append({
            "User": message
        })

    def log_result(self, result):
        print(result)
        self.log["result"] = result

    def dump(self):
        if not os.path.exists(self.logfile):
            open(self.logfile, "w").write("[]")

        logs = json.loads(open(self.logfile).read())
        logs.append(self.log)
        with open(self.logfile, "w") as f:
            f.write(json.dumps(logs, indent=4))
