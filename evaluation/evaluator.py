"""
Module used to evaluate the conversational engine.

The evaluation is done by using an LLM to simulate the user, and the conversational engine to be tested.

The module reads the test cases from the prompts.json file, and for each test case, it executes the conversational engine.
Each test case contains a prompt, a tool, and a payload.
The prompt is the system message for the LLM that simulates the user.
The tool is the tool that the conversational engine should call and the payload is the expected input for the tool.
A test is successful if the conversational engine calls the correct tool with the correct input.

For each output of every node of the conversational engine, the evaluator checks if the target tool call is reached.
If it is, the evaluator raises a SuccessfulExecution exception.
If the maximum number of iterations is reached, the evaluator raises a MaxIterationsReached exception and the test is considered failed.

The results of the evaluation are logged in the logs folder.

The evaluation is done for 2 different types of agents:
- the BasicAgent, which uses the structured tools from Langchain
- the FormAgent, which uses the form tools extension from converso
"""
import json
import os

from evaluator_helpers import *

os.environ["OPENAI_API_KEY"] = "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"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__f2958965106c4e02bb9d6ed6d2c11f36"
os.environ["LANGCHAIN_PROJECT"] = "converso-cost-analysis3"

TEST_CASES_PATH = os.path.join(
    os.path.dirname(__file__), "prompts/prompts.json")

EVALUATE_FORM_AGENT = True
if EVALUATE_FORM_AGENT:
    from tools.form_tools import *
    evaluation_logger = EvaluationLogger(type="form")
    SystemModelClass = FormAgentExecutorForEvaluation
else:
    from tools.structured_tools import *
    evaluation_logger = EvaluationLogger(type="basic")
    SystemModelClass = BasicAgentExecutorForEvaluation

test_cases = json.loads(open(TEST_CASES_PATH).read())

# Hack to resume after a crash
# test_cases = [test_cases[240:]]

test_cases = [
    test_case for test_case in test_cases if test_case["tool"] == "OnlinePurchase"]

for test_case in test_cases:
    try:
        id = test_case["id"]
        prompt = test_case["prompt"]
        tool = test_case["tool"]
        payload = test_case["payload"]
        use_case = test_case["use_case"]

        user_model = UserLLMForEvaluation()
        system_model = SystemModelClass(
            tools=[
                GoogleCalendarCreatorEvaluation(),
                GoogleCalendarRetrieverEvaluation(),
                GmailRetrieverEvaluation(),
                GmailSenderEvaluation(),
                OnlinePurchaseEvaluation()
            ],
            target_tool_call={
                "tool": tool,
                "payload": payload
            }
        )

        evaluation_logger.start_new_log(id, prompt, use_case)

        user_response = user_model.execute_first(prompt)
        evaluation_logger.log_user_message(user_response)

        while True:
            system_response = system_model.execute(user_response)
            evaluation_logger.log_ai_message(system_response)
            user_response = user_model.execute(system_response)
            evaluation_logger.log_user_message(user_response)

            if user_response in ["Goodbye!", "Thank you!",
                                 "Thank you, you too!", "Thank you! Goodbye!"]:
                raise ConversationAborted()
    except SuccessfulExecution:
        evaluation_logger.log_result("Successful execution")
    except MaxIterationsReached:
        evaluation_logger.log_result("Max iterations reached")
    except ConversationAborted:
        evaluation_logger.log_result("Conversation aborted")
    finally:
        evaluation_logger.dump()
