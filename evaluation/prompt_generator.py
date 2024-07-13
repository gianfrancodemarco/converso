import json
from textwrap import dedent

from tools.form_tools import *

NUMBER_OF_TESTS_PER_USE_CASE = 20

configs = {
    "GoogleCalendarCreator": {
        "tool": GoogleCalendarCreatorEvaluation(),
        "task": "create an event on Google Calendar"
    },
    "GoogleCalendarRetriever": {
        "tool": GoogleCalendarRetrieverEvaluation(),
        "task": "retrieve events from Google Calendar"
    },
    "GmailSender": {
        "tool": GmailSenderEvaluation(),
        "task": "send an email"
    },
    "GmailRetriever": {
        "tool": GmailRetrieverEvaluation(),
        "task": "retrieve emails"
    },
    "OnlinePurchase": {
        "tool": OnlinePurchaseEvaluation(),
        "task": "purchase an item"
    },
}

BASE_PROMPT = dedent("""
    You are impersonating a human user using an AI system.
    Your task is to communicate with an external system to {task}.
    Do not ever reveal that you are an AI.
    Follow the system instructions to complete your task.
                     
    The data necessary to {task} is the following:
    {data}
""")

USE_CASES = [
    {
        "name": "ALL_INFORMATION_FIRST_MESSAGE",
        "prompt": dedent("""

            State your will to the system, and then follow his instructions to complete the job.
            Give all the data necessary to the system in the first message.
            \n

            User:
        """)
    },
    {
        "name": "NO_INFORMATION_FIRST_MESSAGE",
        "prompt": dedent("""

            State your will to the system, without providing any data, and then follow his instructions to complete the job.
            For example "I want to create an event", or "I want to buy something".
            \n

            User:
        """)
    },
    {
        "name": "MAIN_INFORMATION_FIRST_MESSAGE",
        "prompt": dedent("""

            State what you want to do, providing only the main information, and then follow his instructions to complete the job.
            For example "I want to create an event called 'Meeting'", or "I want to buy a watch".
            \n

            User:
        """)
    },
    {
        "name": "CONFUSED_USER",
        "prompt": dedent("""

            State your will to the system, without providing any data, and then follow his instructions to complete the job.
            Act like a very naive user who doesn't know what to do, mispell words, give the wrong information and then correct it.
            \n

            User:
        """)
    }
]

def create_prompts():

    test_cases = []
    idx = 0

    for key, config in configs.items():
        task, tool = config["task"], config["tool"]
        for i in range(NUMBER_OF_TESTS_PER_USE_CASE):

            payload = tool.get_random_payload()
            payload_str = "\n".join([f"{key}: {value}" for key, value in payload.items() if value])
 
            for use_case in USE_CASES:

                prompt = dedent(f"""
                    {BASE_PROMPT.format(task=task, data=payload_str)}
                    {use_case["prompt"]}
                """)

                test_case = {
                    "id": idx,
                    "use_case": use_case["name"],
                    "prompt": prompt,
                    "tool": key,
                    "payload": payload
                }

                test_cases.append(test_case)
                print(test_case)

                idx += 1

    with open(f"evaluation/prompts/prompts.json", "w") as f:
        json.dump(test_cases, f, indent=4, default=str)


create_prompts()