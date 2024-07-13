from datetime import datetime
from typing import Dict, Union

import faker

from converso.conversational_engine.tools import GmailSender

fake = faker.Faker()


class GmailSenderEvaluation(GmailSender):
    def _run_when_complete(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        return {
            "to": fake.email(),
            "subject": fake.text(),
            "body": fake.text()
        }
