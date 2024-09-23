from datetime import datetime
from typing import Dict, Union

import faker

from converso.conversational_engine.tools import GmailRetriever

fake = faker.Faker()


class GmailRetrieverEvaluation(GmailRetriever):
    # We need to override the skip_confirm attribute from the parent class for
    # evaluation purposes
    skip_confirm = False

    def _run_when_complete(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        return {
            "number_of_emails": fake.random_int(min=1, max=10)
        }
