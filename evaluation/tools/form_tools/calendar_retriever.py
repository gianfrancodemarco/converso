
import random
from datetime import datetime, timedelta
from typing import Dict, Union

import faker

from converso.conversational_engine.tools import GoogleCalendarRetriever

fake = faker.Faker()


class GoogleCalendarRetrieverEvaluation(GoogleCalendarRetriever):
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
        start = fake.date_time_this_month()
        start = start.replace(second=0, microsecond=0)

        return {
            "start": start,
            "end": start + timedelta(days=random.randint(1, 7), hours=random.randint(1, 3), minutes=random.randint(0, 59))
        }
