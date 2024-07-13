import random
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import faker

from converso.conversational_engine.tools import GoogleCalendarCreator

fake = faker.Faker()


class GoogleCalendarCreatorEvaluation(GoogleCalendarCreator):

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
            "summary": fake.text(max_nb_chars=30),
            "description": fake.text(),
            "start": start,
            "end": start + timedelta(days=random.randint(1, 7), hours=random.randint(1, 3), minutes=random.randint(0, 59))
        }
