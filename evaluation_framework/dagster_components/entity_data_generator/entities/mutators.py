import datetime
from enum import Enum
import warnings
import random
import string

from dagster_components.entity_data_generator.entities.get_random import random_date


class StringMutations(Enum):
    DROP_CHAR = 1
    SWAP_CHAR = 2
    REPEAT_CHAR = 3
    RANDOM_REPLACE = 4
    COMPLETELY_NEW = 5
    DELETE = 6

    def mutate(self, orig_property_value: str) -> str:
        """Applies the desired mutation to orig_property_value."""

        # choose a random character in the string
        char_idx = random.choice([*range(len(orig_property_value))])

        if self.name == "DROP_CHAR":
            # drop the randomly chosen character
            new_val = orig_property_value[:char_idx] + orig_property_value[1 + char_idx:]

        elif self.name == "SWAP_CHAR":
            # swap the randomly chosen character with a character before or after it
            if char_idx == 0:
                new_val = orig_property_value[1] + orig_property_value[0] + orig_property_value[2:]
            elif char_idx == len(orig_property_value) - 1:
                new_val = orig_property_value[:2] + orig_property_value[-1] + orig_property_value[-2]
            else:
                new_val = (
                        orig_property_value[:char_idx] + orig_property_value[char_idx + 1]
                        + orig_property_value[char_idx] + orig_property_value[char_idx + 2:]
                )

        elif self.name == "REPEAT_CHAR":
            # repeat the randomly chosen character
            new_val = orig_property_value[:char_idx] + orig_property_value[char_idx] + orig_property_value[char_idx:]

        elif self.name == "RANDOM_REPLACE":
            # replace the randomly chosen character with a random character
            new_char = random.choice(string.ascii_letters)
            new_val = orig_property_value[:char_idx] + new_char + orig_property_value[1 + char_idx:]

        elif self.name == "COMPLETELY_NEW":
            # return an entirely different string
            new_val = ''.join(random.choices(string.ascii_letters, k=len(orig_property_value)))

        else:
            # delete the information
            new_val = None

        return new_val


class DateMutations(Enum):
    SWAP_DAY_MONTH = 1
    RANDOM_REPLACE = 2
    COMPLETELY_NEW = 3
    DELETE = 4

    def mutate(self, orig_property_value: datetime.date) -> str:
        """Applies the desired mutation to orig_property_value."""

        if self.name == "SWAP_DAY_MONTH":
            # swap the day with the month, if possible
            # this mimics situations where there are 2-digit entries for day and month
            if orig_property_value.day > 12:
                warnings.warn(
                    "The day of the month > 12, so cannot swap day and month.  "
                    "The original value will be returned."
                )
                new_val = orig_property_value
            else:
                new_val = datetime.date(
                    year=orig_property_value.year,
                    month=orig_property_value.day,
                    day=orig_property_value.month
                )

        elif self.name == "RANDOM_REPLACE":
            # replace a digit in the date - without 0 to avoid problems with invalid dates
            what_to_replace = random.choice(['d', 'm', 'y'])
            if what_to_replace == 'd':
                new_int = random.choice([*range(1, 29)])  # simplify to avoid invalid dates
                new_val = datetime.date(
                    year=orig_property_value.year,
                    month=orig_property_value.month,
                    day=new_int
                )
            elif what_to_replace == 'm':
                new_int = random.choice([*range(1, 13)])
                new_val = datetime.date(
                    year=orig_property_value.year,
                    month=new_int,
                    day=orig_property_value.day
                )
            else:
                new_int = random.choice([*range(1900, 2050)])
                new_val = datetime.date(
                    year=new_int,
                    month=orig_property_value.month,
                    day=orig_property_value.day
                )

        elif self.name == "COMPLETELY_NEW":
            # return an entirely different date
            new_val = random_date()

        else:
            # delete the information
            new_val = None

        return new_val
