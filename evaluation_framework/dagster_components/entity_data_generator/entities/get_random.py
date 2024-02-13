import typing as t
import random
import datetime
from faker import Faker


# initialize for given localities
# see docs: https://faker.readthedocs.io/en/master/
fake = Faker(['en_US', 'it_IT'])
Faker.seed(14)


def random_name() -> t.Tuple[str, str]:
    return fake.name().split(" ")[:2]


def random_date() -> datetime.date:
    return datetime.date(
        year=random.randint(1900, 2050),
        month=random.randint(1, 12),
        day=random.randint(1, 28)  # simplify to avoid invalid dates
    )


def random_email():
    return fake.email()
