import datetime
import warnings


from dagster_components.entity_data_generator.entities.entity_property_descriptors import Name, Date, Email
from dagster_components.entity_data_generator.entities.mutators import StringMutations, DateMutations


class Entity:
    first_name = Name()
    last_name = Name()
    birth_date = Date()
    email = Email()

    def __init__(self, id_number: str, first_name: str, last_name: str, birth_date: str, email: str):
        self.id_number = id_number
        self.first_name = first_name
        self.last_name = last_name
        self.birth_date = birth_date
        self.email = email

    @property
    def id_number(self):
        return self._id_number

    @id_number.setter
    def id_number(self, value: str):
        self._id_number = int(value)

    def mutate_property(
        self,
        property_name: str,
        string_mutation: StringMutations = None,
        date_mutation: DateMutations = None,
    ):
        if not string_mutation and not date_mutation:
            raise ValueError("Either mutation or date_mutation argument must be provided.  Both cannot be null.")

        if property_name not in list(self.__dir__()):
            raise ValueError(f"Property {property_name} was not found in properties {self.__dir__()}")

        if string_mutation and date_mutation:
            # if both are provided, check the property_name type and use the corresponding mutation
            if isinstance(self.__dict__[property_name], str):
                warnings.warn(
                    "Both string_mutation and date_mutation were provided, "
                    "but the provided property_name is a string, so date_mutation will be ignored."
                )
                mutation = StringMutations(string_mutation)
            elif isinstance(self.__dict__[property_name], datetime.date):
                warnings.warn(
                    "Both string_mutation and date_mutation were provided, "
                    "but the provided property_name is a datetime.date, so string_mutation will be ignored."
                )
                mutation = DateMutations(date_mutation)
            else:
                raise RuntimeError("Unidentified mutation or property.")
        elif string_mutation and not date_mutation:
            mutation = StringMutations(string_mutation)
        else:
            mutation = DateMutations(date_mutation)

        # update the property value with the desired mutation
        self.__dict__[property_name] = mutation.mutate(orig_property_value=self.__dict__[property_name])
