import datetime


class Name:
    """
    A descriptor class for name properties of Entity instances,
    to simplify getters and setters for these properties.
    """
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        instance.__dict__[self._name] = value.title()


class Date:
    """
    A descriptor class for name properties of Entity instances,
    to simplify getters and setters for these properties.
    """
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        instance.__dict__[self._name] = datetime.date.fromisoformat(value)


class Email:
    """
    A descriptor class for email property of Entity instances,
    to simplify getters and setters for these properties.
    """
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        instance.__dict__[self._name] = value.lower()
