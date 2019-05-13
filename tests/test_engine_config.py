from unittest import TestCase

from cmlkit.engine import Component, _from_config


class MyClass(Component):
    kind = "hello"

    default_context = {"b": 1}

    def __init__(self, a, context={}):
        super().__init__(context=context)
        self.a = a

    def _get_config(self):
        return {"a": self.a}


registry = {"hello": MyClass}


class TestDeserialisation(TestCase):
    def test_it_works(self):
        config = {"hello": {"a": 2}}

        my = _from_config(config, classes=registry)

        self.assertEqual(my.a, 2)

    def test_raises_valueerror_if_unknown(self):
        config = {"darkness": {"a": 2}}

        with self.assertRaises(ValueError):
            my = _from_config(config, classes=registry)
