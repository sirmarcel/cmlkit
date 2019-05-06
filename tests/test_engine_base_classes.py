from unittest import TestCase

from cmlkit.engine import BaseComponent, Configurable


class MyClass(BaseComponent):
    kind = "hello"

    default_context = {"b": 1}

    def __init__(self, a, context={}):
        super().__init__(context=context)
        self.a = a

    def _get_config(self):
        return {"a": self.a}


class TestConfigurable(TestCase):
    def test_from_config(self):
        hello = MyClass.from_config({"a": 1})
        self.assertEqual(hello.a, 1)

    def test_to_config(self):
        hello = MyClass(a=1)
        self.assertEqual(hello.get_config(), {"hello": {"a": 1}})


class TestContext(TestCase):
    def test_default(self):
        hello = MyClass(a=1)
        self.assertEqual(hello.context["b"], 1)

    def test_not_default(self):
        hello = MyClass(a=1, context={"b": 2})
        self.assertEqual(hello.context["b"], 2)
