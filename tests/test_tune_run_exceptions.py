from unittest import TestCase

from concurrent.futures import TimeoutError

from cmlkit.tune.run.exceptions import get_exceptions, get_exceptions_spec, exceptions


class TestRunExceptions(TestCase):
    def test_deserialisation(self):
        result = get_exceptions(["TimeoutError", "ValueError"])

        self.assertEqual(result, (TimeoutError, ValueError))

        with self.assertRaises(ValueError):
            get_exceptions([OSError])

    def test_roundtrip(self):
        self.assertEqual(
            get_exceptions(get_exceptions_spec(list(exceptions.values()))),
            tuple(exceptions.values()),
        )
