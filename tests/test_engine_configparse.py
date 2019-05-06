from unittest import TestCase

from cmlkit.engine import is_config, parse_config


class TestConfigparseBasics(TestCase):
    def test_is_config_valid(self):
        valid = {"a": {"b": 3}}

        self.assertTrue(is_config(valid))

    def test_is_config_invalid(self):
        invalid = {3: {"b": 3}}
        self.assertFalse(is_config(invalid))

        invalid = {3: {"b": 3}, "b": 3}
        self.assertFalse(is_config(invalid))

        invalid = [3, {"b": 3}]
        self.assertFalse(is_config(invalid))

        invalid = {"b": 3}
        self.assertFalse(is_config(invalid))

    def test_parse_valid_config(self):
        valid = {"a": {"b": 3}}
        kind, config = parse_config(valid)

        self.assertEqual(kind, "a")
        self.assertEqual(config, {"b": 3})

    def test_parse_invalid_config(self):
        with self.assertRaises(ValueError):
            invalid = {3: {"b": 3}, "b": 3}
            parse_config(invalid)
