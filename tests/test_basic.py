# Does anything work at all?

from unittest import TestCase


class TestImports(TestCase):
    def test_import_cmlkit(self):
        import cmlkit

    def test_instantiation(TestCase):
        import cmlkit

        cmlkit.from_config(
            {"kind": "krr", "config": {"nl": 1.0, "kernel": ["gaussian", 1.0]}}
        )
