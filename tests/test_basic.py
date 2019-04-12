# Does anything work at all?

from unittest import TestCase
import pathlib
import shutil


class TestImports(TestCase):
    def test_import_cmlkit(self):
        import cmlkit


class TestSerialization(TestCase):
    def setUp(self):
        self.tmpdir = (pathlib.Path(__file__) / "..").resolve() / "tmp_test_basic_serial"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_serialization(self):
        import cmlkit

        krr = cmlkit.from_config(
            {"kind": "krr", "config": {"nl": 1.0, "kernel": ["gaussian", [1.0]]}}
        )

        krr.get_config()

        cmlkit.save_yaml(self.tmpdir / 'krr.yml', krr.get_config())
        krr2 = cmlkit.from_yaml(self.tmpdir / 'krr.yml')

        self.assertEqual(krr2.get_config(), krr.get_config())
