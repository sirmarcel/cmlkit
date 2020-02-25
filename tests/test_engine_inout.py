import numpy as np
import pathlib
import shutil
from unittest import TestCase

from cmlkit.engine import (
    normalize_extension,
    makedir,
    save_npy,
    safe_save_npy,
    read_npy,
    save_yaml,
    read_yaml,
    save_son,
    read_son,
)


class TestInout(TestCase):
    def setUp(self):
        self.tmpdir = (pathlib.Path(__file__) / "..").resolve() / "tmp_test_inout"
        self.tmpdir.mkdir(exist_ok=True)

        self.data = {"list": np.random.rand(10).tolist(), "b": "123"}

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_normalize_extension_with_ext(self):
        path = "lol/text.npy"
        self.assertEqual(normalize_extension(path, ".npy"), pathlib.Path(path))

    def test_normalize_extension_without_ext(self):
        path = "lol/text"
        self.assertEqual(normalize_extension(path, ".npy"), pathlib.Path(path + ".npy"))

    def test_mkdir(self):
        makedir(self.tmpdir / "test")

    def test_roundtrip_npy(self):
        save_npy(self.tmpdir / "npytest", self.data)
        result = read_npy(self.tmpdir / "npytest.npy")

        self.assertEqual(self.data, result)

    def test_roundtrip_yaml(self):
        save_yaml(self.tmpdir / "npytest", self.data)
        result = read_yaml(self.tmpdir / "npytest.yml")

        self.assertEqual(self.data, result)

    def test_roundtrip_safenpy(self):
        safe_save_npy(self.tmpdir / "safenpytest", self.data)
        result = read_npy(self.tmpdir / "safenpytest.npy")

        self.assertEqual(self.data, result)

    def test_son(self):
        payloads = [{"ok": {"loss": 123}}, {"error": {"error": "LolWhat"}}]
        meta = {"name": "it is a test"}

        save_son(self.tmpdir / "sontest", meta, is_metadata=True)
        save_son(self.tmpdir / "sontest", payloads[0], is_metadata=False)
        save_son(self.tmpdir / "sontest", payloads[1], is_metadata=False)
        post_meta, post_payloads = read_son(self.tmpdir / "sontest")

        self.assertEqual(post_meta, meta)
        self.assertEqual(post_payloads, payloads)
