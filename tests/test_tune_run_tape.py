from unittest import TestCase
import pathlib
import shutil
import numpy as np

from cmlkit.tune.run.tape import Tape


class TestTape(TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(__file__).parent / "tmp_test_tape"
        self.tmpdir.mkdir(exist_ok=True)

        self.metadata = {"lol": 123}
        self.payload = [{"key": np.random.random()} for i in range(20)]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_list_mode(self):
        tape = Tape.new(metadata=self.metadata)

        for p in self.payload:
            tape.append(p)

        self.assertEqual(list(tape), self.payload)

    def test_son_mode(self):
        tape = Tape.new(metadata=self.metadata, filename=self.tmpdir / "son")

        for p in self.payload:
            tape.append(p)

        self.assertEqual(list(tape), self.payload)

        tape2 = Tape.restore(filename=self.tmpdir / "son")
        self.assertEqual(list(tape2), self.payload)
