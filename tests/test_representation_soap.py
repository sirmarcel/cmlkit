import numpy as np
from unittest import TestCase
import unittest.mock
import shutil
import pathlib


class TestSOAP(TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(__file__).parent / "tmp_test_soap"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_smoke(self):
        # since quippy is impossible to install,
        # we can't test it here.

        # but we can at least check that things are not TOO broken

        with unittest.mock.patch.dict(
            "os.environ",
            {
                "CML_QUIPPY_PYTHONPATH": str(self.tmpdir),
                "CML_QUIPPY_PYTHON_EXE": "",
                "CML_SCRATCH": str(self.tmpdir),
            },
        ):
            res = np.array([np.array([1.0]), np.array([2.0])], dtype=object)

            def fake_output(*args, **kwargs):
                outfolder = list(self.tmpdir.glob("soap_*"))[0]
                np.save(outfolder / "out", res)

                return "", ""

            fake_task = unittest.mock.MagicMock(side_effect=fake_output)

            with unittest.mock.patch(
                "cmlkit.representation.soap.quippy_interface.run_task", fake_task
            ):

                from cmlkit.representation.soap import SOAP
                from cmlkit import Dataset

                data = Dataset(
                    z=np.array([[1, 1]]),
                    r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
                )
                soap = SOAP(elems=[1], sigma=0.1, n_max=2, l_max=3, cutoff=5.0)

                computed = soap(data)

                np.testing.assert_array_equal(computed, res)
