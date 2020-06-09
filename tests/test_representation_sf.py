import numpy as np
from unittest import TestCase
import unittest.mock
import shutil
import pathlib


def fc(r, cutoff):
    return 0.5 * (np.cos(np.pi * r / cutoff) + 1)


def rad_sf(r, eta, mu):
    return np.exp(-eta * (r - mu) ** 2)


def ang_sf(ang, r1, r2, r3, lambd, zeta, eta):
    return (
        2.0 ** (1 - zeta)
        * (1 + (lambd * np.cos(ang))) ** zeta
        * np.exp(-eta * ((r1 ** 2) + (r2 ** 2) + (r3 ** 2)))
    )


class TestSymmetryFunctions(TestCase):
    def setUp(self):
        self.tmpdir = (pathlib.Path(__file__) / "..").resolve() / "tmp_test_sf"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_infile(self):
        from cmlkit.representation.sf import SymmetryFunctions
        sf = SymmetryFunctions(
                    [1], universal=[{"rad": {"eta": 1.0, "mu": 1.0, "cutoff": 3.0}}]
                )

        sf.get_infile()

    def test_rad_sf(self):

        with unittest.mock.patch.dict("os.environ", {"CML_SCRATCH": str(self.tmpdir)}):
            from cmlkit.representation.sf import SymmetryFunctions
            from cmlkit import Dataset, runner_path

            print(runner_path)

            data = Dataset(
                z=np.array([[1, 1]]), r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
            )

            for i in range(5):
                eta = np.random.random()
                mu = np.random.random()
                cutoff = 5.0
                sf = SymmetryFunctions(
                    [1], universal=[{"rad": {"eta": eta, "mu": mu, "cutoff": cutoff}}]
                )

                computed = sf(data).ragged

                np.testing.assert_almost_equal(
                    computed[0][0][0], rad_sf(2.0, eta, mu) * fc(2.0, cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][1][0], rad_sf(2.0, eta, mu) * fc(2.0, cutoff)
                )

    def test_parametrized_sf(self):

        with unittest.mock.patch.dict("os.environ", {"CML_SCRATCH": str(self.tmpdir)}):
            from cmlkit.representation.sf import SymmetryFunctions
            from cmlkit import Dataset, runner_path

            print(runner_path)

            data = Dataset(
                z=np.array([[1, 1]]), r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
            )

            cutoff = 5.0
            sf = SymmetryFunctions(
                [1], universal=[{"rad_centered": {"n": 3, "cutoff": cutoff}}]
            )

            delta = (cutoff - 1.5) / 2

            computed = sf(data).ragged

            # note that the ordering of the results is not the same as
            # the one you'd expect -- runner internally reorders.
            # I can't be bothered to find out how for now.
            np.testing.assert_almost_equal(
                computed[0][0][2],
                rad_sf(2.0, 0.5 / (0.5 + 0 * delta) ** 2, 0.0) * fc(2.0, cutoff),
            )

            np.testing.assert_almost_equal(
                computed[0][0][1],
                rad_sf(2.0, 0.5 / (0.5 + 1 * delta) ** 2, 0.0) * fc(2.0, cutoff),
            )

            np.testing.assert_almost_equal(
                computed[0][0][0],
                rad_sf(2.0, 0.5 / (0.5 + 2 * delta) ** 2, 0.0) * fc(2.0, cutoff),
            )

    def test_ang_sf(self):

        with unittest.mock.patch.dict("os.environ", {"CML_SCRATCH": str(self.tmpdir)}):
            from cmlkit.representation.sf import SymmetryFunctions
            from cmlkit import Dataset, runner_path

            print(runner_path)

            data = Dataset(
                z=np.array([[1, 1, 1]]),
                r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            )

            for i in range(5):
                eta = np.random.random()
                zeta = np.random.random()
                cutoff = 5.0
                lambd = 1.0 + np.random.random() * 0.3

                sf = SymmetryFunctions(
                    [1],
                    universal=[
                        {
                            "ang": {
                                "eta": eta,
                                "zeta": zeta,
                                "cutoff": cutoff,
                                "lambd": lambd,
                            }
                        }
                    ],
                )

                computed = sf(data).ragged

                print(computed)

                np.testing.assert_almost_equal(
                    computed[0][0][0],
                    ang_sf(np.pi / 2, 1.0, 1.0, np.sqrt(2.0), lambd, zeta, eta)
                    * fc(1.0, cutoff)
                    * fc(1.0, cutoff)
                    * fc(np.sqrt(2.0), cutoff),
                )

                np.testing.assert_almost_equal(
                    computed[0][1][0],
                    ang_sf(np.pi / 4, 1.0, 1.0, np.sqrt(2.0), lambd, zeta, eta)
                    * fc(1.0, cutoff)
                    * fc(1.0, cutoff)
                    * fc(np.sqrt(2.0), cutoff),
                )

    def test_rad_sf_multiple_elements(self):

        with unittest.mock.patch.dict("os.environ", {"CML_SCRATCH": str(self.tmpdir)}):
            from cmlkit.representation.sf import SymmetryFunctions
            from cmlkit import Dataset, runner_path

            print(runner_path)

            data = Dataset(
                z=np.array([[1, 2, 3]]),
                r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            )

            for i in range(5):
                eta = np.random.random()
                mu = np.random.random()
                cutoff = 5.0
                sf = SymmetryFunctions(
                    [1, 2, 3], universal=[{"rad": {"eta": eta, "mu": mu, "cutoff": cutoff}}]
                )

                computed = sf(data).ragged

                print(computed)

                np.testing.assert_almost_equal(
                    computed[0][0][0], 0.0
                )

                np.testing.assert_almost_equal(
                    computed[0][0][1], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][0][2], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][1][3 + 0], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][1][3 + 1], 0.0
                )

                np.testing.assert_almost_equal(
                    computed[0][1][3 + 2], rad_sf(np.sqrt(2), eta, mu) * fc(np.sqrt(2), cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][2][6 + 0], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][2][6 + 1], rad_sf(np.sqrt(2), eta, mu) * fc(np.sqrt(2), cutoff)
                )

                np.testing.assert_almost_equal(
                    computed[0][2][6 + 2], 0.0
                )
