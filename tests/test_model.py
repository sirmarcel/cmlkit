from unittest import TestCase

from cmlkit.model import Model
from cmlkit.representation import MBTR1


class TestModel(TestCase):
    def test_smoke(self):
        mbtr = MBTR1(
            start=0,
            stop=4,
            num=5,
            geomf="count",
            weightf="unity",
            broadening=0.001,
            eindexf="noreversals",
            aindexf="noreversals",
            elems=[0, 1, 2, 3],
            flatten=True,
        )
        model = Model(
            representation=[mbtr, mbtr.get_config()],
            regression={
                "krr": {
                    "kernel": {"kernel_global": {"kernelf": {"gaussian": {"ls": 1.0}}}},
                    "nl": 1.0e-7,
                }
            },
            per="atom",
            context={"flag": "hey"}
        )

        self.assertEqual(model.per, "atom")

        self.assertEqual(model.regression.context["flag"], "hey")
        self.assertEqual(model.regression.kernel.context["flag"], "hey")

        self.assertEqual(model.representation.context["flag"], "hey")
        # the first rep doesn't have the flag! it was already instantiated early.
        self.assertEqual(model.representation.reps[1].context["flag"], "hey")
