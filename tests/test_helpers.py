from unittest import TestCase
from helpers import convert_sequence


class TestSequenceHandling(TestCase):

    def test_handles_single_values_in_sequence(self):
        a = ('gaussian')
        b = convert_sequence(a)

        self.assertEqual(b, 'gaussian')

    def test_handles_single_values_alone(self):
        a = 'gaussian'
        b = convert_sequence(a)

        self.assertEqual(b, 'gaussian')

    def test_handles_associated_values(self):
        a = ('gaussian', 1, 2, 3)
        b = convert_sequence(a)

        self.assertEqual(b, ('gaussian', (1, 2, 3)))
