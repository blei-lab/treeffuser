"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.

tThe idea is to "hide" the particular tree we want to use so that
we can easily switch between different models without having to change
the rest of the code.
"""

import abc


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(self, data):
        pass
