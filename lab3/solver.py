# Viktoriia Nowotka
from abc import ABC, abstractmethod


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, X, *args, **kwargs):
        """
        A method that predict class to the given problem.
        It may accept or require additional parameters.
        Returns the label and may return additional info.
        """
        ...
