import typing as T
import pytest as PT
from spells import Spells

@PT.fixture
def spells() -> Spells:
    """Initialize and return a spells object"""
    return Spells()
