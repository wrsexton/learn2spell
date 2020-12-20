import typing as T
import pytest as PT
import requests_mock as RM
import spells as S

from . import fixture as F

class TestLearn2Spell():

    def test_sanity(self):
        assert True

    def test_spells(self,
                    requests_mock: RM.Mocker,
                    spells: S.Spells):
        requests_mock.register_uri(RM.GET,
                                   S.SPELLS_INDEX_URL,
                                   status_code=200,
                                   json=F.FAKE_SPELLS_INDEX)
        result = spells.getAllSpellIndexes()
        assert result == [s["index"] for s in F.FAKE_SPELLS_INDEX["results"]]
