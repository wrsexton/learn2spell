import typing as T
import pytest as PT
import requests_mock as RM
import spells as S

from . import fixture as F

class TestLearn2Spell():

    def test_sanity(self):
        assert True

    def test_getAllSpellIndexess(self,
                                 requests_mock: RM.Mocker,
                                 spells: S.Spells):
        requests_mock.register_uri(RM.GET,
                                   S.SPELLS_INDEX_URL,
                                   status_code=200,
                                   json=F.FAKE_SPELLS_INDEX)
        result = spells.getAllSpellIndexes()
        assert result == [s["index"] for s in F.FAKE_SPELLS_INDEX["results"]]

    def test_getSpell(self,
                      requests_mock: RM.Mocker,
                      spells: S.Spells):
        index = "fake-spell-1"
        requests_mock.register_uri(RM.GET,
                                   spells.getSpellURL(index),
                                   status_code=200,
                                   json=F.FAKE_SPELL)
        result = spells.getSpell(index)
        assert result == F.FAKE_SPELL
