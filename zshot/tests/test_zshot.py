from typing import List

import spacy

from zshot import Zshot
from zshot.entity import Entity
from zshot.tests.config import EX_ENTITIES_DICT, EX_DOCS, EX_ENTITIES


def test_add_pipe():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config={"entities": {}})
    assert "zshot" in nlp.pipe_names


def test_disable_ner():
    nlp = spacy.load("en_core_web_trf")
    config_zshot = {
        "mentions_extractor": None
    }
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    assert "ner" not in nlp.pipe_names


def test_call_pipe_with_dict_configuration():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("zshot", config={"entities": EX_ENTITIES_DICT}, last=True)
    nlp(EX_DOCS[0])
    assert "zshot" in nlp.pipe_names
    zshot_component: Zshot = [comp for name, comp in nlp.pipeline if name == 'zshot'][0]
    assert len(zshot_component.entities) == len(EX_ENTITIES_DICT)
    assert type(zshot_component.entities[0]) == Entity


def test_call_pipe_with_registered_function_configuration():

    @spacy.registry.misc("create.entities.v1")
    def create_entities() -> List[Entity]:
        return EX_ENTITIES

    nlp = spacy.load("en_core_web_trf")

    nlp.add_pipe("zshot", config={"entities": {"@misc": "create.entities.v1"}}, last=True)
    assert "zshot" in nlp.pipe_names
    zshot_component: Zshot = [comp for name, comp in nlp.pipeline if name == 'zshot'][0]
    assert len(zshot_component.entities) == len(EX_ENTITIES_DICT)
    assert type(zshot_component.entities[0]) == Entity


def test_process_single_document():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("zshot", last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[0])
    assert doc._.mentions is not None


def test_process_pipeline_documents():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("zshot", last=True)
    assert "zshot" in nlp.pipe_names
    assert all(doc._.mentions is not None for doc in nlp.pipe(EX_DOCS))
