import json
import os
import tempfile
from typing import List

import spacy

from zshot import Zshot, PipelineConfig
from zshot.utils.data_models import Entity
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.tests.linker.test_linker import DummyLinker, DummyLinkerEnd2End
from zshot.tests.mentions_extractor.test_mention_extractor import DummyMentionsExtractor, DummyMentionsExtractorWithNER


def test_add_pipe():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot")
    assert "zshot" in nlp.pipe_names


def test_disable_ner():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("zshot", last=True)
    assert "zshot" in nlp.pipe_names
    assert "ner" not in nlp.pipe_names


def test_wrong_pipeline():
    nlp = spacy.blank("en")
    assert "ner" not in nlp.pipe_names
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractorWithNER())
    try:
        nlp.add_pipe("zshot", config=config_zshot, last=True)
    except ValueError:
        assert True


def test_disable_mentions_extractor():
    nlp = spacy.load("en_core_web_sm")
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractorWithNER(), linker=DummyLinkerEnd2End())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names and "ner" not in nlp.pipe_names
    assert not nlp.get_pipe("zshot").mentions_extractor


def test_serialization_zshot():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(mentions_extractor=DummyMentionsExtractor(), linker=DummyLinker())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    assert "ner" not in nlp.pipe_names
    pipes = [p for p in nlp.pipe_names if p != "zshot"]

    d = tempfile.TemporaryDirectory()
    nlp.to_disk(d.name, exclude=pipes)
    config_fn = os.path.join(d.name, "zshot", "config.cfg")
    assert os.path.exists(config_fn)
    with open(config_fn, "r") as f:
        config = json.load(f)
    assert "disable_default_ner" in config and config["disable_default_ner"]
    nlp2 = spacy.load(d.name)
    assert "zshot" in nlp2.pipe_names


def test_call_pipe_with_registered_function_configuration():

    @spacy.registry.misc("dummy.mentions-extractor")
    def create_custom_spacy_extractor():
        return DummyMentionsExtractor()

    @spacy.registry.misc("dummy.linker")
    def create_custom_linker():
        return DummyLinker()

    @spacy.registry.misc("get.entities.v1")
    def get_entities() -> List[Entity]:
        return EX_ENTITIES

    @spacy.registry.misc("get.mentions.v1")
    def get_mentions() -> List[Entity]:
        return EX_ENTITIES

    config_zshot = {
        "mentions_extractor": {"@misc": "dummy.mentions-extractor"},
        "linker": {"@misc": "dummy.linker"},
        "entities": {"@misc": "get.entities.v1"},
        "mentions": {"@misc": "get.mentions.v1"}
    }

    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    zshot_component: Zshot = [comp for name, comp in nlp.pipeline if name == 'zshot'][0]
    assert len(zshot_component.entities) == len(EX_ENTITIES)
    assert len(zshot_component.mentions) == len(EX_ENTITIES)
    assert type(zshot_component.entities[0]) == Entity
    assert type(zshot_component.mentions[0]) == Entity


def test_call_pipe_with_pipeline_configuration():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("zshot", config=PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=DummyLinker(),
        entities=EX_ENTITIES,
        mentions=EX_ENTITIES), last=True)
    assert "zshot" in nlp.pipe_names
    zshot_component: Zshot = [comp for name, comp in nlp.pipeline if name == 'zshot'][0]
    assert len(zshot_component.entities) == len(EX_ENTITIES)
    assert type(zshot_component.entities[0]) == Entity
    assert len(zshot_component.mentions) == len(EX_ENTITIES)
    assert type(zshot_component.mentions[0]) == Entity
    assert len(zshot_component.mentions) == len(EX_ENTITIES)
    assert type(zshot_component.mentions[0]) == Entity


def test_process_single_document():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[0])
    assert doc._.mentions is not None


def test_process_pipeline_documents():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", last=True)
    assert "zshot" in nlp.pipe_names
    assert all(doc._.mentions is not None for doc in nlp.pipe(EX_DOCS))
