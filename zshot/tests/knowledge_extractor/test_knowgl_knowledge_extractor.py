import gc
import logging
import shutil
from pathlib import Path

import pytest
import spacy
from transformers import AutoTokenizer

from zshot import PipelineConfig
from zshot.knowledge_extractor import KnowGL
from zshot.knowledge_extractor.knowgl.utils import ranges, find_sub_list, get_words_mappings, get_spans, get_triples
from zshot.tests.config import TEXTS
from zshot.utils.data_models import Span, Relation
from zshot.utils.data_models.relation_span import RelationSpan

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting regen tests")
    yield True
    logger.warning("Removing cache")
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)
    gc.collect()


def test_knowgl_knowledge_extractor():
    nlp = spacy.blank("en")
    config = PipelineConfig(
        knowledge_extractor=KnowGL()
    )
    nlp.add_pipe("zshot", config=config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(TEXTS[0])
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert len(doc._.relations) > 0
    doc = nlp("")
    assert len(doc.ents) == 0
    assert len(doc._.spans) == 0
    assert len(doc._.relations) == 0
    docs = [doc for doc in nlp.pipe(TEXTS)]
    assert all(len(doc.ents) > 0 for doc in docs)
    assert all(len(doc._.spans) > 0 for doc in docs)
    assert all(len(doc._.relations) > 0 for doc in docs)
    nlp.remove_pipe('zshot')
    del doc, nlp, config


def test_ranges():
    numbers = [0, 1, 2, 3, 7]
    assert list(ranges(numbers)) == [[0, 1, 2, 3], [7]]


def test_find_sub_list():
    numbers = [0, 1, 2, 3, 7]
    sl = [1, 2, 3]
    results = find_sub_list(sl, numbers)
    assert type(results) is list
    init, end = results[0]
    assert init == 1 and end == 3


def test_get_spans():
    tokenizer = AutoTokenizer.from_pretrained("ibm/knowgl-large")
    input_data = tokenizer(TEXTS,
                           truncation=True,
                           padding=True,
                           return_tensors="pt")
    words_mapping, char_mapping = get_words_mappings(input_data.encodings[0], TEXTS[0])
    assert words_mapping and char_mapping
    spans = get_spans("LICIACube", "CubeSat", tokenizer, input_data.encodings[0],
                      words_mapping, char_mapping)
    assert spans == [Span(78, 87, 'CubeSat')]

    words_mapping, char_mapping = get_words_mappings(input_data.encodings[1], TEXTS[1])
    assert words_mapping and char_mapping
    spans = get_spans("CH2O2", "CH2O2", tokenizer, input_data.encodings[1],
                      words_mapping, char_mapping)
    assert spans == [Span(0, 5, 'CH2O2')]


def test_get_triples():
    s1 = Span(78, 87, 'CubeSat')
    o1 = Span(41, 48, 'small satellite')
    o2 = Span(30, 38, 'Satellite')
    objects = [o1, o2]
    rel = "instance of"
    triples = get_triples([s1], rel, objects)
    assert len(triples) == 2
    for obj, triple in zip(objects, triples):
        assert triple[0] == s1 and triple[2] == obj
        assert triple[1] == RelationSpan(start=s1, end=obj, relation=Relation(name=rel, description=""))
