import gc
import logging
import shutil
from pathlib import Path

import pytest
import spacy

from zshot import PipelineConfig
from zshot.linker.linker_regen.linker_regen import LinkerRegen
from zshot.linker.linker_regen.trie import Trie
from zshot.linker.linker_regen.utils import load_wikipedia_trie, spans_to_wikipedia, create_input
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.tests.mentions_extractor.test_mention_extractor import DummyMentionsExtractor
from zshot.utils.data_models import Span

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def teardown():
    logger.warning("Starting regen tests")
    yield True
    logger.warning("Removing cache")
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)
    gc.collect()


def test_regen_linker():
    nlp = spacy.blank("en")
    config = PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=LinkerRegen(),
        entities=EX_ENTITIES
    )
    nlp.add_pipe("zshot", config=config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(len(doc.ents) > 0 for doc in docs)
    del nlp.get_pipe('zshot').mentions_extractor, nlp.get_pipe('zshot').entities, nlp.get_pipe('zshot').nlp
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.trie, \
        nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, config


def test_regen_linker_wikification():
    nlp = spacy.blank("en")
    trie = Trie()
    trie.add([794, 536, 1])
    trie.add([794, 357, 1])
    config = PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=LinkerRegen(trie=trie),
    )
    nlp.add_pipe("zshot", config=config, last=True)
    assert "zshot" in nlp.pipe_names

    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0
    del nlp.get_pipe('zshot').mentions_extractor, nlp.get_pipe('zshot').entities, nlp.get_pipe('zshot').nlp
    del nlp.get_pipe('zshot').linker.tokenizer, nlp.get_pipe('zshot').linker.trie, \
        nlp.get_pipe('zshot').linker.model, nlp.get_pipe('zshot').linker
    nlp.remove_pipe('zshot')
    del doc, nlp, config


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_load_wikipedia_trie():
    trie = load_wikipedia_trie()
    assert len(list(trie.trie_dict.keys())) == 6952


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_span_to_wiki():
    s = Span(label="Surfing", start=0, end=10)
    wiki_links = spans_to_wikipedia([s])
    assert len(wiki_links) > 0
    assert wiki_links[0].startswith("https://en.wikipedia.org/wiki?curid=")


def test_create_input():
    start_delimiter = "[START]"
    end_delimiter = "[END]"
    max_length = 10

    times_rep = 6
    sentence = "[START]" + " test" * times_rep + " [END]"
    input_sentence = create_input(sentence, max_length, start_delimiter, end_delimiter)
    assert input_sentence == sentence
    times_rep = 12
    sentence = "[START]" + " test" * times_rep + " [END]"
    input_sentence = create_input(sentence, max_length, start_delimiter, end_delimiter)
    assert input_sentence == " ".join(["test" for i in range(9)])
