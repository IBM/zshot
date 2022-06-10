import pkgutil

import spacy

from zshot import MentionsExtractor
from zshot.tests.config import EX_DOCS


def test_flair_mentions_extractor():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": MentionsExtractor.FLAIR
    }
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[1])
    assert doc.ents == ()
    assert len(doc._.mentions) > 0


def test_flair_mentions_extractor_pipeline():
    if not pkgutil.find_loader("flair"):
        return
    nlp = spacy.load("en_core_web_sm")
    config_zshot = {
        "mentions_extractor": MentionsExtractor.FLAIR
    }
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    docs = [doc for doc in nlp.pipe(EX_DOCS)]
    assert all(doc.ents == () for doc in docs)
    assert all(len(doc._.mentions) > 0 for doc in docs)
