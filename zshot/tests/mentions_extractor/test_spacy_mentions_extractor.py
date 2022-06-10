import spacy

from zshot import Entity, MentionsExtractor, Linker

DOCS = ["The Domain Name System (DNS) is the hierarchical and decentralized naming system used to identify"
        " computers, services, and other resources reachable through the Internet or other Internet Protocol"
        " (IP) networks.",
        "International Business Machines Corporation (IBM) is an American multinational technology corporation"
        " headquartered in Armonk, New York, with operations in over 171 countries."]


def test_spacy_mentions_extractor():
    nlp = spacy.load("en_core_web_trf")
    config_zshot = {
        "mentions_extractor": MentionsExtractor.SPACY
    }
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(DOCS[1])
    assert doc.ents == ()
    assert doc._.mentions != []
