import spacy

from zshot import Entity

DOCS = ["The Domain Name System (DNS) is the hierarchical and decentralized naming system used to identify"
        " computers, services, and other resources reachable through the Internet or other Internet Protocol"
        " (IP) networks.",
        "International Business Machines Corporation (IBM) is an American multinational technology corporation"
        " headquartered in Armonk, New York, with operations in over 171 countries."]


def test_add_pipe():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config={"entities": {}})
    assert "zshot" in nlp.pipe_names


def test_call_pipe_with_dict():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("zshot", config={"entities": {"apple": "The apple fruit",
                                               "DNS": "domain name system",
                                               "IBM": "technology corporation",
                                               "NYC": "New York city",
                                               "Florida": "southeasternmost U.S. state",
                                               "Paris": "Paris is located in northern central France, "
                                                        "in a north-bending arc of the river Seine"}}, last=True)
    # Process a doc and see the results
    nlp(DOCS[0])
    for doc in nlp.pipe(DOCS):
        print(doc._.acronyms)
    assert "zshot" in nlp.pipe_names


def test_call_pipe_with_entities():
    nlp = spacy.load("en_core_web_trf")
    entities = [
        Entity(name="apple", description="the apple fruit"),
        Entity(name="DNS", description="domain name system", vocabulary=["DNS", "Domain Name System"]),
        Entity(name="IBM", description="technology corporation", vocabulary=["IBM", "International Business machine"]),
        Entity(name="NYC", description="New York city"),
        Entity(name="Florida", description="southeasternmost U.S. state"),
        Entity(name="Paris", description="Paris is located in northern central France, "
                                         "in a north-bending arc of the river Seine"),

    ]
    nlp.add_pipe("zshot", config={"entities": entities}, last=True)
    # Process a doc and see the results
    nlp(DOCS[0])
    for doc in nlp.pipe(DOCS):
        print(doc._.acronyms)
    assert "zshot" in nlp.pipe_names
