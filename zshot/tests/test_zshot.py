import spacy

DOCS = ["The Domain Name System (DNS) is the hierarchical and decentralized naming system used to identify"
        " computers, services, and other resources reachable through the Internet or other Internet Protocol"
        " (IP) networks.",
        "International Business Machines Corporation (IBM) is an American multinational technology corporation"
        " headquartered in Armonk, New York, with operations in over 171 countries."]


def test_add_pipe():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config={"entities": {}})
    assert "zshot" in nlp.pipe_names


def test_call_pipe():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("zshot", config={"entities": {}}, last=True)
    # Process a doc and see the results
    for doc in nlp.pipe(DOCS):
        print(doc._.acronyms)
    assert "zshot" in nlp.pipe_names
