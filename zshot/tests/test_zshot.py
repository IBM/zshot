import spacy


def test_add_pipe():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config={"entities": {}})
    assert "zshot" in nlp.pipe_names


def test_call_pipe():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("zshot", config={"entities": {}})
    # Process a doc and see the results
    doc = nlp("LOL, be right back")
    print(doc._.acronyms)
    assert "zshot" in nlp.pipe_names