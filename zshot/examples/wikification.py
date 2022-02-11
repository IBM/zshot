import spacy
from spacy import displacy

from zshot import Linker, MentionsExtractor

text = "International Business Machines Corporation (IBM) is an American multinational technology corporation " \
       "headquartered in Armonk, New York, with operations in over 171 countries."

nlp = spacy.load("en_core_web_trf")
nlp.disable_pipes('ner')
nlp.add_pipe("zshot", config={"mentions_extractor": MentionsExtractor.FLAIR, "linker": Linker.BLINK}, last=True)
print(nlp.pipe_names)

doc = nlp(text)
displacy.serve(doc, style="ent")
