import spacy
from spacy import displacy

from zshot.linker import LinkerBlink
from zshot.mentions_extractor import SpacyMentionsExtractor
from zshot.utils import ents_colors

text = "International Business Machines Corporation (IBM) is an American multinational technology corporation " \
       "headquartered in Armonk, New York, with operations in over 171 countries."

nlp = spacy.load("en_core_web_trf")
nlp.disable_pipes('ner')
nlp.add_pipe("zshot", config={"mentions_extractor": SpacyMentionsExtractor.id(), "linker": LinkerBlink.id()}, last=True)
print(nlp.pipe_names)

doc = nlp(text)
displacy.serve(doc, style="ent", options={'colors': ents_colors(doc)})
