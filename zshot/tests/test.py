import spacy
from transformers import AutoTokenizer

from zshot import PipelineConfig
from zshot.linker.linker_regen.linker_regen import LinkerRegen
from zshot.linker.linker_regen.utils import load_wikipedia_trie, spans_to_wikipedia
from zshot.mentions_extractor import MentionsExtractorSpacy
from zshot.tests.config import EX_DOCS

tokenizer = AutoTokenizer.from_pretrained("ibm/regen-disambiguation", model_max_length=1024)

input_ids = tokenizer("test1", return_tensors="pt")['input_ids'][0].tolist()
print(input_ids)

input_ids = tokenizer("test2", return_tensors="pt")['input_ids'][0].tolist()
print(input_ids)


nlp = spacy.load("en_core_web_sm")
trie = load_wikipedia_trie()
config = PipelineConfig(
    mentions_extractor=MentionsExtractorSpacy(),
    linker=LinkerRegen(trie=trie),
)
nlp.add_pipe("zshot", config=config, last=True)
assert "zshot" in nlp.pipe_names

doc = nlp(EX_DOCS[1])
print(doc.ents)
print(spans_to_wikipedia(doc._.spans))