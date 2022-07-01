import pkgutil
from typing import Iterator

from spacy.tokens.doc import Doc

from zshot.linker.linker import Linker
from zshot.entity import Entity


class TARSLinker(Linker):

    def __init__(self):
        super().__init__()
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair==0.11")

        self.is_end2end = True
        self.model = None
        self.flat_entities()

    def flat_entities(self):
        if isinstance(self._entities, dict):
            self._entities = list(self._entities.keys())
        if isinstance(self._entities, list):
            self._entities = [e.name if type(e) == Entity else e for e in self._entities]
        if self._entities is None:
            self._entities = []

    def load_models(self):
        from flair.models import TARSTagger
        if self.model is None:
            self.model = TARSTagger.load('tars-ner')

    def link(self, docs: Iterator[Doc], batch_size=None):
        from flair.data import Sentence
        self.load_models()
        self.flat_entities()
        self.model.add_and_switch_to_new_task(f'zshot.ner.{hash(tuple(self._entities))}',
                                              self._entities, label_type='ner')

        for doc in docs:
            sent = Sentence(str(doc), use_tokenizer=True)
            self.model.predict(sent)
            sent_mentions = sent.get_spans('ner')
            for mention in sent_mentions:
                doc.ents += (doc.char_span(mention.start_position, mention.end_position, mention.tag),)
