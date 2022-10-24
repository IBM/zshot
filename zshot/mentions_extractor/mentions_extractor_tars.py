import pkgutil
from typing import Iterator, Optional, Union, List

from spacy.tokens.doc import Doc

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.utils.models.tars.utils import tars_predict
from zshot.utils.data_models import Entity, Span


class MentionsExtractorTARS(MentionsExtractor):
    """ TARS end2end Linker """
    def __init__(self, default_entities: Optional[str] = "conll-short"):
        """
        :param default_entities: Default entities to use in case no custom ones are set
        One of:
            - 'conll-short'
            - 'ontonotes-long'
            - 'ontonotes-short'
            - 'wnut_17-long'
            - 'wnut_17-short'
        """
        super().__init__()
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair==0.11")

        self.is_end2end = True
        self.default_entities = default_entities
        self.model = None
        self.task = None

    def set_kg(self, mentions: Iterator[Entity]):
        """ Set new entities in the model

        :param mentions: New entities to use
        """
        old_entities = self._mentions
        super().set_kg(mentions)
        if old_entities != mentions:
            self.flat_entities()
            self.task = f'zshot.ner.{hash(tuple(self._mentions))}'
            if not self.model:
                self.load_models()
            self.model.add_and_switch_to_new_task(self.task,
                                                  self._mentions, label_type='ner')

    def flat_entities(self):
        """ As TARS use only the labels, take just the name of the entities and not the description """
        if isinstance(self._mentions, dict):
            self._mentions = list(self._mentions.keys())
        if isinstance(self._mentions, list):
            self._mentions = [e.name if type(e) == Entity else e for e in self._mentions]
        if self._mentions is None:
            self._mentions = []

    def load_models(self):
        """ Load TARS model if its not initialized"""
        if not self.model:
            from flair.models import TARSTagger
            self.model = TARSTagger.load('tars-ner')

            if not self.mentions:
                self.model.switch_to_task(self.default_entities)
                self.task = self.default_entities
            else:
                self.flat_entities()
                self.task = f'zshot.ner.{hash(tuple(self._mentions))}'
                self.model.add_and_switch_to_new_task(self.task,
                                                      self._mentions, label_type='ner')

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        from flair.data import Sentence

        self.load_models()

        sentences = [
            Sentence(str(doc), use_tokenizer=True) for doc in docs
        ]

        spans_annotations = tars_predict(self.model, sentences, batch_size)

        return spans_annotations
