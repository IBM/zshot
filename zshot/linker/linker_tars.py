import pkgutil
from typing import Iterator, Optional, Union, List

from spacy.tokens.doc import Doc

from zshot.linker.linker import Linker
from zshot.utils.models.tars.utils import tars_predict
from zshot.utils.data_models import Entity, Span


class LinkerTARS(Linker):
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
                            "Install it with: pip install flair>=0.13")

        self.is_end2end = True
        self.default_entities = default_entities
        self.model = None
        self.task = None

    def set_kg(self, entities: Iterator[Entity]):
        """ Set new entities in the model

        :param entities: New entities to use
        """
        old_entities = self.entities
        super().set_kg(entities)
        self.flat_entities()
        if old_entities != entities:
            self.task = f'zshot.ner.{hash(tuple(self.entities))}'
            if not self.model:
                self.load_models()
            self.model.add_and_switch_to_new_task(self.task,
                                                  self.entities, label_type='ner')

    def flat_entities(self):
        """ As TARS use only the labels, take just the name of the entities and not the description """
        if isinstance(self.entities, dict):
            self._entities = list(self.entities.keys())
        if isinstance(self.entities, list):
            self._entities = [e.name if type(e) is Entity else e for e in self.entities]
        if self.entities is None:
            self._entities = []

    def load_models(self):
        """ Load TARS model if its not initialized"""
        if not self.model:
            from flair.models import TARSTagger
            self.model = TARSTagger.load('tars-ner')

            if not self.entities:
                self.model.switch_to_task(self.default_entities)
                self.task = self.default_entities
            else:
                self.flat_entities()
                self.task = f'zshot.ner.{hash(tuple(self.entities))}'
                self.model.add_and_switch_to_new_task(self.task,
                                                      self.entities, label_type='ner')

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
