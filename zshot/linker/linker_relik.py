import contextlib
import logging
import pkgutil
from typing import Iterator, List, Optional, Union

from relik import Relik
from relik.inference.data.objects import RelikOutput
from relik.retriever.indexers.document import Document
from spacy.tokens import Doc

from zshot.config import MODELS_CACHE_PATH
from zshot.linker.linker import Linker
from zshot.utils.data_models import Span

logging.getLogger("relik").setLevel(logging.ERROR)

MODEL_NAME = "sapienzanlp/relik-entity-linking-large"


class LinkerRelik(Linker):
    """ Relik linker """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        if not pkgutil.find_loader("relik"):
            raise Exception("relik module not installed. You need to install relik in order to use the relik Linker."
                            "Install it with: pip install relik")

        self.model_name = model_name
        self.model = None
        # self.device = {
        #     "retriever_device": self.device,
        #     "index_device": self.device,
        #     "reader_device": self.device
        # }

    @property
    def is_end2end(self) -> bool:
        """ relik is end2end """
        return True

    def load_models(self):
        """ Load relik model """
        # Remove RELIK print
        with contextlib.redirect_stdout(None):
            if self.model is None:
                if self._entities:
                    self.model = Relik.from_pretrained(self.model_name,
                                                       cache_dir=MODELS_CACHE_PATH,
                                                       retriever=None, device=self.device)
                else:
                    self.model = Relik.from_pretrained(self.model_name,
                                                       cache_dir=MODELS_CACHE_PATH, device=self.device,
                                                       index_device='cpu')

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
        """
        candidates = None
        if self._entities:
            candidates = [
                Document(text=ent.name, id=i, metadata={'definition': ent.description})
                for i, ent in enumerate(self._entities)
            ]

        sentences = [doc.text for doc in docs]

        self.load_models()
        span_annotations = []
        for sent in sentences:
            relik_out: RelikOutput = self.model(sent, candidates=candidates)
            span_annotations.append([Span(start=relik_span.start, end=relik_span.end, label=relik_span.label)
                                     for relik_span in relik_out.spans])

        return span_annotations
