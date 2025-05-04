import contextlib
import logging
import pkgutil
from typing import List, Tuple, Iterator, Optional, Union

from relik import Relik
from relik.inference.data.objects import RelikOutput
from spacy.tokens import Doc

from zshot.config import MODELS_CACHE_PATH
from zshot.knowledge_extractor.knowledge_extractor import KnowledgeExtractor
from zshot.utils.data_models import Span, Relation
from zshot.utils.data_models.relation_span import RelationSpan

logging.getLogger("relik").setLevel(logging.ERROR)

MODEL_NAME = "sapienzanlp/relik-relation-extraction-nyt-large"


class KnowledgeExtractorRelik(KnowledgeExtractor):
    def __init__(self, model_name=MODEL_NAME):
        """ Instantiate the KnowGL Knowledge Extractor """
        super().__init__()

        if not pkgutil.find_loader("relik"):
            raise Exception("relik module not installed. "
                            "You need to install relik in order to use the relik Knowledge Extractor."
                            "Install it with: pip install relik")

        self.model_name = model_name
        self.model = None

    def load_models(self):
        """ Load relik model """
        # Remove RELIK print
        with contextlib.redirect_stdout(None):
            if self.model is None:
                self.model = Relik.from_pretrained(self.model_name,
                                                   cache_dir=MODELS_CACHE_PATH, device=self.device)

    def parse_result(self, relik_out: RelikOutput, doc: Doc) -> List[Tuple[Span, RelationSpan, Span]]:
        triples = []
        for triple in relik_out.triplets:
            subject = Span(triple.subject.start, triple.subject.end, triple.subject.label)
            object_ = Span(triple.object.start, triple.object.end, triple.object.label)

            relation = Relation(name=triple.label, description="")
            triples.append((subject,
                            RelationSpan(start=subject, end=object_, relation=relation),
                            object_))
        return triples

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) \
            -> List[List[Tuple[Span, RelationSpan, Span]]]:
        """ Extract triples from docs

        :param docs: Spacy Docs to process
        :param batch_size: Batch size for processing
        :return: Triples (subject, relation, object) extracted for each document
        """
        if not self.model:
            self.load_models()

        texts = [d.text for d in docs]
        relik_out = self.model(texts)
        if type(relik_out) is RelikOutput:
            relik_out = [relik_out]

        triples = []
        for doc, output in zip(docs, relik_out):
            triples.append(self.parse_result(output, doc))

        return triples
