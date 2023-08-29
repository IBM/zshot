from typing import List, Tuple, Iterator, Optional, Union

from spacy.tokens import Doc
from tokenizers import Encoding
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from zshot.knowledge_extractor.knowgl.utils import get_words_mappings, get_spans, get_triples
from zshot.knowledge_extractor.knowledge_extractor import KnowledgeExtractor
from zshot.utils.data_models import Span
from zshot.utils.data_models.relation_span import RelationSpan


class KnowGL(KnowledgeExtractor):
    def __init__(self, model_name="ibm/knowgl-large"):
        """ Instantiate the KnowGL Knowledge Extractor """
        super().__init__()

        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_models(self):
        """ Load KnowGL model """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def parse_result(self, result: str, doc: Doc,
                     encodings: Encoding) -> List[Tuple[Span, RelationSpan, Span]]:
        """  Parse the text result into a list of triples

        :param result: Text generate by the KnowGL model
        :param doc: Spacy doc
        :param encodings: Encodings result of the tokenization
        :return: List of triples (subject, relation, object)
        """
        words_mapping, char_mapping = get_words_mappings(encodings, doc.text)
        triples = []
        for triple in result.split("$"):
            subject_, relation, object_ = triple.split("|")
            s_mention, s_label, s_type = subject_.strip("[()]").split("#")
            o_mention, o_label, o_type = object_.strip("[()]").split("#")
            s_type = s_label if s_label != "None" else s_type
            o_type = o_label if o_label != "None" else o_type
            subject_spans = get_spans(s_mention, s_type, self.tokenizer, encodings,
                                      words_mapping, char_mapping)
            object_spans = get_spans(o_mention, o_type, self.tokenizer, encodings,
                                     words_mapping, char_mapping)
            triples += get_triples(subject_spans, relation, object_spans)

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
        input_data = self.tokenizer(texts,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="pt")
        input_ids = input_data.input_ids.to(self.model.device)
        outputs = self.model.generate(inputs=input_ids)

        triples = []
        for doc, output, encodings in zip(docs, outputs, input_data.encodings):
            result = self.tokenizer.decode(token_ids=output, skip_special_tokens=True)
            triples.append(self.parse_result(result, doc, encodings))

        return triples
