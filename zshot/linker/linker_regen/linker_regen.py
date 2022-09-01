import pkgutil
from typing import Iterator, Optional, Union, List

from spacy.tokens import Doc

from zshot.utils.data_models import Entity, Span
from zshot.linker.linker import Linker
from zshot.linker.linker_regen.trie import Trie
from zshot.linker.linker_regen.utils import create_input

MODEL_NAME = "gabriele-picco/regen-disambiguation"

START_ENT_TOKEN = "[START_ENT]"
END_ENT_TOKEN = "[END_ENT]"


class LinkerRegen(Linker):

    def __init__(self, max_input_len=384, max_output_len=15, num_beams=10):
        super().__init__()

        if not pkgutil.find_loader("transformers"):
            raise Exception("transformers module not installed. You need to install transformers in order to use this"
                            " Linker. Install it with: pip install transformers")
        self.model = None
        self.tokenizer = None
        self.trie = None
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.num_beams = num_beams

    def set_kg(self, entities: Iterator[Entity]):
        super().set_kg(entities)
        self.load_tokenizer()
        self.trie = Trie(
            [
                self.tokenizer(e.name, return_tensors="pt")['input_ids'][0].tolist()
                for e in entities
            ]
        )

    def load_models(self):
        from transformers import AutoModelForSeq2SeqLM
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.load_tokenizer()

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=1024)

    def restrict_decode_vocab(self, _, prefix_beam):
        return self.trie.postfix(prefix_beam.tolist())

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        self.load_models()
        data_to_link = []
        docs = list(docs)
        for doc_id, doc in enumerate(docs):
            for mention_id, mention in enumerate(doc._.mentions):
                left_context = doc.text[:mention.start_char]
                right_context = doc.text[mention.end_char:]
                sentence = f"{left_context} {START_ENT_TOKEN} {mention.text} {END_ENT_TOKEN} {right_context}"
                data_to_link.append(
                    {
                        "id": doc_id,
                        "mention_id": mention_id,
                        "text": sentence,
                    })

        sentences = [create_input(d['text'],
                                  max_length=self.max_input_len,
                                  start_delimiter=START_ENT_TOKEN,
                                  end_delimiter=END_ENT_TOKEN,
                                  ) for d in data_to_link]
        input_args = {
            k: v
            for k, v in self.tokenizer.batch_encode_plus(
                sentences, padding=True, return_tensors="pt"
            ).items()
        }

        outputs = self.model.generate(
            **input_args,
            min_length=0,
            max_length=self.max_output_len,
            num_beams=self.num_beams,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            prefix_allowed_tokens_fn=None
            if self.trie is None
            else self.restrict_decode_vocab,
        )

        docs_pred = {}

        for data, out, score in zip(data_to_link, outputs.sequences, outputs.sequences_scores):
            doc_id = data['id']
            mention = docs[doc_id]._.mentions[data['mention_id']]
            label = self.tokenizer.decode(out, skip_special_tokens=True)
            if doc_id not in docs_pred:
                docs_pred[doc_id] = []
            docs_pred[doc_id].append(Span(mention.start_char, mention.end_char, label=label,
                                          score=score.detach().numpy().tolist()))
        return [val for key, val in sorted(docs_pred.items(), reverse=False)]
