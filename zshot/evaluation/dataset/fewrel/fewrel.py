from datasets import load_dataset
from tqdm import tqdm

from zshot import Linker
from zshot.utils.data_models import Span


class DummyLinkerEnd2End(Linker):
    @property
    def is_end2end(self) -> bool:
        return True

    def predict(self, data):
        return [
            [
                Span(
                    item["start"],
                    item["end"],
                    item["label"],
                )
                for item in doc_ents
            ]
            for doc_ents in enumerate(data)
        ]


def get_entity_data(e, tokenized_sentence):
    # import pdb
    # pdb.set_trace()
    d = {"start": None, "end": None, "label": e["type"]}
    token_indices = e["indices"][0]
    s = ""
    curr_idx = 0
    for idx, token in enumerate(tokenized_sentence):
        if idx == token_indices[0]:
            d["start"] = curr_idx
        s += token + " "
        curr_idx = len(s.strip())
        if idx == token_indices[-1]:
            d["end"] = curr_idx
    d["sentence"] = s.strip()
    # pdb.set_trace()
    return d


def get_few_rel_data(split_name="val_wiki", limit=-1):
    wiki_val = load_dataset("few_rel", split=split_name)
    relations_descriptions = wiki_val["names"][:limit]
    tokenized_sentences = wiki_val["tokens"][:limit]
    sentences = [" ".join(tokens) for tokens in tokenized_sentences]
    if limit != -1:
        sentences = sentences[:limit]

    gt = [item[0] for item in relations_descriptions]
    # label_mapping = {l: idx for idx, l in enumerate(list(set(gt)))}
    # gt = [label_mapping.get(item) for item in gt]
    heads = wiki_val["head"]
    tails = wiki_val["tail"]
    entities_data = []
    for idx in tqdm(range(len(tokenized_sentences))):
        e1 = heads[idx]
        e2 = tails[idx]
        entities_data.append(
            [
                get_entity_data(e1, tokenized_sentences[idx]),
                get_entity_data(e2, tokenized_sentences[idx]),
            ]
        )

    return entities_data, sentences, relations_descriptions, gt


if __name__ == "__main__":
    get_few_rel_data()
