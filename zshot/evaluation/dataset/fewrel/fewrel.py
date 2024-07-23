from typing import Optional, Union, Dict

from datasets import load_dataset, Split, Dataset
from tqdm import tqdm

from zshot.evaluation.dataset.dataset import DatasetWithRelations
from zshot.utils.data_models import Relation


def get_entity_data(e, tokenized_sentence):
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
    return d


def load_few_rel_zs(split: Optional[Union[str, Split]] = "val_wiki") -> Union[Dict[DatasetWithRelations,
                                                                                   Dataset], Dataset]:
    dataset = load_dataset("few_rel", split=split, trust_remote_code=True)
    relations_descriptions = dataset["names"]
    tokenized_sentences = dataset["tokens"]
    sentences = [" ".join(tokens) for tokens in tokenized_sentences]
    gt = [item[0] for item in relations_descriptions]
    heads = dataset["head"]
    tails = dataset["tail"]
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
    relations = [Relation(name=name, description=desc) for name, desc in
                 set([(i, j) for i, j in relations_descriptions])]
    dataset = Dataset.from_dict({
        "sentences": sentences,
        "sentence_entities": entities_data,
        "labels": gt,
    })
    dataset.relations = relations
    return dataset
