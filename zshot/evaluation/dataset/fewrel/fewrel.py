from typing import Optional, Union

from datasets import load_dataset, Split
from tqdm import tqdm


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


def get_few_rel_data(split_name: Optional[Union[str, Split]] = "val_wiki"):
    wiki_val = load_dataset("few_rel", split=split_name)
    relations_descriptions = wiki_val["names"]
    tokenized_sentences = wiki_val["tokens"]
    sentences = [" ".join(tokens) for tokens in tokenized_sentences]
    gt = [item[0] for item in relations_descriptions]
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
