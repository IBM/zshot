from typing import Dict, Union

from datasets import ClassLabel, load_dataset, DatasetDict, Split

from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.evaluation.dataset.ontonotes.entities import ONTONOTES_ENTITIES

LABELS = ONTONOTES_ENTITIES
labels = ClassLabel(num_classes=37,
                    names=["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC",
                           "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC",
                           "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
                           "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
                           "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT",
                           "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"])

CLASSES_PER_SPLIT = {
    "train": ["PERSON", "GPE", "ORG", "DATE"],
    "validation": ["NORP", "MONEY", 'ORDINAL', "PERCENT", "EVENT", "PRODUCT", "LAW"],
    "test": ["CARDINAL", "TIME", "LOC", "WORK_OF_ART", "FAC", "QUANTITY", "LANGUAGE"]
}
TRIVIAL_CLASSES = ["ORDINAL", "QUANTITY", "MONEY", "PERCENT", "CARDINAL", "LANGUAGE", "TIME"]


def remove_other_tasks(sentence):
    if 'pos_tags' in sentence:
        del sentence['pos_tags']
    if 'parse_tree' in sentence:
        del sentence['parse_tree']
    if 'predicate_framenet_ids' in sentence:
        del sentence['predicate_framenet_ids']
    if 'word_senses' in sentence:
        del sentence['word_senses']
    if 'speaker' in sentence:
        del sentence['speaker']
    if 'predicate_lemmas' in sentence:
        del sentence['predicate_lemmas']
    if 'coref_spans' in sentence:
        del sentence['coref_spans']
    if 'srl_frames' in sentence:
        del sentence['srl_frames']
    return sentence


def is_not_empty(sentence):
    return not all([s == 0 for s in sentence['named_entities']])


def remove_out_of_split(sentence, split):
    for i, ent in enumerate(sentence['named_entities']):
        label = labels.int2str(ent)
        if label == 'O' or label[2:] in TRIVIAL_CLASSES or label[2:] not in CLASSES_PER_SPLIT[split]:
            sentence['named_entities'][i] = 0
    return sentence


def load_ontonotes() -> Dict[Union[str, Split], DatasetWithEntities]:
    dataset_zs = load_dataset("conll2012_ontonotesv5", "english_v12")
    ontonotes_zs = DatasetDict()

    for split in dataset_zs:
        dataset_zs[split] = dataset_zs[split].map(lambda example, idx: {
            "sentences": [remove_out_of_split(s, split) for s in example['sentences']]
        }, with_indices=True)
        dataset_zs[split] = dataset_zs[split].map(lambda example, idx: {
            "sentences": list(filter(is_not_empty, example['sentences']))
        }, with_indices=True)

        tokens = []
        ner_tags = []
        for example in dataset_zs[split]:
            tokens += [s['words'] for s in example['sentences']]
            ner_tags += [[labels.int2str(ent) for ent in s['named_entities']] for s in example['sentences']]

        split_entities = [ent for ent in ONTONOTES_ENTITIES
                          if ent.name in ['NEG'] + CLASSES_PER_SPLIT[split] and ent.name not in TRIVIAL_CLASSES]

        ontonotes_zs[split] = DatasetWithEntities.from_dict({
            'tokens': tokens,
            'ner_tags': ner_tags
        }, split=split, entities=split_entities)

    return ontonotes_zs
