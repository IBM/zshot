import os

import spacy
from tqdm import tqdm

from zshot.evaluation.dataset.med_mentions.entities import MEDMENTIONS_TYPE_INV


class Token(object):
    def __init__(self, word, label, label_id):
        self.word = word
        self.label = label
        self.label_id = label_id


def convert_to_iob(id_, text, entities, nlp, end_index, start_index):
    count = 0
    sentences = []
    current_sent_pos = 0
    doc = nlp(text)
    for sent in doc.sents:
        sentence = []
        for tok in sent:
            word = tok.text
            ent = entities[0][1] if entities else "O"
            if entities and tok.idx + current_sent_pos == start_index[0]:
                count += 1
                label = 'B-' + MEDMENTIONS_TYPE_INV[ent]
                label_id = 'B-' + ent
                token = Token(word=word, label=label, label_id=label_id)
            elif entities and start_index[0] < tok.idx + current_sent_pos <= end_index[0]:
                label = 'I-' + MEDMENTIONS_TYPE_INV[ent]
                label_id = 'I-' + ent
                token = Token(word=word, label=label, label_id=label_id)
            else:
                token = Token(word=word, label='O', label_id="O")
            if entities and len(word) + tok.idx + current_sent_pos >= end_index[0]:
                start_index.pop(0)
                end_index.pop(0)
                entities.pop(0)
            sentence.append(token)
        current_sent_pos += len(sent) + 1
        sentences.append((id_, sentence))
    return sentences, count


def preprocess_medmentions(input_path):
    data_path = os.path.join(input_path, 'corpus_pubtator.txt')
    train_id_path = os.path.join(input_path, 'corpus_pubtator_pmids_train.txt')
    dev_id_path = os.path.join(input_path, 'corpus_pubtator_pmids_dev.txt')
    test_id_path = os.path.join(input_path, 'corpus_pubtator_pmids_test.txt')

    nlp = spacy.load("en_core_web_sm")
    sentences = []
    with open(data_path, 'r') as f:
        data = f.readlines()
        ids = []
        titles = []
        abstracts = []
        entities_title = []
        entities_abstract = []
        ends_title = []
        ends_abstract = []
        starts_title = []
        starts_abstract = []
        id_tmp = None
        starts_title_tmp = []
        starts_abstract_tmp = []
        ends_title_tmp = []
        ends_abstract_tmp = []
        entities_title_tmp = []
        entities_abstract_tmp = []
        for line in data:
            if '|t|' in line:
                id_tmp = line.split("|t|")[0]
                ids.append(id_tmp)
                titles.append(line.split("|t|")[1])
            elif '|a|' in line:
                abstracts.append(line.split('|a|')[1])
            elif id_tmp in line:
                _, start_idx, end_idx, ent, label, _ = line.split('\t')
                if len(ends_title_tmp) > 0 and int(end_idx) <= ends_title_tmp[-1] or \
                        len(ends_abstract_tmp) > 0 and int(end_idx) - len(titles[-1]) <= ends_abstract_tmp[-1]:
                    continue
                if int(end_idx) < len(titles[-1]):
                    starts_title_tmp.append(int(start_idx))
                    ends_title_tmp.append(int(end_idx))
                    entities_title_tmp.append((ent, label))
                else:
                    starts_abstract_tmp.append(int(start_idx) - len(titles[-1]))
                    ends_abstract_tmp.append(int(end_idx) - len(titles[-1]))
                    entities_abstract_tmp.append((ent, label))
            else:
                starts_title.append(starts_title_tmp)
                ends_title.append(ends_title_tmp)
                entities_title.append(entities_title_tmp)
                starts_abstract.append(starts_abstract_tmp)
                ends_abstract.append(ends_abstract_tmp)
                entities_abstract.append(entities_abstract_tmp)
                starts_title_tmp = []
                starts_abstract_tmp = []
                ends_title_tmp = []
                ends_abstract_tmp = []
                entities_title_tmp = []
                entities_abstract_tmp = []
                id_tmp = None

        count = 0
        for id_, title, entities_title_tmp, start_title, end_title, abstract, entities_abstract_tmp, \
            start_abstract, end_abstract in tqdm(zip(ids, titles, entities_title, starts_title, ends_title,
                                                     abstracts, entities_abstract, starts_abstract, ends_abstract)):
            sentences_title, count_t = convert_to_iob(id_, title, entities_title_tmp, nlp,
                                                      end_title, start_title)
            sentences_abstract, count_a = convert_to_iob(id_, abstract, entities_abstract_tmp, nlp,
                                                         end_abstract, start_abstract)
            count += count_a + count_t
            sentences += sentences_title
            sentences += sentences_abstract

    with open(train_id_path, "r") as f:
        train_ids = [line.strip() for line in f.readlines()]
        train_sentences = [sent for id_, sent in sentences if id_ in train_ids]
    with open(dev_id_path, "r") as f:
        dev_ids = [line.strip() for line in f.readlines()]
        dev_sentences = [sent for id_, sent in sentences if id_ in dev_ids]
    with open(test_id_path, "r") as f:
        test_ids = [line.strip() for line in f.readlines()]
        test_sentences = [sent for id_, sent in sentences if id_ in test_ids]

    return train_sentences, dev_sentences, test_sentences
