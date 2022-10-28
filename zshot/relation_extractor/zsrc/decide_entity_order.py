import numpy as np
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def score(premise, e1_text, e2_text, rel_name):
    sentence1 = "{} {} {}".format(e1_text, rel_name, e2_text)
    sentence2 = "{} {} {}".format(e2_text, rel_name, e1_text)
    output = classifier(premise, (sentence1, sentence2))
    scores = output["scores"]
    if np.argmax(scores) == output["labels"].index(sentence1):
        return e1_text, e2_text
    else:
        return e2_text, e1_text


def has_negation(premise, e1_text, e2_text, rel_name):
    if "is " in rel_name:
        negated_rel_name = rel_name.replace("is", "is not", 1)
    else:
        negated_rel_name = "does not " + rel_name
    negated = "{} {} {}".format(e1_text, negated_rel_name, e2_text)
    positive = "{} {} {}".format(e1_text, rel_name, e2_text)
    output = classifier(premise, (negated, positive))
    return np.argmax(output["scores"]) == output["labels"].index(negated)


def get_entity_order(e1_text, e2_text, rel_name, sentence):
    return score(sentence, e1_text, e2_text, rel_name)


if __name__ == "__main__":
    print("models downloaded")
