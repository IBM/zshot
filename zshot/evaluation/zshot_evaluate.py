from typing import Optional, List, Union

import spacy
from evaluate import EvaluationModule
from prettytable import PrettyTable

from zshot.evaluation import load_medmentions, load_ontonotes
from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.evaluation.evaluator import ZeroShotTokenClassificationEvaluator, MentionsExtractorEvaluator
from zshot.evaluation.pipeline import LinkerPipeline, MentionsExtractorPipeline


def evaluate(nlp: spacy.language.Language,
             datasets: Union[DatasetWithEntities, List[DatasetWithEntities]],
             splits: Optional[Union[str, List[str]]] = None,
             metric: Optional[Union[str, EvaluationModule]] = None,
             batch_size: Optional[int] = 16) -> str:
    """ Evaluate a spacy zshot model

    :param nlp: Spacy Language pipeline with ZShot components
    :param datasets: Dataset or list of datasets to evaluate
    :param splits: Optional. Split or list of splits to evaluate. All splits available by default
    :param metric: Metrics to use in evaluation.
        Options available: precision, recall, f1-score-micro, f1-score-macro. All by default
    :return: Result of the evaluation. Dict with precision, recall and f1-score for each component
    :param batch_size: the batch size
    """
    linker_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")
    mentions_extractor_evaluator = MentionsExtractorEvaluator("token-classification")

    if type(splits) == str:
        splits = [splits]

    result = {}
    field_names = ["Metric"]
    for dataset_name in datasets:
        if dataset_name.lower() == "medmentions":
            dataset = load_medmentions()
        else:
            dataset = load_ontonotes()

        for split in splits:
            field_name = f"{dataset_name} {split}"
            field_names.append(field_name)
            nlp.get_pipe("zshot").entities = dataset[split].entities
            if nlp.get_pipe("zshot").linker:
                pipe = LinkerPipeline(nlp, batch_size)
                result.update(
                    {
                        field_name: {
                            'linker': linker_evaluator.compute(pipe, dataset[split], metric=metric)
                        }
                    }
                )
            # TODO: Add support for mentions_extractor pipelines and evaluation
            if nlp.get_pipe("zshot").mentions_extractor:
                pipe = MentionsExtractorPipeline(nlp, batch_size)
                result.update(
                    {
                        field_name: {
                            'mentions_extractor': mentions_extractor_evaluator.compute(pipe, dataset[split],
                                                                                       metric=metric)
                        }
                    }
                )

    table = PrettyTable()
    table.field_names = field_names
    rows = []
    if nlp.get_pipe("zshot").linker:
        linker_precisions = []
        linker_recalls = []
        linker_micros = []
        linker_macros = []
        linker_accuracies = []
        for field_name in field_names:
            if field_name == "Metric":
                continue
            linker_precisions.append("{:.2f}%".format(result[field_name]['linker']['overall_precision_macro'] * 100))
            linker_recalls.append("{:.2f}%".format(result[field_name]['linker']['overall_recall_macro'] * 100))
            linker_accuracies.append("{:.2f}%".format(result[field_name]['linker']['overall_accuracy'] * 100))
            linker_micros.append("{:.2f}%".format(result[field_name]['linker']['overall_f1_micro'] * 100))
            linker_macros.append("{:.2f}%".format(result[field_name]['linker']['overall_f1_macro'] * 100))

        rows.append(["Linker Precision"] + linker_precisions)
        rows.append(["Linker Recall"] + linker_recalls)
        rows.append(["Linker Accuracy"] + linker_accuracies)
        rows.append(["Linker F1-score micro"] + linker_micros)
        rows.append(["Linker F1-score macro"] + linker_macros)

    if nlp.get_pipe("zshot").mentions_extractor:
        mentions_extractor_precisions = []
        mentions_extractor_recalls = []
        mentions_extractor_micros = []
        mentions_extractor_accuracies = []
        mentions_extractor_macros = []
        for field_name in field_names:
            if field_name == "Metric":
                continue
            mentions_extractor_precisions.append(
                "{:.2f}%".format(result[field_name]['mentions_extractor']['overall_precision_macro'] * 100))
            mentions_extractor_recalls.append(
                "{:.2f}%".format(result[field_name]['mentions_extractor']['overall_recall_macro'] * 100))
            mentions_extractor_accuracies.append(
                "{:.2f}%".format(result[field_name]['mentions_extractor']['overall_accuracy'] * 100))
            mentions_extractor_micros.append(
                "{:.2f}%".format(result[field_name]['mentions_extractor']['overall_f1_micro'] * 100))
            mentions_extractor_macros.append(
                "{:.2f}%".format(result[field_name]['mentions_extractor']['overall_f1_macro'] * 100))

        rows.append(["Mentions extractor Precision"] + mentions_extractor_precisions)
        rows.append(["Mentions extractor Recall"] + mentions_extractor_recalls)
        rows.append(["Mentions extractor Accuracy"] + mentions_extractor_accuracies)
        rows.append(["Mentions extractor F1-score micro"] + mentions_extractor_micros)
        rows.append(["Mentions extractor F1-score macro"] + mentions_extractor_macros)

    table.add_rows(rows)

    return table.get_string()
