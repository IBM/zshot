from typing import Optional, Union, Dict

import spacy
from datasets import Dataset
from evaluate import EvaluationModule
from prettytable import PrettyTable

from zshot.evaluation.evaluator import ZeroShotTokenClassificationEvaluator, MentionsExtractorEvaluator
from zshot.evaluation.pipeline import LinkerPipeline, MentionsExtractorPipeline


def evaluate(nlp: spacy.language.Language,
             dataset: Dataset,
             metric: Optional[Union[str, EvaluationModule]] = None,
             batch_size: Optional[int] = 16) -> dict:
    """ Evaluate a spacy zshot model

    :param nlp: Spacy Language pipeline with ZShot components
    :param dataset: Dataset used to evaluate
    :param metric: Metrics to use in evaluation.
    :return: Result of the evaluation. Dict with metrics results for each component
    :param batch_size: the batch size
    """
    linker_evaluator = ZeroShotTokenClassificationEvaluator()
    mentions_extractor_evaluator = MentionsExtractorEvaluator()

    results = {}
    if nlp.get_pipe("zshot").linker:
        pipe = LinkerPipeline(nlp, batch_size)
        results['linker'] = linker_evaluator.compute(pipe, dataset, metric=metric)
    if nlp.get_pipe("zshot").mentions_extractor:
        pipe = MentionsExtractorPipeline(nlp, batch_size)
        results['mentions_extractor'] = mentions_extractor_evaluator.compute(pipe, dataset, metric=metric)
    return results


def prettify_evaluate_report(evaluation: Dict, name: str = "", decimals: int = 4) -> list[PrettyTable]:
    """
    Convert an evaluation report Dict to a formatted string
    :param evaluation: The evaluation report dict
    :param name: Reference name
    :param decimals: Number of decimals to show
    :return: Formatted evaluation table as string, for each component
    """
    tables = []
    for component in evaluation:
        table = PrettyTable()
        table.field_names = ["Metric", name]
        for metric in evaluation[component]:
            if isinstance(evaluation[component][metric], (float, int)):
                table.add_row([metric, f'{evaluation[component][metric]:.{decimals}f}'])
        tables.append(table)
    return tables
