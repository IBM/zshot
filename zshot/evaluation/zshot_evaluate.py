from typing import Optional, Union, Dict, List

import spacy
from datasets import Dataset
from evaluate import EvaluationModule
from prettytable import PrettyTable

from zshot.evaluation.evaluator import ZeroShotTokenClassificationEvaluator, MentionsExtractorEvaluator
from zshot.evaluation.metrics.seqeval.seqeval import Seqeval
from zshot.evaluation.pipeline import LinkerPipeline, MentionsExtractorPipeline


def evaluate(nlp: spacy.language.Language,
             dataset: Dataset,
             metric: Optional[Union[str, EvaluationModule]] = Seqeval(),
             mode: Optional[str] = 'span',
             batch_size: Optional[int] = 16,
             entity_mapper: Optional[Dict[str, str]] = None) -> dict:
    """ Evaluate a spacy zshot model

    :param nlp: Spacy Language pipeline with ZShot components
    :param dataset: Dataset used to evaluate
    :param metric: Metrics to use in evaluation.
    :param mode: Mode of token evaluation. One of: span; token. Default: span
        - span: If the entity has more than one token, all of them have to be recognised
        - token: The evaluation is done at token level,
            so if any of the tokens of the entity is missing the other are still valid
    :param batch_size: the batch size
    :param entity_mapper: Mapper for entity names
    :return: Result of the evaluation. Dict with metrics results for each component
    """
    linker_evaluator = ZeroShotTokenClassificationEvaluator(mode=mode, entity_mapper=entity_mapper)
    mentions_extractor_evaluator = MentionsExtractorEvaluator(mode=mode, entity_mapper=entity_mapper)

    results = {'evaluation_mode': mode}
    if nlp.get_pipe("zshot").linker:
        pipe = LinkerPipeline(nlp, batch_size)
        results['linker'] = linker_evaluator.compute(pipe, dataset, metric=metric)
    if nlp.get_pipe("zshot").mentions_extractor:
        pipe = MentionsExtractorPipeline(nlp, batch_size)
        results['mentions_extractor'] = mentions_extractor_evaluator.compute(pipe, dataset, metric=metric)
    return results


def prettify_evaluate_report(evaluation: Dict, name: str = "", decimals: int = 4,
                             show_full_report: Optional[bool] = True) -> List[str]:
    """
    Convert an evaluation report Dict to a formatted string
    :param evaluation: The evaluation report dict
    :param name: Reference name
    :param decimals: Number of decimals to show
    :param show_full_report: If True, it will show also the metrics for each label
    :return: Formatted evaluation table as string, for each component
    """

    def fix_table_title(table):
        rows = table.split("\n")
        len_row = len(rows[0])
        title = rows[1][1:].strip()
        n_spaces = len_row - 2 - len(title)
        n_spaces_left = n_spaces // 2
        n_spaces_right = n_spaces - n_spaces_left
        rows[1] = "|" + " " * n_spaces_left + title + " " * n_spaces_right + "|"
        second_title = rows[2][:-1].strip()
        n_spaces = len_row - 2 - len(second_title)
        n_spaces_left = n_spaces // 2
        n_spaces_right = n_spaces - n_spaces_left
        rows[2] = "|" + " " * n_spaces_left + second_title + " " * n_spaces_right + "|"
        return "\n".join(rows)

    def make_table(data: Dict, title: str = ""):
        table = PrettyTable()
        table.title = title
        table.field_names = ["Metric", ""]
        for metric in data:
            if isinstance(data[metric], (float, int)):
                table.add_row([metric, f'{data[metric]:.{decimals}f}'])
        return table

    tables = []
    mode = evaluation.get('evaluation_mode')
    for component in evaluation:
        if component == 'evaluation_mode':
            continue
        # General evaluation
        t_repr = make_table(evaluation[component], f"{component} - {name} \n General - {mode}-based").get_string()
        tables.append(fix_table_title(t_repr))

        # Classification report
        if show_full_report:
            for value in evaluation[component]:
                if isinstance(evaluation[component][value], dict):
                    t_repr = make_table(evaluation[component][value],
                                        f"{component} - {name} \n "
                                        f"{value} - {mode}-based").get_string()

                    tables.append(fix_table_title(t_repr))

    return tables
