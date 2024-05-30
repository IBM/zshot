import argparse
import json
import numpy as np
import os
import spacy
from os.path import isfile, join

from zshot import PipelineConfig
from zshot.evaluation import load_medmentions_zs, load_ontonotes_zs
from zshot.evaluation.metrics.seqeval.seqeval import Seqeval
from zshot.evaluation.zshot_evaluate import evaluate, prettify_evaluate_report
from zshot.linker import LinkerSMXM
from zshot.utils.data_models import Entity


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_current_entities(all_entities, focus_entity_name, description_variation, is_only_negative):
    focus_entitity = Entity(name=focus_entity_name, description=description_variation)
    if is_only_negative:
        return [focus_entitity]
    else:
        entities = [
            entity for entity in all_entities if entity.name != focus_entity_name
        ]
        entities.append(focus_entitity)
        return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variations-path", type=str, required=True, help="Path to the folder with variations"
    )
    parser.add_argument("--dataset", type=str, required=False, default=None,
                        help="Dataset used for evaluation: `ontonotes` or `medmentions`")
    parser.add_argument("--split", type=str, default=None, required=False,
                        help="The dataset split to test on")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="The batch size")
    parser.add_argument("--model-checkpoint", type=str, default=None, required=False)
    args = parser.parse_args()

    if not os.path.exists(args.variations_path):
        raise FileNotFoundError("Variations path doesn't exist")

    output_folder = f"{args.variations_path}-eval"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    variations_files = [f for f in os.listdir(args.variations_path) if isfile(join(args.variations_path, f))]

    for variation_file in variations_files:
        print(f"evaluating variation file: {variation_file}")

        file_path = join(output_folder, variation_file) + ".eval.jsonl"

        if os.path.exists(file_path):
            print(f"Skipping {file_path} as it already exists")
            continue

        with open(join(args.variations_path, variation_file), "r") as f:
            config = json.load(f)

        split = config["split"] if args.split is None else args.split
        dataset_name = config["dataset"] if args.dataset is None else args.dataset
        model_name = config["model_checkpoint"] if args.model_checkpoint is None else args.model_checkpoint
        is_only_negative = config["is_only_negative"]
        variations = config["variations"]
        focus_entity = config["entity"]

        if dataset_name == "ontonotes":
            dataset = load_ontonotes_zs(split=split)
        elif dataset_name == "medmentions":
            dataset = load_medmentions_zs(split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        all_entities = [entity for entity in dataset.entities]

        pipeline_config = PipelineConfig(entities=all_entities, linker=LinkerSMXM(model_name=model_name))
        nlp_pipeline = spacy.blank("en")
        nlp_pipeline.add_pipe("zshot", config=pipeline_config, last=True)

        for j, (variation_desc, variation_score) in enumerate(variations):
            print(
                f"{j + 1}/{len(variations)} candidates for {focus_entity}"
            )

            entities_to_use = get_current_entities(
                all_entities, focus_entity, variation_desc, is_only_negative
            )
            nlp_pipeline.get_pipe("zshot").entities = entities_to_use
            print(f"Entities: {entities_to_use}")

            evaluation = evaluate(nlp_pipeline, dataset, metric=Seqeval(), batch_size=args.batch_size)

            print(evaluation)
            print(prettify_evaluate_report(evaluation, name=f"{dataset_name}-{split}"))

            report = dict(config)
            report.pop("variations")
            report['variation'] = f"{j + 1}/{len(variations)}"
            report['variation_desc'] = variation_desc
            report['evaluation'] = evaluation

            with open(file_path, "a+") as f:
                f.write(json.dumps(report, cls=NpEncoder) + "\n")

    print(20 * "-" + "\n" + "Evaluation done" + "\n" + 20 * "-")
