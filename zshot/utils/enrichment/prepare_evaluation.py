import argparse
import json
import os

import spacy
from zshot.utils.enrichment import (
    FineTunedLMExtensionStrategy,
    ParaphrasingStrategy,
    PreTrainedLMExtensionStrategy,
    SummarizationStrategy, EntropyHeuristic,
)
from zshot import PipelineConfig

from zshot.evaluation import load_medmentions_zs, load_ontonotes_zs
from zshot.linker import LinkerSMXM


strategies_map = {
    "pretrained": PreTrainedLMExtensionStrategy(),
    "finetuned": FineTunedLMExtensionStrategy(),
    "summarization": SummarizationStrategy(),
    "paraphrasing": ParaphrasingStrategy()
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset used for evaluation: `ontonotes` or `medmentions`")
    parser.add_argument("--model-checkpoint", type=str, default="ibm/smxm")
    parser.add_argument("--variations", type=int, default=2,
                        help="Number of description variations that will be generated for each strategy", )
    parser.add_argument("--split", type=str, default="test[0:5]",
                        help="The dataset split to test on")
    parser.add_argument('--strategies',
                        help='strategies to use, one or more in: pretrained, finetuned, summarization, paraphrasing',
                        default="paraphrasing")
    parser.add_argument("--is-only-negative", type=bool, default=False,
                        help="Only use the negative entity for evaluation")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="The batch size")
    parser.add_argument("--out", type=str, default="results",
                        help="The output folder")
    args = parser.parse_args()

    print(args)

    if args.dataset == "ontonotes":
        dataset = load_ontonotes_zs(split=args.split)
    elif args.dataset == "medmentions":
        dataset = load_medmentions_zs(split=args.split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    all_entities = [entity for entity in dataset.entities if entity.name != "NEG"]
    pipeline_config = PipelineConfig(entities=all_entities, linker=LinkerSMXM(model_name=args.model_checkpoint))
    nlp_pipeline = spacy.blank("en")
    nlp_pipeline.add_pipe("zshot", config=pipeline_config, last=True)

    results_folder = f"./{args.out}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    print(f"Dataset: {args.dataset}")
    print(f"Dataset size: {len(dataset)}")
    entropy_heuristic = EntropyHeuristic()
    strategies = args.strategies.split(" ")
    for strategy_name in strategies:
        strategy = strategies_map[strategy_name]
        print(f"Strategy: {strategy_name}")
        for entity in all_entities:
            filepath = os.path.join(results_folder, f"{entity.name}_strategy_{strategy_name}"
                                                    f"_variations__{args.variations}"
                                                    f"_{args.split}_{args.is_only_negative}.json")
            if os.path.exists(filepath):
                print(f"Skipping {entity.name} as it already exists")
                continue
            print(f"Entity: {entity.name}")
            variations = entropy_heuristic.evaluate_variations_strategy_for_entity(dataset,
                                                                                   entities=all_entities,
                                                                                   focus_entity=entity,
                                                                                   alter_strategy=strategy,
                                                                                   num_variations=args.variations,
                                                                                   nlp_pipeline=nlp_pipeline,
                                                                                   is_only_negative=args.is_only_negative,
                                                                                   batch_size=args.batch_size)

            config = {"strategy": strategy_name, "num_variations": args.variations, "entity": entity.name,
                      "is_only_negative": args.is_only_negative, "dataset": args.dataset, "split": args.split,
                      "variations": variations, "original_description": entity.description,
                      "model_checkpoint": args.model_checkpoint}
            with open(filepath, "w+") as f:
                f.write(json.dumps(config))
