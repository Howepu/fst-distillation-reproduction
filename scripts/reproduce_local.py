"""Run a self-contained local reproduction without a hosted W&B sweep.

The upstream ``src.sweep`` entry point assumes access to the authors' W&B
entity and launches 50 neural plus 100 extraction runs for small datasets.
This runner keeps W&B offline, trains one explicitly configured RNN, and
performs a deterministic extraction grid selected by validation F1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import torch
import wandb

from src.extract_fst import (
    ExtractionHyperparameters,
    compute_activations,
    extract_fst,
)
from src.paths import create_paths
from src.run_alignment import run_alignment
from src.train_rnn import train_rnn


def comma_separated_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def comma_separated_thresholds(value: str) -> list[int | None]:
    return [
        None if item.strip().lower() == "none" else int(item)
        for item in value.split(",")
        if item
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="data/inflection")
    parser.add_argument("--dataset", default="czn")
    parser.add_argument("--models-folder", default="models")
    parser.add_argument("--results-folder", default="results")
    parser.add_argument("--run-name", default="local-czn-transduction-seed0")
    parser.add_argument("--features", action="store_true", default=True)
    parser.add_argument(
        "--merge-outputs", choices=("none", "right", "bpe"), default="bpe"
    )
    parser.add_argument(
        "--full-domain-mode", choices=("sample", "search"), default="sample"
    )
    parser.add_argument("--alignment-iterations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clusters", type=comma_separated_ints, default="50,100,150,200")
    parser.add_argument(
        "--thresholds", type=comma_separated_thresholds, default="none,2,5,10"
    )
    parser.add_argument("--search-n", type=int, default=3)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.time()
    paths = create_paths(
        data_folder=args.data_folder,
        dataset=args.dataset,
        has_features=args.features,
        output_split=False,
        merge_outputs=args.merge_outputs,
        models_folder=args.models_folder,
    )

    if not paths["train_aligned"].exists() or not paths["eval_aligned"].exists():
        run_alignment(paths, iterations=args.alignment_iterations)

    if not args.skip_training:
        run = wandb.init(
            project="fst-distillation-local",
            name=args.run_name,
            mode="offline",
            config={},
        )
        train_rnn(
            paths=paths,
            objective="transduction",
            batch_size=args.batch_size,
            epochs=args.epochs,
            d_model=args.d_model,
            num_layers=1,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            merge_outputs=args.merge_outputs,
            activation="tanh",
            spectral_norm_weight=0.1,
            label_smoothing=0.1,
            wandb_run=run,
            seed=args.seed,
        )

    checkpoint = paths["models_folder"] / args.run_name / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint does not exist: {checkpoint}")

    activation_hparams = ExtractionHyperparameters(
        model_shortname=args.run_name,
        clustering_method="kmeans",
        kmeans_num_clusters=50,
        full_domain_mode=args.full_domain_mode,
        full_domain_search_n=args.search_n,
    )
    activations, transition_labels = compute_activations(activation_hparams, paths)
    unique_activations = len(np.unique(activations, axis=0))

    trials: list[dict] = []
    best_trial: dict | None = None
    best_fst = None
    for clusters in args.clusters:
        if clusters > unique_activations:
            continue
        for classifier in ("svm", "logistic"):
            for threshold in args.thresholds:
                hparams = ExtractionHyperparameters(
                    model_shortname=args.run_name,
                    clustering_method="kmeans",
                    kmeans_num_clusters=clusters,
                    use_faiss=False,
                    state_split_classifier=classifier,
                    minimum_transition_count=threshold,
                    full_domain_mode=args.full_domain_mode,
                    full_domain_search_n=args.search_n,
                )
                metrics, fst = extract_fst(
                    hparams,
                    paths,
                    precomputed_activations=(activations, transition_labels),
                )
                trial = {
                    "clusters": clusters,
                    "classifier": classifier,
                    "minimum_transition_count": threshold,
                    "metrics": metrics,
                }
                trials.append(trial)
                if best_trial is None or metrics["eval"]["f1"] > best_trial["metrics"]["eval"]["f1"]:
                    best_trial = trial
                    best_fst = fst

    if best_trial is None or best_fst is None:
        raise RuntimeError("The extraction grid produced no FST")

    results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    result_path = results_folder / f"{args.dataset}-local-reproduction.json"
    result = {
        "dataset": args.dataset,
        "selection_metric": "validation F1",
        "paper_test_accuracy": 0.666 if args.dataset == "czn" else None,
        "best": best_trial,
        "trials": trials,
        "unique_activations": unique_activations,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "arguments": vars(args),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # PyFoma serializes its state graph recursively.  Large extracted FSTs can
    # exceed Python's conservative default recursion depth during pickling.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    best_fst.save(str(results_folder / f"{args.dataset}-best.fst"))
    print(json.dumps({"result": str(result_path), "best": best_trial}, indent=2))


if __name__ == "__main__":
    main()
