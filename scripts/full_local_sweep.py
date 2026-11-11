"""Run the paper's two-stage Bayesian sweep locally without hosted W&B.

The search spaces and run counts mirror ``src.sweep`` for datasets with fewer
than 5,000 training examples: 50 RNN trials followed by 100 FST extraction
trials. Optuna's seeded TPE sampler replaces the hosted W&B Bayesian scheduler.
Both studies use resumable SQLite storage and select configurations only by the
development split.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import wandb

from src.data.aligned.example import load_examples_from_file
from src.extract_fst import (
    ExtractionHyperparameters,
    compute_activations,
    extract_fst,
)
from src.paths import create_paths
from src.run_alignment import run_alignment
from src.train_rnn import train_rnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="izh")
    parser.add_argument("--data-folder", default="data/inflection")
    parser.add_argument("--models-folder", default="models/izh-full-sweep")
    parser.add_argument("--results-folder", default="results/izh-full-sweep")
    parser.add_argument("--storage", default="sqlite:///results/izh-full-sweep.db")
    parser.add_argument("--rnn-trials", type=int, default=50)
    parser.add_argument("--extraction-trials", type=int, default=100)
    parser.add_argument(
        "--rnn-trials-to-run",
        type=int,
        help="Run exactly this many additional RNN trials (parallel worker mode)",
    )
    parser.add_argument("--stop-after-rnn", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Offset the sampler seed when several worker processes share a study",
    )
    parser.add_argument("--alignment-iterations", type=int, default=100)
    return parser.parse_args()


def finished_trials(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def serializable_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "state": trial.state.name,
        "duration_seconds": (
            trial.duration.total_seconds() if trial.duration is not None else None
        ),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.time()
    results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    Path(args.models_folder).mkdir(parents=True, exist_ok=True)

    paths = create_paths(
        data_folder=args.data_folder,
        dataset=args.dataset,
        has_features=True,
        output_split=False,
        merge_outputs="bpe",
        models_folder=args.models_folder,
    )
    if not paths["train_aligned"].exists() or not paths["eval_aligned"].exists():
        run_alignment(paths, iterations=args.alignment_iterations)

    train_examples, _ = load_examples_from_file(
        paths["train_aligned"], merge_outputs=paths["merge_outputs"]
    )
    max_batch_size = len(train_examples) // 5
    batch_choices = [
        batch
        for batch in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if batch <= max_batch_size
    ][-4:]

    rnn_study = optuna.create_study(
        study_name=f"{args.dataset}-rnn-transduction",
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed + args.worker_id,
            constant_liar=True,
        ),
    )

    def rnn_objective(trial: optuna.Trial) -> float:
        params = {
            "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.3]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [2e-4, 1e-3, 2e-3, 1e-2]
            ),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "epochs": trial.suggest_categorical("epochs", [200, 400, 600]),
        }
        run_name = f"{args.dataset}-optuna-rnn-{trial.number:03d}"
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"
        run = wandb.init(
            project="fst-distillation-local-full-sweep",
            name=run_name,
            mode="offline",
            reinit=True,
            config=params,
        )
        if run is None:
            raise RuntimeError("Failed to initialize the offline W&B run")
        _, validation_loss = train_rnn(
            paths=paths,
            objective="transduction",
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            d_model=params["d_model"],
            num_layers=1,
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            merge_outputs=paths["merge_outputs"],
            activation="tanh",
            spectral_norm_weight=0.1,
            label_smoothing=0.1,
            wandb_run=run,
            seed=args.seed,
        )
        checkpoint = paths["models_folder"] / run_name / "model.pt"
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("checkpoint", str(checkpoint))
        return float(validation_loss)

    remaining_rnn = (
        args.rnn_trials_to_run
        if args.rnn_trials_to_run is not None
        else max(0, args.rnn_trials - finished_trials(rnn_study))
    )
    if remaining_rnn:
        rnn_study.optimize(rnn_objective, n_trials=remaining_rnn, gc_after_trial=True)

    if args.stop_after_rnn:
        print(
            json.dumps(
                {
                    "study": rnn_study.study_name,
                    "completed_trials": finished_trials(rnn_study),
                    "best_value": rnn_study.best_value,
                    "best_params": rnn_study.best_params,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    best_rnn = rnn_study.best_trial
    best_run_name = str(best_rnn.user_attrs["run_name"])
    best_checkpoint = Path(str(best_rnn.user_attrs["checkpoint"]))
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best RNN checkpoint is missing: {best_checkpoint}")

    activation_hparams = ExtractionHyperparameters(
        model_shortname=best_run_name,
        clustering_method="kmeans",
        kmeans_num_clusters=1,
        n_components=None,
        full_domain_mode="sample",
        full_domain_search_n=3,
    )
    activations, transition_labels = compute_activations(activation_hparams, paths)
    max_clusters = int(len(np.unique(activations, axis=0)))

    extraction_study = optuna.create_study(
        study_name=f"{args.dataset}-fst-extraction",
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, constant_liar=True),
    )

    thresholds: list[int | None] = [None, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]

    def extraction_objective(trial: optuna.Trial) -> float:
        classifier = trial.suggest_categorical(
            "state_split_classifier", ["svm", "logistic"]
        )
        threshold = trial.suggest_categorical("minimum_transition_count", thresholds)
        clusters = trial.suggest_int("kmeans_num_clusters", 50, max_clusters)
        hparams = ExtractionHyperparameters(
            model_shortname=best_run_name,
            clustering_method="kmeans",
            use_faiss=True,
            state_split_classifier=classifier,
            n_components=None,
            minimum_transition_count=threshold,
            kmeans_num_clusters=clusters,
            full_domain_mode="sample",
            full_domain_search_n=3,
        )
        metrics, _ = extract_fst(
            hparams,
            paths,
            precomputed_activations=(activations, transition_labels),
        )
        trial.set_user_attr("train_f1", float(metrics["train"]["f1"]))
        trial.set_user_attr("dev_f1", float(metrics["eval"]["f1"]))
        trial.set_user_attr("num_states", int(metrics["num_states"]))
        return float(metrics["eval"]["f1"])

    remaining_extraction = max(
        0, args.extraction_trials - finished_trials(extraction_study)
    )
    if remaining_extraction:
        extraction_study.optimize(
            extraction_objective,
            n_trials=remaining_extraction,
            gc_after_trial=True,
        )

    best_extraction = extraction_study.best_trial
    final_hparams = ExtractionHyperparameters(
        model_shortname=best_run_name,
        clustering_method="kmeans",
        use_faiss=True,
        state_split_classifier=best_extraction.params["state_split_classifier"],
        n_components=None,
        minimum_transition_count=best_extraction.params["minimum_transition_count"],
        kmeans_num_clusters=best_extraction.params["kmeans_num_clusters"],
        full_domain_mode="sample",
        full_domain_search_n=3,
    )
    final_metrics, final_fst = extract_fst(
        final_hparams,
        paths,
        precomputed_activations=(activations, transition_labels),
    )

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))
    final_fst_path = results_folder / f"{args.dataset}-full-sweep-best.fst"
    final_fst.save(str(final_fst_path))
    final_model_path = results_folder / f"{args.dataset}-full-sweep-model.pt"
    shutil.copy2(best_checkpoint, final_model_path)

    summary = {
        "dataset": args.dataset,
        "selection_rule": "RNN by minimum dev loss; FST by maximum dev F1",
        "test_used_for_selection": False,
        "rnn": {
            "requested_trials": args.rnn_trials,
            "completed_trials": finished_trials(rnn_study),
            "best": serializable_trial(best_rnn),
        },
        "extraction": {
            "requested_trials": args.extraction_trials,
            "completed_trials": finished_trials(extraction_study),
            "max_clusters": max_clusters,
            "best": serializable_trial(best_extraction),
        },
        "final_metrics": final_metrics,
        "artifacts": {
            "model": str(final_model_path),
            "fst": str(final_fst_path),
            "study_storage": args.storage,
        },
        "elapsed_seconds_this_invocation": time.time() - started,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "arguments": vars(args),
    }
    summary_path = results_folder / f"{args.dataset}-full-sweep-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
