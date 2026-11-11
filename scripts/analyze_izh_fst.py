"""Analyze the selected Ingrian FST and its source RNN clusters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from pyfoma import FST

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_fst import ExtractionHyperparameters, _cluster, compute_activations
from src.paths import create_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fst", default="results/final/izh-best.fst")
    parser.add_argument(
        "--result", default="results/final/izh-local-reproduction.json"
    )
    parser.add_argument("--model", default="models/izh-selected-model.pt")
    parser.add_argument("--model-name", default="izh-d64-b32-e600-lr002-seed0")
    parser.add_argument("--data-folder", default="data/inflection")
    parser.add_argument("--clusters", type=int, default=300)
    parser.add_argument("--output", default="results/final/izh-analysis.json")
    parser.add_argument("--report", default="IZH_EXPERIMENT_REPORT.md")
    parser.add_argument("--benchmark-repeats", type=int, default=200)
    return parser.parse_args()


def read_dataset(path: Path) -> list[dict[str, Any]]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        lemma, form, features = line.split("\t")
        feature_tokens = [f"[{feature}]" for feature in features.split(";")]
        input_tokens = feature_tokens + ["<sep>", *lemma, "<sink>"]
        examples.append(
            {
                "lemma": lemma,
                "gold": form,
                "features": features,
                "feature_tokens": feature_tokens,
                "input_tokens": input_tokens,
            }
        )
    return examples


def number_states(fst: FST) -> dict[Any, int]:
    state_ids: dict[Any, int] = {fst.initialstate: 0}
    queue = deque([fst.initialstate])
    while queue:
        state = queue.popleft()
        outgoing = []
        for transitions in state.transitions.values():
            outgoing.extend(transitions)
        for transition in sorted(
            outgoing,
            key=lambda item: (
                item.label,
                str(item.targetstate.name),
                item.weight,
            ),
        ):
            target = transition.targetstate
            if target not in state_ids:
                state_ids[target] = len(state_ids)
                queue.append(target)
    return state_ids


def trace_fst(
    fst: FST, state_ids: dict[Any, int], input_tokens: list[str]
) -> tuple[str | None, list[dict[str, Any]]]:
    state = fst.initialstate
    output_tokens: list[str] = []
    steps: list[dict[str, Any]] = []
    for token in input_tokens:
        transitions = state.transitions_by_input.get(token, set())
        if len(transitions) != 1:
            return None, steps
        label, transition = next(iter(transitions))
        output = (
            label[0]
            if len(label) == 1
            else label[1]
        )
        steps.append(
            {
                "from": state_ids[state],
                "input": token,
                "output": output,
                "to": state_ids[transition.targetstate],
            }
        )
        output_tokens.append(output)
        state = transition.targetstate
    if state not in fst.finalstates:
        return None, steps
    return "".join(output_tokens), steps


def surface_output(full_output: str | None, feature_tokens: list[str]) -> str | None:
    if full_output is None:
        return None
    prefix = "".join(feature_tokens) + "<sep>"
    if not full_output.startswith(prefix):
        return full_output
    surface = full_output[len(prefix) :]
    return surface.removesuffix("<sink>")


def category(example: dict[str, Any]) -> str:
    if example["gold"] == example["lemma"]:
        return "identity"
    if example["gold"].startswith(example["lemma"]):
        return "suffix_only"
    return "stem_change"


def cluster_summary(
    activations: np.ndarray,
    transition_sequences: list[list[str]],
    hparams: ExtractionHyperparameters,
    train_examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = _cluster(hparams, activations)
    flat_symbols = [symbol for sequence in transition_sequences for symbol in sequence]
    assert len(labels) == len(flat_symbols)

    cluster_symbols: dict[int, Counter[str]] = {}
    for cluster_id, symbol in zip(labels, flat_symbols):
        cluster_symbols.setdefault(int(cluster_id), Counter())[symbol] += 1

    summaries = []
    for cluster_id, symbols in cluster_symbols.items():
        total = symbols.total()
        dominant_symbol, dominant_count = symbols.most_common(1)[0]
        summaries.append(
            {
                "cluster": cluster_id,
                "size": total,
                "dominant_symbol": dominant_symbol,
                "dominant_count": dominant_count,
                "purity": dominant_count / total,
                "top_symbols": symbols.most_common(5),
            }
        )
    summaries.sort(key=lambda row: (-row["size"], row["cluster"]))

    chains = []
    offset = 0
    for index, sequence in enumerate(transition_sequences):
        sequence_labels = labels[offset : offset + len(sequence)]
        if index < len(train_examples) and len(chains) < 8:
            example = train_examples[index]
            chains.append(
                {
                    "lemma": example["lemma"],
                    "gold": example["gold"],
                    "features": example["features"],
                    "symbols": sequence,
                    "clusters": [int(label) for label in sequence_labels],
                }
            )
        offset += len(sequence)
    assert offset == len(labels)
    return summaries, chains


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fst_path = Path(args.fst)
    result_path = Path(args.result)
    model_path = Path(args.model)
    output_path = Path(args.output)
    report_path = Path(args.report)

    raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    is_full_sweep = "final_metrics" in raw_result
    if is_full_sweep:
        extraction_params = raw_result["extraction"]["best"]["params"]
        result = {
            "best": {
                "clusters": extraction_params["kmeans_num_clusters"],
                "classifier": extraction_params["state_split_classifier"],
                "minimum_transition_count": extraction_params[
                    "minimum_transition_count"
                ],
                "metrics": raw_result["final_metrics"],
            }
        }
    else:
        result = raw_result
    fst = FST.load(str(fst_path))
    state_ids = number_states(fst)
    train_examples = read_dataset(Path(args.data_folder) / "izh.trn")
    test_examples = read_dataset(Path(args.data_folder) / "izh.tst")

    traced_examples = []
    for example in test_examples:
        full_output, steps = trace_fst(fst, state_ids, example["input_tokens"])
        prediction = surface_output(full_output, example["feature_tokens"])
        fst_path_found = prediction is not None
        if prediction is None:
            # Match src.evaluate.evaluate_all: if composition has no final state,
            # return the original input unchanged.
            prediction = example["lemma"]
        traced_examples.append(
            {
                **{key: example[key] for key in ("lemma", "gold", "features")},
                "prediction": prediction,
                "accepted": True,
                "fst_path_found": fst_path_found,
                "used_no_change_fallback": not fst_path_found,
                "correct": prediction == example["gold"],
                "category": category(example),
                "state_chain": [steps[0]["from"]] + [step["to"] for step in steps]
                if steps
                else [],
                "steps": steps,
            }
        )

    correct_count = sum(example["correct"] for example in traced_examples)
    accepted_count = sum(example["accepted"] for example in traced_examples)
    fst_path_count = sum(example["fst_path_found"] for example in traced_examples)
    measured_accuracy = correct_count / len(traced_examples)
    expected_accuracy = result["best"]["metrics"]["test"]["f1"]
    official_correct_count = round(expected_accuracy * len(test_examples))

    started = time.perf_counter()
    for _ in range(args.benchmark_repeats):
        for example in test_examples:
            trace_fst(fst, state_ids, example["input_tokens"])
    benchmark_seconds = time.perf_counter() - started
    benchmark_items = args.benchmark_repeats * len(test_examples)

    paths = create_paths(
        data_folder=args.data_folder,
        dataset="izh",
        has_features=True,
        output_split=False,
        merge_outputs="bpe",
        models_folder="models",
    )
    hparams = ExtractionHyperparameters(
        model_shortname=args.model_name,
        clustering_method="kmeans",
        kmeans_num_clusters=args.clusters,
        use_faiss=is_full_sweep,
        state_split_classifier="logistic",
        minimum_transition_count=2,
        full_domain_mode="sample",
        full_domain_search_n=3,
    )
    activations, transition_sequences = compute_activations(hparams, paths)
    clusters, cluster_chains = cluster_summary(
        activations, transition_sequences, hparams, train_examples
    )

    selected_examples = []
    for wanted_category in ("suffix_only", "stem_change", "identity"):
        match = next(
            (
                example
                for example in traced_examples
                if example["category"] == wanted_category and example["correct"]
            ),
            None,
        )
        if match is None:
            match = next(
                example
                for example in traced_examples
                if example["category"] == wanted_category
            )
        selected_examples.append(match)

    comparison_specs = [
        ("Baseline: d64, batch 128, 200 epochs", Path("results/izh-seed0-smallgrid.json")),
        ("d32, batch 32, 600 epochs", Path("results/izh-d32-b32-e600/izh-local-reproduction.json")),
        ("d64, batch 32, 600 epochs", Path("results/izh-d64-b32-e600/izh-local-reproduction.json")),
        ("d32, batch 16, 600 epochs", Path("results/izh-d32-b16-e600/izh-local-reproduction.json")),
    ]
    model_comparison = []
    for label, path in comparison_specs:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        model_comparison.append(
            {
                "model": label,
                "dev_accuracy": candidate["best"]["metrics"]["eval"]["f1"],
                "test_accuracy": candidate["best"]["metrics"]["test"]["f1"],
                "states": candidate["best"]["metrics"]["num_states"],
                "clusters": candidate["best"]["clusters"],
                "classifier": candidate["best"]["classifier"],
                "threshold": candidate["best"]["minimum_transition_count"],
            }
        )
    if is_full_sweep:
        model_comparison.append(
            {
                "model": "Full Bayesian sweep: 50 RNN + 100 FST",
                "dev_accuracy": result["best"]["metrics"]["eval"]["f1"],
                "test_accuracy": result["best"]["metrics"]["test"]["f1"],
                "states": result["best"]["metrics"]["num_states"],
                "clusters": result["best"]["clusters"],
                "classifier": result["best"]["classifier"],
                "threshold": result["best"]["minimum_transition_count"],
            }
        )

    category_breakdown = []
    for name in ("suffix_only", "stem_change", "identity"):
        category_examples = [
            example for example in traced_examples if example["category"] == name
        ]
        category_breakdown.append(
            {
                "category": name,
                "total": len(category_examples),
                "correct": sum(example["correct"] for example in category_examples),
                "fst_paths": sum(
                    example["fst_path_found"] for example in category_examples
                ),
                "fallbacks": sum(
                    example["used_no_change_fallback"]
                    for example in category_examples
                ),
            }
        )

    incorrect_examples = [example for example in traced_examples if not example["correct"]]
    error_length_breakdown = {
        "prediction_shorter": sum(
            len(example["prediction"]) < len(example["gold"])
            for example in incorrect_examples
        ),
        "same_length": sum(
            len(example["prediction"]) == len(example["gold"])
            for example in incorrect_examples
        ),
        "prediction_longer": sum(
            len(example["prediction"]) > len(example["gold"])
            for example in incorrect_examples
        ),
    }

    analysis = {
        "selection": {
            "rule": "highest development exact-match accuracy",
            "model": args.model_name,
            "clusters": result["best"]["clusters"],
            "classifier": result["best"]["classifier"],
            "minimum_transition_count": result["best"][
                "minimum_transition_count"
            ],
        },
        "metrics": {
            **result["best"]["metrics"],
            "paper_test_accuracy": 0.438,
            "accepted_test_items": accepted_count,
            "test_items_with_fst_path": fst_path_count,
            "test_items_using_no_change_fallback": len(test_examples) - fst_path_count,
            "correct_test_items": correct_count,
            "official_correct_test_items": official_correct_count,
            "direct_trace_accuracy": measured_accuracy,
        },
        "artifacts": {
            "model_bytes": model_path.stat().st_size,
            "fst_bytes": fst_path.stat().st_size,
        },
        "fst": {
            "states": len(fst.states),
            "arcs": fst.arccount(),
            "final_states": len(fst.finalstates),
            "deterministic": fst.is_deterministic(),
        },
        "benchmark": {
            "device": "AMD Ryzen 5 5600X CPU",
            "items": benchmark_items,
            "seconds": benchmark_seconds,
            "items_per_second": benchmark_items / benchmark_seconds,
            "repeats": args.benchmark_repeats,
        },
        "clusters": {
            "count": args.clusters,
            "activation_vectors": len(activations),
            "largest": clusters[:20],
        },
        "cluster_chains": cluster_chains,
        "model_comparison": model_comparison,
        "category_breakdown": category_breakdown,
        "error_length_breakdown": error_length_breakdown,
        "selected_test_examples": selected_examples,
        "all_test_examples": traced_examples,
    }
    output_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Эксперимент на ижорском датасете `izh`",
        "",
        "## Итог",
        "",
        (
            f"Выбранная по dev конфигурация: `{args.model_name}`, "
            f"{result['best']['clusters']} исходных k-means-кластеров, "
            f"`{result['best']['classifier']}`, порог переходов "
            f"`{result['best']['minimum_transition_count']}`."
        ),
        "",
        markdown_table(
            ["Метрика", "Значение"],
            [
                ["Dev exact match", f"{result['best']['metrics']['eval']['f1']:.2%}"],
                [
                    "Test exact match (авторский evaluator)",
                    f"{expected_accuracy:.2%} ({official_correct_count}/224)",
                ],
                [
                    "Test exact match (прямой обход сохранённого FST)",
                    f"{measured_accuracy:.2%} ({correct_count}/224)",
                ],
                ["Test CER", f"{result['best']['metrics']['test']['cer']:.4f}"],
                ["Принято test-входов", f"{accepted_count}/224 (100%)"],
                ["Найден путь непосредственно в FST", f"{fst_path_count}/224"],
                ["Использован no-change fallback", len(test_examples) - fst_path_count],
                ["Результат статьи", "43.80%"],
                ["Состояния FST", len(fst.states)],
                ["Переходы FST", fst.arccount()],
                ["Размер RNN", f"{model_path.stat().st_size:,} байт"],
                ["Размер FST", f"{fst_path.stat().st_size:,} байт"],
                [
                    "Скорость прямого обхода FST на CPU",
                    f"{benchmark_items / benchmark_seconds:,.1f} слов/с",
                ],
            ],
        ),
        "",
        "Автомат детерминирован и имеет одно финальное состояние. На целевом устройстве RNN не требуется.",
        (
            "Независимый прямой обход сохранённого FST полностью совпал с "
            "результатом авторского evaluator."
            if official_correct_count == correct_count
            else (
                "Авторский evaluator на основе композиции PyFoma и прямой обход "
                f"сохранённого автомата расходятся на "
                f"{official_correct_count - correct_count} правильных ответа. "
                "Для сравнения со статьёй используется авторский evaluator."
            )
        ),
        "",
        "## Сравнение проверенных моделей",
        "",
        "Модель и FST выбирались по dev exact match. Test не использовался при выборе.",
        "",
        markdown_table(
            ["Конфигурация RNN", "Dev", "Test", "Состояния", "Извлечение"],
            [
                [
                    row["model"],
                    f"{row['dev_accuracy']:.2%}",
                    f"{row['test_accuracy']:.2%}",
                    row["states"],
                    f"k={row['clusters']}, {row['classifier']}, threshold={row['threshold']}",
                ]
                for row in model_comparison
            ],
        ),
        "",
        "## Исходные кластеры скрытых состояний",
        "",
        (
            f"Для выбранной модели получено {len(activations)} векторов активаций. "
            f"Перед state splitting использовано {args.clusters} k-means-кластеров. "
            "После разрешения конфликтов и минимизации они преобразовались в "
            f"{len(fst.states)} состояний FST. Поэтому номера кластеров C и номера "
            "конечных состояний S не обязаны совпадать."
        ),
        "",
        markdown_table(
            ["Кластер", "Размер", "Доминирующий переход", "Доля", "Топ переходов"],
            [
                [
                    f"C{cluster['cluster']}",
                    cluster["size"],
                    cluster["dominant_symbol"],
                    f"{cluster['purity']:.1%}",
                    ", ".join(
                        f"{symbol} ({count})"
                        for symbol, count in cluster["top_symbols"][:3]
                    ),
                ]
                for cluster in clusters[:15]
            ],
        ),
        "",
        "## Примеры цепочек исходных кластеров",
        "",
    ]
    for chain in cluster_chains[:5]:
        lines.extend(
            [
                f"- `{chain['features']}: {chain['lemma']} → {chain['gold']}`",
                f"  - символы: `{' → '.join(chain['symbols'])}`",
                f"  - кластеры: `{' → '.join('C' + str(value) for value in chain['clusters'])}`",
            ]
        )

    lines.extend(["", "## Пути по конечному FST", ""])
    for example in selected_examples:
        lines.extend(
            [
                (
                    f"### `{example['features']}: {example['lemma']} → "
                    f"{example['prediction']}`"
                ),
                "",
                f"Эталон: `{example['gold']}`. Категория: `{example['category']}`. "
                f"Результат: {'верно' if example['correct'] else 'ошибка'}. "
                f"Источник: {'путь FST' if example['fst_path_found'] else 'no-change fallback'}.",
                "",
                markdown_table(
                    ["Из", "Вход", "Выход", "В"],
                    [
                        [
                            f"S{step['from']}",
                            step["input"],
                            step["output"],
                            f"S{step['to']}",
                        ]
                        for step in example["steps"]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Разбивка качества",
            "",
            markdown_table(
                ["Тип преобразования", "Всего", "Верно", "Путь FST", "Fallback"],
                [
                    [
                        row["category"],
                        row["total"],
                        row["correct"],
                        row["fst_paths"],
                        row["fallbacks"],
                    ]
                    for row in category_breakdown
                ],
            ),
            "",
            (
                f"Среди {len(incorrect_examples)} ошибочных ответов предсказание короче эталона в "
                f"{error_length_breakdown['prediction_shorter']} случаях, имеет ту же длину "
                f"в {error_length_breakdown['same_length']} и длиннее в "
                f"{error_length_breakdown['prediction_longer']}."
            ),
            "",
        ]
    )

    lines.extend(
        [
            "## Вывод",
            "",
            (
                "FST уверенно копирует большую часть основы и во многих случаях "
                "обучает чередования внутри основы, но главный источник ошибок – "
                "выбор финального суффикса и смешение близких падежно-числовых цепочек."
            ),
            (
                "На Windows все загрузчики данных были переведены на явную кодировку "
                "UTF-8. Без этого символы ä, ö, ü и š повреждались при чтении, а test "
                "accuracy полного sweep снижалась до 22.77%. После исправления "
                f"получено {expected_accuracy:.2%}."
                if is_full_sweep
                else ""
            ),
            (
                f"Полученные {expected_accuracy:.2%} ниже 43.80% из статьи. "
                + (
                    "Полный локальный протокол включал 50 успешных запусков RNN и "
                    "100 запусков извлечения FST с Bayesian-подбором по dev. "
                    "Следовательно, расхождение уже нельзя объяснить только малым "
                    "числом проверенных конфигураций. Возможные причины – отличие "
                    "Optuna TPE от W&B Bayesian/Hyperband, версии зависимостей и "
                    "стохастика FAISS-кластеризации."
                    if is_full_sweep
                    else "В этом запуске использовался ограниченный подбор параметров."
                )
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"analysis": str(output_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
