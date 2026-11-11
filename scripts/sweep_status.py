"""Print compact progress for the resumable local izh sweep."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import optuna


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    storage = f"sqlite:///{(root / 'results' / 'izh-full-sweep-utf8.db').as_posix()}"
    status_path = root / "results" / "izh-full-sweep-utf8-background-status.json"
    summary_path = (
        root / "results" / "izh-full-sweep-utf8" / "izh-full-sweep-summary.json"
    )

    if status_path.exists():
        print("Background process:")
        print(status_path.read_text(encoding="utf-8-sig").strip())

    for study_name in ("izh-rnn-transduction", "izh-fst-extraction"):
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            print(f"{study_name}: not started")
            continue
        counts = Counter(trial.state.name for trial in study.trials)
        completed = counts.get("COMPLETE", 0)
        best = study.best_value if completed else None
        print(f"{study_name}: {dict(counts)}, best={best}")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Final summary is ready:")
        print(json.dumps(summary["final_metrics"], ensure_ascii=False, indent=2))
        print(summary_path)
    else:
        print("Final summary is not ready yet.")


if __name__ == "__main__":
    main()
