# Motivation
FSTs have traditionally been popular for tasks such as morphological inflection, grapheme-to-phoneme, transliteration, and other noisy channel problems. They provide many benefits, including:
- Fast inference on CPUs (no GPUs required)
- Minimal storage requirements
- Interpretability/editability
- Predictable behavior on OOD inputs

With careful design, FSTs can often achieve near-perfect accuracy and compete with SOTA neural systems, as in [Beemer et al., 2020](https://aclanthology.org/2020.sigmorphon-1.18.pdf). However, creating FSTs is a laborious and painstaking process, requiring expert knowledge of both the target domain and finite-state theory. While algorithms for automatic induction of FSTs exist, they are generally very sample inefficient, requiring huge amounts of labeled pairs to converge to an accurate solution.

# Method
This package provides an algorithm for FST induction using a neural surrogate model (specifically, an Elman RNN) and the state-clustering algorithm of Giles et al. (1991).

1. Align the input and output strings using **CRPAlign**.
2. Train an **alignment prediction model** (seq2seq transformer) to predict the aligned input string from the unaligned string. Use this model to predict aligned inputs for the full domain (or a large sample if infinite).
3. Train an **Elman RNN** on language modeling or binary classification.
4. Extract activations from the RNN for both the training data and full domain as predicted in (2).
5. Cluster activations into *macrostates* and aggregate *microtransitions* for each macrostate.
6. Run the *state splitting algorithm* to split any macrostates with non-deterministic outgoing transitions.

# Usage
```shell
# Python >=3.11
pip install -r requirements.txt

gcc -O3 -Wall -Wextra -fPIC -shared src/crpalign/align.c -o src/crpalign/libalign.so

# Run a full extraction on some dataset
python -m src.sweep <data/inflection> <ceb> --objective <lm>
```

- The first parameter `<data/inflection>` is the path to a folder containing the raw data files.
- The second parameter `<ceb>` is the name of the dataset. Files named `ceb.trn`, `ceb.dev`, and `ceb.tst` must exist. The format is the 2020 SIGMORPHON shared task format (see files for an example).
- The RNN training objective may either be `lm` or `classification`


# Experiments

Analysis
- Impact of hidden state size
- Impact of standardization
- Impact of clustering process/n clusters

Ablations
- State splitting
- Full coverage
- Spec norm

Datasets:
- inflection
- g2p
- transliteration
- historical normalization

# Windows/Conda reproduction

This repository includes a local offline runner for reproducing the
morphological-inflection experiment without access to the authors' W&B
project. The tested setup uses Python 3.11 and CPU-only PyTorch on Windows.

```powershell
git clone --recurse-submodules https://github.com/Howepu/fst-distillation-reproduction.git
cd fst-distillation-reproduction

conda env create -f environment.windows.yml
conda run -n fst-distillation x86_64-w64-mingw32-gcc -O3 -Wall -Wextra `
  -shared src/crpalign/align.c -o src/crpalign/libalign.so

powershell -ExecutionPolicy Bypass -File scripts/prepare_inflection_data.ps1

conda run -n fst-distillation python -m scripts.reproduce_local --dataset czn
```

Run the same pipeline for Ingrian (`izh`) with:

```powershell
conda run -n fst-distillation python -m scripts.reproduce_local `
  --dataset izh --run-name local-izh-transduction-seed0
```

The data preparation script copies the train and development splits from the
official SIGMORPHON 2020 submodule and the labeled test splits from its
`GOLD-TEST` directory. Models are written to `models/` and extraction results
to `results/`.

The verified `czn` reproduction summary is available in
`REPRODUCTION_REPORT.md`; machine-readable trial results and the selected
PyFoma transducer are in `results/final/`.
