# xLSTM from Scratch: Benchmarking Against LSTM and Transformer

This project is a **graduate thesis** investigating the performance of a **from-scratch implementation of xLSTM** (extreme LSTM) compared to the official xLSTM library, classic LSTM, and Transformer models.  
The experiments cover both **language modeling** (on WikiText-2) and a synthetic **nearest neighbor search (NNS)** sequence task.

---

## Project Structure



.
├── main.ipynb                # Main notebook: run all benchmarks & plots
├── xlstm\_replica.py          # From-scratch implementation of xLSTM (faithful to paper)
├── experiments/
│   ├── lm\_benchmark.py       # Language modeling benchmarks (WikiText-2)
│   └── nns\_benchmark.py      # Nearest neighbor search (NNS) benchmarks
├── utils/
│   ├── data\_preparation.py   # Dataset loaders & synthetic data generation
│   ├── model\_architectures.py# Baseline model wrappers: LSTM, Transformer, etc.
│   ├── plotting.py           # Plotting utilities for experiment results
│   └── training\_loops.py     # Training, evaluation, and scheduler logic
├── lm\_benchmark\_xlstm\_vs\_lib\_results.json        # Pre-saved LM benchmark results
├── lm\_benchmark\_baselines\_vs\_xlstm\_results.json  # Pre-saved LM baseline results
└── ...



---

## How to Run

**All experiments and plots can be reproduced by running:**

```python
main.ipynb
````

This notebook executes the following experiments:

* **Language Modeling:**

  * From-scratch xLSTM vs. official xLSTM library
  * xLSTM vs. LSTM vs. Transformer
* **Nearest Neighbor Search (NNS):**

  * From-scratch xLSTM vs. official xLSTM library
  * xLSTM vs. LSTM vs. Transformer

The notebook saves and displays comparison plots for each benchmark.

---

## Results Summary

### Language Modeling (WikiText-2) — Test Set Perplexity (PPL)

| Model             | Rare (<100) | Medium (100-1000) | Frequent (>1000) | Overall PPL |
| ----------------- | :---------: | :---------------: | :--------------: | :---------: |
| **LSTM**          |    205115   |       8215.5      |       104.6      |    2031.2   |
| **Library xLSTM** |    57969    |       706.3       |       17.0       |    327.2    |
| **Scratch xLSTM** |    51352    |       614.4       |       17.8       |    313.4    |
| **Transformer**   |    56597    |       656.5       |       18.3       |    331.5    |

* **Observation:**
  Both the scratch xLSTM and library xLSTM significantly outperform LSTM and match Transformer on frequent/medium words, with much lower PPL overall.

---

## Implementation Highlights

* **Faithful, modular xLSTM replica** with careful numerical stabilization.
* **Unified benchmarking pipeline** for fair comparison with LSTM, Transformer, and official xLSTM.
* **Extensible codebase** — add more experiments by editing `main.ipynb` or scripts in `experiments/`.

---

## Citation / Credits

* xLSTM: [Official Paper & Library](https://github.com/NVIDIA/transformer-lstm)
* Project by: *Mohamed Ali*, Graduate Thesis, JKU, 2025

---
