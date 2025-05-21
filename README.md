# xLSTM from Scratch: Benchmarking Against LSTM and Transformer

This project is a **graduate thesis** investigating the performance of a **from-scratch implementation of xLSTM**  compared to the official xLSTM library, classic LSTM, and Transformer models.  
The experiments cover both **language modeling** (on WikiText-2) and a synthetic **nearest neighbor search (NNS)** sequence task.

---
## 🛠️ Setting Up the Environment

Run the following command from the project root directory to create the environment:

```bash
conda env create -f environment.yml
```

Once the environment is created, activate it with:

```bash
conda activate xlstm2
```

You're now ready to run the project — mainly through `main.ipynb`.

---


## Project Structure



```

.
├── main.ipynb                  # Main notebook to run all benchmarks and view results
├── xlstm_replica.py            # From-scratch implementation of xLSTM components
├── experiments
│   ├── lm_benchmark.py         # Logic for Language Modeling benchmarks
│   └── nns_benchmark.py        # Logic for Nearest Neighbor Search benchmarks
├── utils
│   ├── data_preparation.py     # Data loading and preprocessing utilities
│   ├── model_architectures.py  # Baseline models (LSTM, Transformer) and wrappers
│   ├── plotting.py             # Utilities for generating result plots
│   └── training_loops.py       # Training and evaluation loop implementations
├── lm_benchmark_xlstm_vs_lib_results.json      # LM benchmark results (Scratch vs. Library xLSTM)
└── lm_benchmark_baselines_vs_xlstm_results.json # LM benchmark results (Baselines vs. xLSTM)

```



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
