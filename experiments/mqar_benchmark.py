# ==============================================================================
# MQAR (Zoology-style) Benchmark — Full Replacement with Result Saving + Plots
# ==============================================================================

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union
import math, random, time, os, json, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- your models & plotting ----------------------------------------------------
from utils.model_architectures import GeneralHybridLM, GeneralHybridConfig
from utils.plotting import plot_mqar_heatmap, plot_mqar_capacity_panels

# -------------------------
# Global defaults
# -------------------------
SEED        = 1337
VOCAB_SIZE  = 8192      # |C|
SEQ_LEN     = 128       # N  (sweep can overwrite)
KV_PAIRS    = 32        # D
EMBED_DIM   = 128       # d_model
N_HEADS     = 1         # Zoology synthetic runs use a single head
DROPOUT     = 0.0       # no dropout in Appendix E/E.2

EPOCHS      = 35
#LR_GRID     = (1e-4, 3e-4, 1e-3, 3e-3)  # four LRs between 1e-4 and 1e-2
LR_GRID     = (1e-2,)  
WEIGHT_DECAY= 0.1
WARMUP_FRAC = 0.10

EARLY_STOP      = True       # set True to enable
ES_PATIENCE     = 7          # stop if no improvement for this many epochs
ES_MIN_EPOCHS   = 5          # don't stop before this many epochs
ES_MIN_DELTA    = 1e-4       # required improvement over best so far
ES_RESTORE_BEST = True       # load best weights before returning
MAX_MINUTES     = None       # e.g., 15.0 to stop after ~15 minutes
# -------------------------
# Utilities
# -------------------------
def _set_all_seeds(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _batch_size_by_rule(n: int, d: int) -> int:
    # Zoology Appendix E/E.2 policy:
    # 64 otherwise; 16 if N>=256 or d>=256; 8 if N>=512 or d>=512
    if n >= 512 or d >= 512: return 8
    if n >= 256 or d >= 256: return 16
    return 64

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)

def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _print_run_header(header: Dict):
    # Nicely print a consistent header before each model
    print("\n" + "="*78)
    print(f"[Run] {header.get('model_name','(unknown)')}")
    for k in [
        "benchmark_type","device","seq_len","kv_pairs","embed_dim","n_heads","epochs",
        "batch_size","train_size","val_size","weight_decay","warmup_frac"
    ]:
        print(f"  {k:>13}: {header.get(k)}")
    print(f"  lr_grid    : {header.get('lr_grid')}")
    print("="*78)

# -------------------------
# MQAR (Zoology) data generation — Appendix E Algorithm 1
# -------------------------
@dataclass
class MQARZoologyConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int    = SEQ_LEN
    kv_pairs: int   = KV_PAIRS
    alpha: float    = 0.1     # power-law parameter (Appendix E)
    filler_id: int  = 0       # fixed filler id; reserve keys from 1..V/2-1 to avoid collision
    train_examples: int = 100_000
    val_examples:   int = 3_000

def _power_law_choices(low:int, high:int, alpha:float, num:int,
                       avoid:set=None, two_slots:bool=True) -> List[int]:
    assert low <= high, "Invalid range for power-law sampling"
    idxs = np.arange(low, high + 1, dtype=np.int64)
    if two_slots and len(idxs) and idxs[-1] == high:
        idxs = idxs[idxs <= high - 1]   # avoid overflow for p+1

    ranks = np.arange(1, len(idxs)+1, dtype=np.float64)
    probs = ranks ** (-alpha); s = probs.sum(); probs = probs / s if s > 0 else np.ones_like(probs)/len(probs)

    chosen = []
    occupied = set(avoid) if avoid else set()
    tries_per = 20
    for _ in range(num):
        p = None
        for _t in range(tries_per):
            cand = int(np.random.choice(idxs, p=probs))
            if two_slots:
                if cand not in occupied and (cand+1) not in occupied:
                    p = cand; break
            else:
                if cand not in occupied:
                    p = cand; break
        if p is None:
            # linear scan fallback
            for cand in idxs:
                if two_slots:
                    if cand not in occupied and (cand+1) not in occupied:
                        p = int(cand); break
                else:
                    if cand not in occupied:
                        p = int(cand); break
        if p is None:
            p = int(low)
        chosen.append(p)
        occupied.add(p)
        if two_slots: occupied.add(p+1)
    return chosen

def _generate_one_sequence(cfg: MQARZoologyConfig,
                           key2val: np.ndarray,
                           rng: random.Random) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, D, V = cfg.seq_len, cfg.kv_pairs, cfg.vocab_size
    seq   = np.full(N, fill_value=cfg.filler_id, dtype=np.int64)
    qmask = np.zeros(N, dtype=np.int8)

    # Keys in [1 .. V/2-1] to avoid filler_id=0; values in [V/2 .. V-1]
    keys = rng.sample(range(1, V//2), D)
    vals = [int(key2val[k]) for k in keys]

    # place D pairs at the start: [k1,v1,k2,v2,...]
    pos = 0
    for k, v in zip(keys, vals):
        seq[pos] = k; seq[pos+1] = v; pos += 2

    # second occurrences: choose start in [2D .. N-2] with power-law spacing
    low = 2*D; high = N-1
    avoid = set(range(0, 2*D))
    starts = _power_law_choices(low, high, alpha=0.1, num=D, avoid=avoid, two_slots=True)

    order = list(range(D))
    rng.shuffle(order)
    for i, idx in enumerate(order):
        k, v = keys[idx], vals[idx]
        p = starts[i]
        # ensure free; if occupied, scan forward then backward
        if seq[p] != cfg.filler_id or (p+1 < N and seq[p+1] != cfg.filler_id):
            placed = False
            for q in range(2*D, N-1):
                if seq[q] == cfg.filler_id and seq[q+1] == cfg.filler_id:
                    p = q; placed = True; break
            if not placed:
                for q in range(N-2, 2*D-1, -1):
                    if seq[q] == cfg.filler_id and seq[q+1] == cfg.filler_id:
                        p = q; break
        seq[p] = k
        if p+1 < N: seq[p+1] = v
        qmask[p] = 1

    labels = np.full(N, fill_value=-100, dtype=np.int64)
    labels[:-1] = seq[1:]
    return seq, qmask, labels

def _make_datasets(cfg: MQARZoologyConfig) -> Tuple[TensorDataset, TensorDataset]:
    V, N, D = cfg.vocab_size, cfg.seq_len, cfg.kv_pairs

    # Global bijection from keys→values (values live in top half)
    key2val = np.zeros(V//2, dtype=np.int64)
    vals_pool = list(range(V//2, V))
    rng = random.Random(SEED)
    rng.shuffle(vals_pool)
    for k in range(1, V//2):
        key2val[k] = vals_pool[(k-1) % (V - V//2)]

    def build(num: int, seed_offset: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = np.zeros((num, N), dtype=np.int64)
        qms= np.zeros((num, N), dtype=np.int8)
        ys = np.zeros((num, N), dtype=np.int64)
        local_rng = random.Random(SEED + seed_offset)
        for i in range(num):
            s, qm, lab = _generate_one_sequence(cfg, key2val, local_rng)
            xs[i], qms[i], ys[i] = s, qm, lab
        return torch.from_numpy(xs), torch.from_numpy(qms), torch.from_numpy(ys)

    x_tr, qm_tr, y_tr = build(cfg.train_examples, 12345)
    x_va, qm_va, y_va = build(cfg.val_examples,   54321)
    return TensorDataset(x_tr, qm_tr, y_tr), TensorDataset(x_va, qm_va, y_va)
#import glob

# def _collect_existing_mqar_results(
#     results_dir: str,
#     *,
#     benchmark_type: str,
#     seq_len: int,
#     kv_pairs: int,
#     embed_dim: int,
#     train_size: int,
#     val_size: int
# ) -> Dict[str, List[float]]:
#     """
#     Returns {model_name: [val_acc, ...]} for runs that exactly match the config.
#     We read the per-model JSONs saved by run_mqar_benchmark(...).
#     """
#     existing: Dict[str, List[float]] = {}
#     for path in glob.glob(os.path.join(results_dir, "*.json")):
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 j = json.load(f)
#         except Exception:
#             continue
#         # only consider per-model receipts with 'run' and 'best'
#         run = j.get("run", {})
#         best = j.get("best", {})
#         if not run or "val_acc" not in best:
#             continue
#         if j.get("benchmark_type") != benchmark_type:
#             continue
#         # strict match on config (so we don’t mix apples/oranges)
#         if (
#             run.get("seq_len") == seq_len and
#             run.get("kv_pairs") == kv_pairs and
#             run.get("embed_dim") == embed_dim and
#             run.get("train_size") == train_size and
#             run.get("val_size") == val_size
#         ):
#             name = j.get("model_name", "")
#             acc  = best.get("val_acc", None)
#             if isinstance(acc, (int, float)) and math.isfinite(acc):
#                 existing.setdefault(name, []).append(float(acc))
#     return existing


# def resume_sweep_mqar_capacity(
#     device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
#     *,
#     benchmark_type: str = "xlstm_ablation",
#     kv_settings: Tuple[int, ...] = (16, 32),
#     width_settings: Tuple[int, ...] = (64, 128, 256),
#     seq_len: int = 128,
#     train_size: int = 20_000,
#     val_size: int = 4_000,
#     repetitions: int = 5
# ) -> Dict[int, Dict[str, Dict[int, List[float]]]]:
#     """
#     Exactly like sweep_mqar_capacity(...), but:
#       • Looks in results/<benchmark_type>/ for matching JSONs you already saved
#       • Skips combinations (kv, width, model) that already have >= repetitions runs
#       • Runs only the missing reps
#       • Rebuilds plots and a fresh summary JSON
#     """
#     global SEQ_LEN, EMBED_DIM
#     _set_all_seeds(SEED)
#     device = torch.device(device_str)

#     out_dir = _ensure_dir(os.path.join("results", _safe_name(benchmark_type)))
#     dataset_cache: dict[Tuple[int, int, int, int], Tuple[TensorDataset, TensorDataset]] = {}
#     results_by_kv: Dict[int, Dict[str, Dict[int, List[float]]]] = {}

#     for kv in kv_settings:
#         cache_key = (seq_len, kv, train_size, val_size)
#         if cache_key not in dataset_cache:
#             cfg = MQARZoologyConfig(
#                 vocab_size=VOCAB_SIZE, seq_len=seq_len, kv_pairs=kv,
#                 train_examples=train_size, val_examples=val_size
#             )
#             dataset_cache[cache_key] = _make_datasets(cfg)
#         prebuilt_ds = dataset_cache[cache_key]

#         for width in width_settings:
#             SEQ_LEN = seq_len
#             EMBED_DIM = width
#             bs = _batch_size_by_rule(seq_len, width)

#             # Build the model dict once, just to know expected names
#             models = _build_models(benchmark_type, VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)
#             model_names = list(models.keys())

#             # Load what you already have on disk for THIS (kv,width,seq_len,train,val,embed_dim)
#             existing = _collect_existing_mqar_results(
#                 out_dir,
#                 benchmark_type=benchmark_type,
#                 seq_len=seq_len, kv_pairs=kv, embed_dim=width,
#                 train_size=train_size, val_size=val_size
#             )

#             # Seed results_by_kv with the existing lists
#             for m in model_names:
#                 already = existing.get(m, [])
#                 results_by_kv.setdefault(kv, {}).setdefault(m, {}).setdefault(width, []).extend(already)

#             # How many more do we need per model?
#             for m in model_names:
#                 have = len(results_by_kv[kv][m][width])
#                 need = max(0, repetitions - have)
#                 if need == 0:
#                     print(f"[resume] KV={kv} d={width} {m}: have {have}/{repetitions} → skipping")
#                     continue

#                 print(f"[resume] KV={kv} d={width} {m}: have {have}/{repetitions} → running {need} more")
#                 # Run the missing reps one by one (keeping dataset/BS identical)
#                 for rep_idx in range(have + 1, have + need + 1):
#                     run_out = run_mqar_benchmark(
#                         device_str=device_str,
#                         benchmark_type=benchmark_type,
#                         kv_pairs=kv,
#                         train_size=train_size,
#                         val_size=val_size,
#                         prebuilt_ds=prebuilt_ds,
#                         batch_size_override=bs,
#                         results_dir=out_dir,
#                         rep_index=rep_idx,
#                         rep_total=repetitions
#                     )
#                     # Collect this model’s new number
#                     if m in run_out:
#                         results_by_kv[kv][m][width].append(run_out[m])

#     # ── plots (same behavior as sweep_mqar_capacity) ─────────────────────────
#     if benchmark_type.startswith("xlstm_ablation"):
#         _plot_ablation_triads(results_by_kv, out_dir=out_dir, context_len=seq_len)  # uses your existing helper
#     else:
#         panels_path = os.path.join(out_dir, f"capacity_panels_N{seq_len}_train{train_size}_val{val_size}.png")
#         plot_mqar_capacity_panels(results_by_kv, context_len=seq_len, save_path=panels_path)
#         print(f"Capacity panels saved → {panels_path}")

#     # Heatmap (best width per model at largest KV), same as your original
#     try:
#         kv_for_heatmap = max(results_by_kv.keys())
#         flat_for_heatmap: Dict[str, float] = {}
#         for model_name, width_dict in results_by_kv[kv_for_heatmap].items():
#             best_w, best_mean = None, float("-inf")
#             for w, accs in width_dict.items():
#                 m = float(np.mean(accs)) if accs else float("-inf")
#                 if not math.isfinite(m): m = float("-inf")
#                 if m >= best_mean: best_mean, best_w = m, w
#             label = f"{model_name} [d={best_w}]" if best_w is not None else model_name
#             flat_for_heatmap[label] = 0.0 if best_mean == float("-inf") else best_mean
#         heatmap_path = os.path.join(out_dir, f"mqar_heatmap_N{seq_len}_kv{kv_for_heatmap}.png")
#         plot_mqar_heatmap(flat_for_heatmap, filename=heatmap_path)
#         print(f"Heatmap saved → {heatmap_path}")
#     except Exception as e:
#         print("Heatmap skipped due to error:", e)

#     # Save a refreshed aggregate summary
#     summary_path = os.path.join(out_dir, f"summary_N{seq_len}_train{train_size}_val{val_size}.json")
#     _save_json(summary_path, results_by_kv)
#     print(f"Summary JSON saved → {summary_path}")

#     return results_by_kv

import glob

def combine_saved_results_for_plot(
    benchmark_types: List[str],
    *,
    kv_settings: Tuple[int, ...],
    width_settings: Tuple[int, ...],
    seq_len: int,
    train_size: int,
    val_size: int,
    save_dir: str = "results/combined_ablation",
    heatmap: bool = True
) -> Dict[int, Dict[str, Dict[int, List[float]]]]:
    """
    Scan multiple results/<benchmark_type>/ folders, load all per-model JSON receipts,
    filter by (seq_len, kv_pairs, embed_dim, train_size, val_size), and aggregate into
    results_by_kv = { kv : { model_name : { width : [acc,...] } } }. Then plot the
    capacity panels (and a heatmap at the largest KV).
    """
    _ensure_dir(save_dir)
    results_by_kv: Dict[int, Dict[str, Dict[int, List[float]]]] = {}

    for bt in benchmark_types:
        in_dir = os.path.join("results", _safe_name(bt))
        for path in glob.glob(os.path.join(in_dir, "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                continue
            run, best = j.get("run", {}), j.get("best", {})
            if not run or "val_acc" not in best:
                continue

            if (run.get("seq_len") != seq_len or
                run.get("train_size") != train_size or
                run.get("val_size")   != val_size):
                continue

            kv = run.get("kv_pairs")
            d  = run.get("embed_dim")
            if kv not in kv_settings or d not in width_settings:
                continue

            name = j.get("model_name", "")
            acc  = best.get("val_acc", None)
            if not (isinstance(acc, (int, float)) and math.isfinite(acc)):
                continue

            results_by_kv \
                .setdefault(kv, {}) \
                .setdefault(name, {}) \
                .setdefault(d, []) \
                .append(float(acc))

    # ---- Plot capacity panels
    panels_path = os.path.join(save_dir, f"capacity_panels_N{seq_len}_train{train_size}_val{val_size}.png")
    plot_mqar_capacity_panels(results_by_kv, context_len=seq_len, save_path=panels_path)
    print(f"Capacity panels saved → {panels_path}")

    # ---- Optional: heatmap (pick largest KV)
    if heatmap and results_by_kv:
        kv_for_heatmap = max(results_by_kv.keys())
        flat_for_heatmap: Dict[str, float] = {}
        for model_name, width_dict in results_by_kv[kv_for_heatmap].items():
            best_w, best_mean = None, float("-inf")
            for w, accs in width_dict.items():
                m = float(np.mean(accs)) if accs else float("-inf")
                if not math.isfinite(m): m = float("-inf")
                if m >= best_mean: best_mean, best_w = m, w
            label = f"{model_name} [d={best_w}]" if best_w is not None else model_name
            flat_for_heatmap[label] = 0.0 if best_mean == float("-inf") else best_mean
        heatmap_path = os.path.join(save_dir, f"mqar_heatmap_N{seq_len}_kv{kv_for_heatmap}.png")
        plot_mqar_heatmap(flat_for_heatmap, filename=heatmap_path)
        print(f"Heatmap saved → {heatmap_path}")

    return results_by_kv

# -------------------------
# Model building per benchmark_type
# -------------------------
def _xlstm_ablation_variants(base_bs: str) -> List[Tuple[str, str]]:
    """
    For a given base in { 'ss','sm','ms','mm' } build the ablation set:
      - original
      - s → l
      - m → t           # <-- NEW: gives ST for SM and TS for MS
      - s → l and m → t
    Example: base='sm' → [('sm','orig'), ('lm','s→l'), ('st','m→t'), ('lt','s→l,m→t')]
    """
    base = base_bs.lower()
    variants: List[Tuple[str, str]] = []

    # 1) original
    variants.append((base, "orig"))

    # 2) s → l
    s2l = base.replace("s", "l")
    variants.append((s2l, "s→l"))

    # 3) m → t   (NEW)
    m2t = base.replace("m", "t")
    variants.append((m2t, "m→t"))

    # 4) s → l AND m → t
    both = s2l.replace("m", "t")
    variants.append((both, "s→l,m→t"))

    # de-duplicate while preserving order
    seen = set()
    unique: List[Tuple[str, str]] = []
    for bs, tag in variants:
        if bs not in seen:
            seen.add(bs)
            unique.append((bs, tag))
    return unique

def _build_models(benchmark_type: str,
                  vocab_size: int,
                  d_model: int,
                  seq_len: int,
                  n_heads: int,
                  num_layers: int = 1) -> Dict[str, nn.Module]:
    """
    Build a dictionary of models keyed by a readable name.
    """
    models: Dict[str, nn.Module] = {}

    def make(block_string: str, label: Optional[str] = None) -> Tuple[str, nn.Module]:
        cfg = GeneralHybridConfig(
            vocab_size     = vocab_size,
            embed_dim      = d_model,
            hidden_dim     = d_model,
            block_string   = block_string,
            context_length = seq_len,
            n_heads        = n_heads,
            dropout        = DROPOUT,
            pad_idx        = 0,
            num_layers     = 1,      # one encoder layer per block symbol
        )
        name = label or f"General-{block_string.upper()} (MQAR)"
        return name, GeneralHybridLM(cfg)

    if benchmark_type in ("tt_only", "zoology_tt"):
        k, m = make("tt", "General-TT (MQAR)")
        models[k] = m

    elif benchmark_type == "mm_vs_tt":
        for bs in ("mm", "tt"):
            k, m = make(bs, f"General-{bs.upper()} (MQAR)")
            models[k] = m

    elif benchmark_type == "tl_two_block_combinations":
        for a in "tl":
            for b in "tl":
                bs = a + b
                k, m = make(bs, f"General-{bs.upper()} (MQAR)")
                models[k] = m

    elif benchmark_type == "xlstm_two_block_combinations":
        for a in "sm":
            for b in "sm":
                bs = a + b
                k, m = make(bs, f"General-{bs.upper()} (MQAR)")
                models[k] = m

    elif benchmark_type == "all_two_block_combos":
        for a in "tlms":
            for b in "tlms":
                bs = a + b
                k, m = make(bs, f"General-{bs.upper()} (MQAR)")
                models[k] = m

    elif benchmark_type == "the benchmark name":
        # placeholder for your custom tests
        models = {}
    elif benchmark_type == "xlstm_ablation":
        # triads for each base in {ss, sm, ms, mm}
        for base in ("ss", "sm", "ms", "mm"):
            for bs, tag in _xlstm_ablation_variants(base):
                label = f"Ablation[{base.upper()}] {bs.upper()} ({tag})"
                k, m = make(bs, f"{label} (MQAR)")
                models[k] = m

    elif benchmark_type == "xlstm_ablation_sm":
        for bs, tag in _xlstm_ablation_variants("sm"):
            label = f"Ablation[SM] {bs.upper()} ({tag})"
            k, m = make(bs, f"{label} (MQAR)")
            models[k] = m

    elif benchmark_type == "xlstm_ablation_ms":
        for bs, tag in _xlstm_ablation_variants("ms"):
            label = f"Ablation[MS] {bs.upper()} ({tag})"
            k, m = make(bs, f"{label} (MQAR)")
            models[k] = m

    elif benchmark_type == "xlstm_ablation_ss":
        for bs, tag in _xlstm_ablation_variants("ss"):
            label = f"Ablation[SS] {bs.upper()} ({tag})"
            k, m = make(bs, f"{label} (MQAR)")
            models[k] = m

    elif benchmark_type == "xlstm_ablation_mm":
        for bs, tag in _xlstm_ablation_variants("mm"):
            label = f"Ablation[MM] {bs.upper()} ({tag})"
            k, m = make(bs, f"{label} (MQAR)")
            models[k] = m
    elif benchmark_type in ("xlstm_ablation_st_ts_only", "st_ts_only"):
        for bs, tag, base in (("st", "m→t", "SM"), ("ts", "m→t", "MS")):
            label = f"Ablation[{base}] {bs.upper()} ({tag})"
            k, m = make(bs, f"{label} (MQAR)")
            models[k] = m
    else:
        k, m = make("tt", "General-TT (MQAR)")
        models[k] = m

    return models

# -------------------------
# Evaluation on MQAR: accuracy on second-key positions only
# -------------------------
@torch.no_grad()
def _evaluate_mqar(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device) -> float:
    model.eval()
    tot, correct = 0, 0
    for xb, qmask, yb in loader:
        xb, qmask, yb = xb.to(device), qmask.to(device).bool(), yb.to(device)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1)
        mask = qmask & (yb != -100)
        tot += mask.sum().item()
        if mask.any():
            correct += (preds[mask] == yb[mask]).sum().item()
    return correct / max(1, tot)

# -------------------------
# Training driver (single model, LR sweep)
# -------------------------
def _train_one_model(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     device: torch.device,
                     base_lr: float,
                     epochs: int = EPOCHS,
                     weight_decay: float = WEIGHT_DECAY,
                     warmup_frac: float = WARMUP_FRAC,
                     *,
                     early_stop: bool = EARLY_STOP,
                     patience: int = ES_PATIENCE,
                     min_delta: float = ES_MIN_DELTA,
                     min_epochs: int = ES_MIN_EPOCHS,
                     restore_best: bool = ES_RESTORE_BEST,
                     max_minutes: float | None = MAX_MINUTES) -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=weight_decay, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def set_lr(step: int):
        scale = (step + 1) / warmup_steps if step < warmup_steps else 1.0
        for pg in opt.param_groups:
            pg["lr"] = base_lr * scale

    best_val = float("-inf")
    best_state = None
    no_improve = 0
    start_time = time.time()
    deadline = start_time + (max_minutes * 60.0) if max_minutes else None

    step = 0
    for epoch in range(1, epochs + 1):
        if deadline and time.time() >= deadline:
            print(f"    [ES] max_minutes reached at epoch {epoch-1}")
            break

        model.train()
        running = 0.0
        for xb, qmask, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            set_lr(step); step += 1
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        # eval
        val_acc = _evaluate_mqar(model, val_loader, device)
        print(f"    epoch {epoch:02d}/{epochs}  train_loss={running/max(1,steps_per_epoch):.4f}  val_mqar_acc={val_acc:.4f}")

        # improvement check
        improved = val_acc > (best_val + min_delta)
        if improved and epoch >= min_epochs:
            best_val = val_acc
            no_improve = 0
            if restore_best:
                # keep a light copy on CPU
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        # early stopping?
        if early_stop and epoch >= min_epochs and no_improve >= patience:
            print(f"    [ES] stop at epoch {epoch} (best={best_val:.4f}, no_improve={no_improve})")
            break

    # restore best weights for fair reporting
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    # best_val is what we want to return
    return 0.0 if best_val == float("-inf") else best_val

# ------------------------------------------------------------------------------
# Public API 1: run_mqar_benchmark  (now also saves JSONs + prints full header)
# ------------------------------------------------------------------------------
def run_mqar_benchmark(device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
                       benchmark_type: str = "tt_only",
                       kv_pairs: int = KV_PAIRS,
                       train_size: int = 100_000,
                       val_size: int = 3_000,
                       *,
                       prebuilt_ds: Optional[Tuple[TensorDataset, TensorDataset]] = None,
                       batch_size_override: Optional[int] = None,
                       results_dir: Optional[str] = None,
                       rep_index: int | None = None,
                       rep_total: int | None = None) -> Dict[str, float]:
    """
    Train the chosen model(s) on Zoology MQAR and return best val accuracy
    per model after sweeping the LR grid. Also writes JSONs to ./results/<benchmark_type>/.
    """
    _set_all_seeds(SEED)
    device = torch.device(device_str)
    print(f"\n--- MQAR benchmark ({benchmark_type}) on {device} ---")
        # # Full header printed for each model

        # _print_run_header(header)
    # Build (or reuse) dataset with fixed order across runs
    if prebuilt_ds is None:
        cfg = MQARZoologyConfig(
            vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, kv_pairs=kv_pairs,
            train_examples=train_size, val_examples=val_size
        )
        train_ds, val_ds = _make_datasets(cfg)
    else:
        train_ds, val_ds = prebuilt_ds

    bs = batch_size_override if batch_size_override is not None else _batch_size_by_rule(SEQ_LEN, EMBED_DIM)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, drop_last=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False, pin_memory=True)

    # Prepare output directory
    out_dir = results_dir or _ensure_dir(os.path.join("results", _safe_name(benchmark_type)))
    run_tag = f"N{SEQ_LEN}_D{kv_pairs}_d{EMBED_DIM}_train{train_size}_val{val_size}"

    # Build models
    models = _build_models(benchmark_type, VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)

    results: Dict[str, float] = {}
    for name, _ in models.items():
        print(f"name={name} | === KV={kv_pairs} | width={EMBED_DIM} | rep {rep_index}/{rep_total} | benchmark={benchmark_type} ===")
        header = {
            "model_name"    : name,
            "benchmark_type": benchmark_type,
            "device"        : str(device),
            "seq_len"       : SEQ_LEN,
            "kv_pairs"      : kv_pairs,
            "embed_dim"     : EMBED_DIM,
            "n_heads"       : N_HEADS,
            "epochs"        : EPOCHS,
            "batch_size"    : bs,
            "train_size"    : train_size,
            "val_size"      : val_size,
            "lr_grid"       : list(LR_GRID),
            "weight_decay"  : WEIGHT_DECAY,
            "warmup_frac"   : WARMUP_FRAC
        }

        best_val_acc = float("-inf")
        best_lr = LR_GRID[0]
        per_lr = {}

        for lr in LR_GRID:
            print(f"  [lr={lr:.1e}]")
            # fresh init per LR
            model = _build_models(benchmark_type, VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)[name]
            acc = _train_one_model(model, train_loader, val_loader, device, base_lr=lr)
            if not math.isfinite(acc): acc = float("-inf")
            per_lr[f"{lr:.1e}"] = 0.0 if acc == float("-inf") else float(acc)
            if acc >= best_val_acc:
                best_val_acc, best_lr = acc, lr

        # Safe print & record
        if not math.isfinite(best_val_acc): best_val_acc = 0.0
        print(f"  Best for {name}: acc={best_val_acc:.4f} at lr={best_lr:.1e}")
        results[name] = float(best_val_acc)

        # Save JSON per model
        file_tag = f"{_safe_name(name)}__{run_tag}__{_now_tag()}.json"
        out_path = os.path.join(out_dir, file_tag)
        _save_json(out_path, {
            "timestamp"      : _now_tag(),
            "benchmark_type" : benchmark_type,
            "model_name"     : name,
            "run"            : header,
            "val_acc_per_lr" : per_lr,
            "best"           : {"lr": best_lr, "val_acc": best_val_acc}
        })
        print(f"  Saved results → {out_path}")

    return results
import re

def _plot_ablation_triads(results_by_kv, out_dir: str, context_len: int):
    """
    For ablation runs, split models into triads by base (SS/SM/MS/MM)
    and save one figure per triad using plot_mqar_capacity_panels.
    """
    # Collect all model names from one kv (any kv works for grouping)
    any_kv = next(iter(results_by_kv))
    all_models = list(results_by_kv[any_kv].keys())

    # Group by base tag inside "Ablation[XX]"
    triads = {}  # base -> [model names]
    for m in all_models:
        m_base = None
        mtag = re.search(r"Ablation\[(SS|SM|MS|MM)\]", m)
        if mtag: 
            m_base = mtag.group(1)
        else:
            # if someone renamed labels, fallback: try to infer from leading token
            # (won't trigger with our current labels; safe no-op otherwise)
            pass
        if m_base:
            triads.setdefault(m_base, []).append(m)

    # Make a figure per base
    for base, models in sorted(triads.items()):
        # filter the big dict down to only these models
        sub = {}
        for kv, panel in results_by_kv.items():
            sub[kv] = {m: panel[m] for m in panel.keys() if m in models}

        # save
        save_path = os.path.join(out_dir, f"ablation_{base}_capacity_panels_N{context_len}.png")
        plot_mqar_capacity_panels(sub, context_len=context_len, save_path=save_path)
        print(f"Ablation panels saved → {save_path}")

# ------------------------------------------------------------------------------
# Public API 2: sweep_mqar_capacity  (plots + saves aggregated JSON)
# ------------------------------------------------------------------------------
def sweep_mqar_capacity(device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
                        benchmark_type: str = "tl_two_block_combinations",
                        kv_settings: Tuple[int, ...] = (16, 32),
                        width_settings: Tuple[int, ...] = (64, 128, 256),
                        seq_len: int = 128,
                        train_size: int = 20_000,
                        val_size: int = 4_000,
                        repetitions: int = 1) -> Dict[int, Dict[str, Dict[int, List[float]]]]:
    """
    Returns:
      results_by_kv: { kv : { model_name : { width : [acc, acc, ...] } } }
    Also saves:
      - JSON summary in ./results/<benchmark_type>/
      - capacity panels image via plot_mqar_capacity_panels(..., save_path=...)
      - heatmap "mqar_benchmark_heatmap.png" in the same folder
    """
    global SEQ_LEN, EMBED_DIM
    _set_all_seeds(SEED)
    device = torch.device(device_str)

    # Output directory for this benchmark
    out_dir = _ensure_dir(os.path.join("results", _safe_name(benchmark_type)))
    # Cache datasets per (seq_len, kv, train_size, val_size)
    dataset_cache: dict[Tuple[int, int, int, int], Tuple[TensorDataset, TensorDataset]] = {}

    # Collect in the exact structure the main plotter expects
    results_by_kv: Dict[int, Dict[str, Dict[int, List[float]]]] = {}

    for kv in kv_settings:
        print(f"\n===== KV={kv}  (seq_len={seq_len}) =====")

        cache_key = (seq_len, kv, train_size, val_size)
        if cache_key not in dataset_cache:
            cfg = MQARZoologyConfig(
                vocab_size=VOCAB_SIZE, seq_len=seq_len, kv_pairs=kv,
                train_examples=train_size, val_examples=val_size
            )
            dataset_cache[cache_key] = _make_datasets(cfg)
        prebuilt_ds = dataset_cache[cache_key]

        for width in width_settings:
            SEQ_LEN = seq_len
            EMBED_DIM = width
            bs = _batch_size_by_rule(seq_len, width)

            for rep in range(repetitions):
                print(f"=== KV={kv} | width={width} | rep {rep+1}/{repetitions} | benchmark={benchmark_type} ===")
                run_out = run_mqar_benchmark(
                    device_str=device_str,
                    benchmark_type=benchmark_type,
                    kv_pairs=kv,
                    train_size=train_size,
                    val_size=val_size,
                    prebuilt_ds=prebuilt_ds,       # reuse dataset
                    batch_size_override=bs,        # correct BS per width
                    results_dir=out_dir,
                    rep_index=rep+1,              
                    rep_total=repetitions         
                )
                for model_name, acc in run_out.items():
                    results_by_kv.setdefault(kv, {}).setdefault(model_name, {}).setdefault(width, []).append(acc)

    # ---- MAIN PLOT -----------------------------------------------------------
# ---- MAIN PLOT(S) -----------------------------------------------------------
    if benchmark_type.startswith("xlstm_ablation"):
        # produce one figure per triad (e.g., SM vs LM vs LT)
        _plot_ablation_triads(results_by_kv, out_dir=out_dir, context_len=seq_len)
    else:
        # non-ablation: single aggregated figure as before
        panels_path = os.path.join(out_dir, f"capacity_panels_N{seq_len}_train{train_size}_val{val_size}.png")
        plot_mqar_capacity_panels(results_by_kv, context_len=seq_len, save_path=panels_path)
        print(f"Capacity panels saved → {panels_path}")


    # ---- Heatmap summary: best width per model at largest KV -----------------
    try:
        kv_for_heatmap = max(results_by_kv.keys())
        flat_for_heatmap: Dict[str, float] = {}
        for model_name, width_dict in results_by_kv[kv_for_heatmap].items():
            best_w, best_mean = None, float("-inf")
            for w, accs in width_dict.items():
                m = float(np.mean(accs)) if accs else float("-inf")
                if not math.isfinite(m): m = float("-inf")
                if m >= best_mean: best_mean, best_w = m, w
            label = f"{model_name} [d={best_w}]" if best_w is not None else model_name
            flat_for_heatmap[label] = 0.0 if best_mean == float("-inf") else best_mean
        heatmap_path = os.path.join(out_dir, f"mqar_heatmap_N{seq_len}_kv{kv_for_heatmap}.png")
        plot_mqar_heatmap(flat_for_heatmap, filename=heatmap_path)
        print(f"Heatmap saved → {heatmap_path}")
    except Exception as e:
        print("Heatmap skipped due to error:", e)

    # ---- Save aggregated JSON -----------------------------------------------
    summary_path = os.path.join(out_dir, f"summary_N{seq_len}_train{train_size}_val{val_size}.json")
    _save_json(summary_path, results_by_kv)
    print(f"Summary JSON saved → {summary_path}")

    return results_by_kv

# ──────────────────────────────────────────────────────────────────────────────
# Fast LR scout for all two-block combos (stage 1)
# ──────────────────────────────────────────────────────────────────────────────
def lr_scout_all_two_block_combos(
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    *,
    seq_len: int = 128,
    kv_pairs: int = 32,
    width: int = 128,
    epochs: int = 6,
    lr_candidates: Tuple[float, ...] = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    train_size: int = 10_000,
    val_size: int = 2_000
) -> Dict[str, float]:
    """
    Returns {model_name: best_lr} using a tiny dataset + short runs.
    """
    _set_all_seeds(SEED)
    device = torch.device(device_str)

    # Build dataset once (cached locally inside function)
    cfg = MQARZoologyConfig(
        vocab_size=VOCAB_SIZE, seq_len=seq_len, kv_pairs=kv_pairs,
        train_examples=train_size, val_examples=val_size
    )
    train_ds, val_ds = _make_datasets(cfg)
    bs = _batch_size_by_rule(seq_len, width)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, drop_last=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False, pin_memory=True)

    # Temporarily override globals for builder
    global SEQ_LEN, EMBED_DIM
    old_seq, old_w = SEQ_LEN, EMBED_DIM
    SEQ_LEN, EMBED_DIM = seq_len, width

    try:
        models = _build_models("all_two_block_combos", VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)
        best_lrs: Dict[str, float] = {}

        for name in sorted(models.keys()):
            print(f"\n[LR-Scout] {name}")
            best_acc = float("-inf")
            best_lr  = lr_candidates[0]

            for lr in lr_candidates:
                # fresh init each LR
                model = _build_models("all_two_block_combos", VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)[name]
                acc = _train_one_model(model, train_loader, val_loader, device,
                                       base_lr=lr, epochs=epochs,
                                       weight_decay=WEIGHT_DECAY, warmup_frac=WARMUP_FRAC)
                if not math.isfinite(acc): acc = float("-inf")
                if acc >= best_acc:
                    best_acc, best_lr = acc, lr

            print(f"  → best_lr={best_lr:.1e}  scout_acc={0.0 if not math.isfinite(best_acc) else best_acc:.4f}")
            best_lrs[name] = best_lr

        # Save LR map
        out_dir = _ensure_dir(os.path.join("results", "all_two_block_combos"))
        _save_json(os.path.join(out_dir, f"lr_scout_N{seq_len}_D{kv_pairs}_d{width}.json"), best_lrs)
        return best_lrs
    finally:
        SEQ_LEN, EMBED_DIM = old_seq, old_w

# ──────────────────────────────────────────────────────────────────────────────
# Full runs with fixed LRs per model (stage 2)
# ──────────────────────────────────────────────────────────────────────────────
def run_mqar_with_fixed_lrs(
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    *,
    benchmark_type: str = "all_two_block_combos",
    kv_pairs: int = 32,
    seq_len: int = 128,
    width: int = 128,
    train_size: int = 100_000,
    val_size: int = 3_000,
    best_lr_map: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Uses provided best_lr_map {model_name: lr} to train each model once
    (no LR sweep). Returns {model_name: best_val_acc}.
    """
    assert best_lr_map, "best_lr_map is required"
    _set_all_seeds(SEED)
    device = torch.device(device_str)

    # Build dataset once
    cfg = MQARZoologyConfig(
        vocab_size=VOCAB_SIZE, seq_len=seq_len, kv_pairs=kv_pairs,
        train_examples=train_size, val_examples=val_size
    )
    train_ds, val_ds = _make_datasets(cfg)
    bs = _batch_size_by_rule(seq_len, width)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, drop_last=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False, pin_memory=True)

    # Temporarily override globals
    global SEQ_LEN, EMBED_DIM
    old_seq, old_w = SEQ_LEN, EMBED_DIM
    SEQ_LEN, EMBED_DIM = seq_len, width

    try:
        models = _build_models(benchmark_type, VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)
        results: Dict[str, float] = {}
        out_dir = _ensure_dir(os.path.join("results", _safe_name(benchmark_type)))

        for name, _ in models.items():
            lr = best_lr_map.get(name, None)
            if lr is None:
                print(f"  (skip) no LR for {name}")
                continue
            # Full header print for consistency
            header = {
                "model_name": name, "benchmark_type": benchmark_type, "device": str(device),
                "seq_len": SEQ_LEN, "kv_pairs": kv_pairs, "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
                "epochs": EPOCHS, "batch_size": bs, "train_size": train_size, "val_size": val_size,
                "lr_grid": [lr], "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC
            }
            _print_run_header(header)

            # fresh init
            model = _build_models(benchmark_type, VOCAB_SIZE, EMBED_DIM, SEQ_LEN, N_HEADS, num_layers=1)[name]
            acc = _train_one_model(model, train_loader, val_loader, device,
                                   base_lr=lr, epochs=EPOCHS,
                                   weight_decay=WEIGHT_DECAY, warmup_frac=WARMUP_FRAC)
            results[name] = float(0.0 if not math.isfinite(acc) else acc)

            # Save JSON
            run_tag = f"N{SEQ_LEN}_D{kv_pairs}_d{EMBED_DIM}_train{train_size}_val{val_size}"
            file_tag = f"{_safe_name(name)}__fixedLR__{run_tag}__{_now_tag()}.json"
            _save_json(os.path.join(out_dir, file_tag), {
                "timestamp": _now_tag(),
                "benchmark_type": benchmark_type,
                "model_name": name,
                "run": header,
                "best": {"lr": lr, "val_acc": results[name]}
            })

        return results
    finally:
        SEQ_LEN, EMBED_DIM = old_seq, old_w
