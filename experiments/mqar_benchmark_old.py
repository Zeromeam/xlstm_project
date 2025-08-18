
import torch, math, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
import xlstm_replica_old as xlstm_scratch
from xlstm import (
    xLSTMBlockStack as LibXLSTMBlockStack,
    xLSTMBlockStackConfig as LibXLSTMBlockStackConfig,
    sLSTMBlockConfig as LibSLSTMBlockConfig,
    sLSTMLayerConfig as LibSLSTMLayerConfig,
    mLSTMBlockConfig as LibmLSTMBlockConfig,
    mLSTMLayerConfig as LibmLSTMLayerConfig,
    FeedForwardConfig as LibFeedForwardConfig,
)
from utils.training_loops import make_lm_scheduler         
from utils.plotting import plot_nns_results as plot_results 
from utils.model_architectures import build_hybrid_core_lm, SimpleLMWrapper
from utils.model_architectures import GeneralHybridLM,GeneralHybridConfig 
from utils.plotting import plot_mqar_capacity_panels      
import contextlib
import numpy as np


# ─── MQAR hyper-params ─────────────────────────────────────────────────────────
VOCAB_SIZE     = 8192  
KV_PAIRS     = 16      
SEQ_LEN        = 64        
BATCH_SIZE     = 256
EMBED_DIM      = 128
N_HEADS       = 8
TOTAL_STEPS    = 25_000
VALID_EVERY    = 500
LR_PEAK        = 1e-3
LR_FINAL       = 1e-4
DROPOUT        = 0.0
WEIGHT_DECAY   = 0.1
GRAD_CLIP_NORM = 1.0
PATIENCE       = 5          

# ======================================================================
# Zoology-style MQAR (Appendix E/E.2) – dataset + metric
# Keeps your old sentinel MQAR intact; this is an alternate path.
# ======================================================================
from dataclasses import dataclass
from typing import Tuple

class _ZoologyRNG:
    def __init__(self, seed: int):
        import random, numpy as _np
        self.py = random.Random(seed)
        self.np = _np.random.RandomState(seed)

@dataclass
class MQARZoologyConfig:
    vocab_size: int = 8192
    seq_len: int = 64
    alpha: float = 0.1
    kv_pairs: int = None           # default N//8
    train_examples: int = 100_000
    val_examples: int   = 3_000

    # model
    d_model: int = 64
    n_heads: int = 1               # paper uses 1 head
    n_layers: int = 2              # two T blocks total ("TT")

    # schedule
    epochs: int = 64
    lr_grid: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    weight_decay: float = 0.1
    warmup_frac: float = 0.10

    # derived
    batch_size: int = 64

    def finalize(self):
        if self.kv_pairs is None:
            self.kv_pairs = max(2, self.seq_len // 8)
        # batch policy from Appendix E.2
        if self.seq_len >= 512 or self.d_model >= 512:
            self.batch_size = 8
        elif self.seq_len >= 256 or self.d_model >= 256:
            self.batch_size = 16
        else:
            self.batch_size = 64

def _zoo_powerlaw_positions(low, high, alpha, num, rs, avoid=None, need_two=True):

    assert low <= high
    idx = np.arange(low, high + 1, dtype=np.int64)
    if need_two and len(idx) and idx[-1] == high:
        idx = idx[idx <= high - 1]
    ranks = np.arange(1, len(idx) + 1, dtype=np.float64)
    p = ranks ** (-alpha); p = p / p.sum()
    chosen, occ = [], set(avoid) if avoid else set()
    for _ in range(num):
        pos = None
        # try sampling a few times, then scan
        for __ in range(20):
            cand = int(rs.choice(idx, p=p))
            if need_two:
                if cand not in occ and (cand+1) not in occ:
                    pos = cand; break
            else:
                if cand not in occ:
                    pos = cand; break
        if pos is None:
            for cand in idx:
                if need_two:
                    if cand not in occ and (cand+1) not in occ:
                        pos = int(cand); break
                else:
                    if cand not in occ:
                        pos = int(cand); break
        if pos is None: pos = int(low)
        chosen.append(pos); occ.add(pos)
        if need_two: occ.add(pos+1)
    return chosen

def _zoo_generate_one(cfg: MQARZoologyConfig, key2val, rng: _ZoologyRNG):
    N, D, V = cfg.seq_len, cfg.kv_pairs, cfg.vocab_size
    seq   = np.full(N, fill_value=0, dtype=np.int64)    # filler token = 0
    qmask = np.zeros(N, dtype=np.int8)

    # split vocab halves: keys [0..V/2), values [V/2..V)
    keys = rng.py.sample(range(0, V//2), D)
    vals = [int(key2val[k]) for k in keys]

    # place first D pairs at the start: [k1, v1, k2, v2, ...]
    p = 0
    for k, v in zip(keys, vals):
        seq[p] = k; seq[p+1] = v; p += 2

    # second occurrences: power-law start positions in [2D .. N-2]
    avoid = set(range(0, 2*D))
    starts = _zoo_powerlaw_positions(2*D, N-1, cfg.alpha, D, rs=rng.np, avoid=avoid, need_two=True)

    order = list(range(D)); rng.py.shuffle(order)
    for i, idx in enumerate(order):
        k, v = keys[idx], vals[idx]
        s = starts[i]
        if seq[s] != 0 or (s+1 < N and seq[s+1] != 0):
            placed = False
            for q in range(2*D, N-1):
                if seq[q] == 0 and seq[q+1] == 0:
                    s = q; placed = True; break
            if not placed:
                for q in range(N-2, 2*D-1, -1):
                    if seq[q] == 0 and seq[q+1] == 0:
                        s = q; break
        seq[s] = k
        if s+1 < N: seq[s+1] = v
        qmask[s] = 1  # query at the second key

    labels = np.full(N, fill_value=-100, dtype=np.int64)
    labels[:-1] = seq[1:]
    return seq, qmask, labels

def _make_zoology_dataset(cfg: MQARZoologyConfig):
    rng = _ZoologyRNG(1337)

    # global bijection keys→values
    key2val = np.zeros(cfg.vocab_size//2, dtype=np.int64)
    vals_pool = list(range(cfg.vocab_size//2, cfg.vocab_size))
    rng.py.shuffle(vals_pool)
    for k in range(cfg.vocab_size//2):
        key2val[k] = vals_pool[k % (cfg.vocab_size - cfg.vocab_size//2)]

    def build(num, seed_offset):
        xs = np.zeros((num, cfg.seq_len), dtype=np.int64)
        q  = np.zeros((num, cfg.seq_len), dtype=np.int8)
        ys = np.zeros((num, cfg.seq_len), dtype=np.int64)
        rng_local = _ZoologyRNG(1337 + seed_offset)
        for i in range(num):
            s, qm, lab = _zoo_generate_one(cfg, key2val, rng_local)
            xs[i], q[i], ys[i] = s, qm, lab
        return torch.from_numpy(xs), torch.from_numpy(q), torch.from_numpy(ys)

    x_tr, q_tr, y_tr = build(cfg.train_examples, 12345)
    x_va, q_va, y_va = build(cfg.val_examples,   54321)
    
    return TensorDataset(x_tr, q_tr, y_tr), TensorDataset(x_va, q_va, y_va)

@torch.no_grad()
def _zoology_mqar_acc(model, loader, device):
    """Accuracy only at second-key query positions (Appendix E metric)."""
    model.eval()
    tot = correct = 0
    for xb, qmask, yb in loader:
        xb, qmask, yb = xb.to(device), qmask.to(device).bool(), yb.to(device)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1)
        mask = qmask & (yb != -100)
        tot += mask.sum().item()
        correct += (preds[mask] == yb[mask]).sum().item()
    return correct / max(1, tot)


# ─── helper: sentinel-only accuracy (paper’s metric) ──────────────────────────
@torch.no_grad()
def _sentinel_acc(model, loader, device):
    model.eval(); hit=tot=0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        pred = model(xb)[0].argmax(-1)
        mask = (xb == 1)
        hit += (pred[mask] == yb[mask]).sum().item()
        tot += mask.sum().item()
    return hit/tot if tot else float("nan")

# ─── Model helpers ─────────────────────────────────────────────────────────────
class _ScratchXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        layer_cfg   = xlstm_scratch.sLSTMLayerConfig(
            embedding_dim=EMBED_DIM, context_length=SEQ_LEN,
            num_heads=4, dropout=DROPOUT, conv1d_kernel_size=0
        )
        block_cfg   = xlstm_scratch.sLSTMBlockConfig(
            slstm_layer_config=layer_cfg,
            feedforward_config=xlstm_scratch.FeedForwardConfig(
                proj_factor=1.0, act_fn="gelu", dropout=DROPOUT)
        )
        stack_cfg   = xlstm_scratch.xLSTMBlockStackConfig(
            mlstm_block_template=None,
            slstm_block_template=block_cfg,
            num_blocks=2, embedding_dim=EMBED_DIM,
            context_length=SEQ_LEN, dropout=DROPOUT,
            add_post_blocks_norm=True, slstm_at="all")
        self.body  = xlstm_scratch.xLSTMBlockStack(stack_cfg)
        self.proj  = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):               # [B, L] → [B, L, V]
        h = self.embed(x)
        h = self.body(h)
        return self.proj(h)

class _LibXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        cfg = LibXLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=LibSLSTMBlockConfig(
                slstm=LibSLSTMLayerConfig(
                    num_heads=4, dropout=DROPOUT,
                    conv1d_kernel_size=0, backend="vanilla"),
                feedforward=LibFeedForwardConfig(
                    proj_factor=1.0, act_fn="gelu", dropout=DROPOUT)
            ),
            context_length=SEQ_LEN, num_blocks=2,
            embedding_dim=EMBED_DIM, dropout=DROPOUT,
            add_post_blocks_norm=True)
        self.body = LibXLSTMBlockStack(cfg)
        self.proj = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        h = self.embed(x)
        h = self.body(h)
        return self.proj(h)

class _LibXLSTM_Mixed(nn.Module):
    """
    Library xLSTM with 4 blocks, alternating sLSTM ↔ mLSTM, all sharing
    the same embedding dim so it is directly comparable to the Hybrid TLTL
    (LSTM ↔ Transformer) stack.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        # ── sLSTM block template ────────────────────────────────────────────
        s_cfg = LibSLSTMBlockConfig(
            slstm=LibSLSTMLayerConfig(
                num_heads=4,
                dropout=DROPOUT,
                conv1d_kernel_size=0,
                backend="vanilla"),
            feedforward=LibFeedForwardConfig(
                proj_factor=1.0,
                act_fn="gelu",
                dropout=DROPOUT)
        )


        m_cfg = LibmLSTMBlockConfig(
            LibmLSTMLayerConfig(               
                EMBED_DIM // 4,              
                4,                            
                DROPOUT                      
            ),
            LibFeedForwardConfig(            
                proj_factor=1.0,
                act_fn="gelu",
                dropout=DROPOUT
            )
)
        stack_cfg = LibXLSTMBlockStackConfig(
            slstm_block=s_cfg,
            mlstm_block=m_cfg,
            num_blocks=4,              # S M S M  
            context_length=SEQ_LEN,
            embedding_dim=EMBED_DIM,
            dropout=DROPOUT,
            add_post_blocks_norm=True,
        )
        self.body = LibXLSTMBlockStack(stack_cfg)
        self.proj = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        h = self.embed(x)
        h = self.body(h)
        return self.proj(h)

# ─── Evaluation helper ────────────────────────────────────────────────────────
def _accuracy(model, loader, device):
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            raw_out = model(xb)
            logits  = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            preds   = logits.argmax(-1)
            mask    = yb != -1
            correct += (preds[mask] == yb[mask]).sum().item()
            tot     += mask.sum().item()
    return float("nan") if tot == 0 else correct / tot


# ─── Training driver ──────────────────────────────────────────────────────────
def run_mqar_benchmark(device_str="cpu",
                       benchmark_type="xlstm_vs_lib",
                       kv_pairs=KV_PAIRS,
                       train_size=100_000,
                       val_size=3_000):
    device = torch.device(device_str)
    print(f"\n--- MQAR benchmark ({benchmark_type}) on {device} ---")

    train_ds = MQARDataset(train_size, kv_pairs)
    val_ds   = MQARDataset(val_size, kv_pairs)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,  BATCH_SIZE)


    if benchmark_type == "xlstm_vs_lib":
        
        models = {#"lstm_trans_hypered"   : lstm_trans_hypered,
                  "xLSTM-Scratch": _ScratchXLSTM(),
                  "xLSTM-Lib"   : _LibXLSTM()}
    elif benchmark_type == "hybrid_vs_xlstm":
        # ── Hybrid TLTL core (Transformer↔LSTM↔Transformer↔LSTM) ───────────
        hybrid_core = build_hybrid_core_lm(
            input_dim=EMBED_DIM,
            hidden_dim=EMBED_DIM,
            pattern="TLTL",      
            n_heads=4,
            dropout=DROPOUT,
            max_len=SEQ_LEN
        )
        hybrid_model = SimpleLMWrapper(
            hybrid_core,
            vocab_size=VOCAB_SIZE,
            model_dim=EMBED_DIM,
            pad_idx=0
        )

        models = {
            "Hybrid-TLTL": hybrid_model,
            "xLSTM-Mixed": _LibXLSTM_Mixed()
        }
    elif benchmark_type == "tt_only_2": 
        hybrid_core = build_hybrid_core_lm(
            input_dim=EMBED_DIM,
            hidden_dim=EMBED_DIM,
            pattern="TT",      
            n_heads=4,
            dropout=DROPOUT,
            max_len=SEQ_LEN
        )
        hybrid_model = SimpleLMWrapper(
            hybrid_core,
            vocab_size=VOCAB_SIZE,
            model_dim=EMBED_DIM,
            pad_idx=0
        )

        models = {
            "Hybrid-TLTL": hybrid_model,
            #"xLSTM-Mixed": _LibXLSTM_Mixed()
        }
    elif benchmark_type == "general_hybrid_mqar":
        cfg_TLTL = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM, 
                block_string="TLTL",
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0
        )
        cfg_msms = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM, 
                block_string="msms",
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0
        )
        models = {
            "General-S (MQAR)": GeneralHybridLM(cfg_TLTL),
            "General-SM (MQAR)": GeneralHybridLM(cfg_msms),
            "xLSTM-Lib": _LibXLSTM_Mixed()
        }
    elif benchmark_type == "sm_combinations":
        models = {}
        sm_combinations = [
            'SSSS', 'SSSM', 'SSMS', 'SMSS', 'MSSS', 'SSMM', 'SMSM', 'SMMS', 
            'MSMS', 'MSSM', 'MMSS', 'SMMM', 'MSMM', 'MMSM', 'MMMS', 'MMMM'
        ]
        for combo in sm_combinations:
            cfg = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM, 
                block_string=combo,
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0
            )
            models[f"General-{combo} (MQAR)"] = GeneralHybridLM(cfg)

    elif benchmark_type == "lt_combinations":
        models = {}
        lt_combinations = [
            'LLLL', 'LLLT', 'LLTL', 'LTLL', 'TLLL', 'LLTT', 'LTLT', 'LTTL', 
            'TLTL', 'TLLT', 'TTLL', 'LTTT', 'TLTT', 'TTLT', 'TTTL', 'TTTT'
        ]
        for combo in lt_combinations:
            cfg = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM, 
                block_string=combo,
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0
            )
            models[f"General-{combo} (MQAR)"] = GeneralHybridLM(cfg)
    # ─── 2-block xLSTM (S/M) sweep ───────────────────────────────────────────
    elif benchmark_type == "xlstm_two_block_combinations":
        models = {}
        for combo in (a + b for a in "SM" for b in "SM"):           # SS, SM, MS, MM
            cfg = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM,
                block_string=combo,
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0,
                num_layers=1
            )
            models[f"General-{combo} (MQAR)"] = GeneralHybridLM(cfg)

    # ─── 2-block TL (L/T) sweep ──────────────────────────────────────────────
    elif benchmark_type == "tl_two_block_combinations":
        models = {}
        for combo in (a + b for a in "LT" for b in "LT"):           # LL, LT, TL, TT
            cfg = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM,
                block_string=combo,
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0,
                num_layers=1
            )
            models[f"General-{combo} (MQAR)"] = GeneralHybridLM(cfg)
    # –– 2-block TT-only sweep ––––––––––––––––––––––––––––––––––––––––––
    elif benchmark_type == "tt_only":
        models = {}                 # ← loop over depths
        cfg = GeneralHybridConfig(
            vocab_size     = VOCAB_SIZE,
            embed_dim      = EMBED_DIM,
            hidden_dim     = EMBED_DIM,
            block_string   = "TT",                # fixed pattern
            context_length = SEQ_LEN,
            n_heads        = 4,
            dropout        = DROPOUT,
            pad_idx        = 0,
            num_layers     = 1            # ← varying depth
        )
        models[f"General-TT-{1}L (MQAR)"] = GeneralHybridLM(cfg)
    elif benchmark_type == "only_tt":
        models = {}
        cfg = GeneralHybridConfig(
            vocab_size      = VOCAB_SIZE,
            embed_dim       = 128,    # width
            hidden_dim      = 128,
            block_string    = "TT",   # TWO transformer blocks
            context_length  = SEQ_LEN,
            n_heads         = 4,
            dropout         = 0.10,
            pad_idx         = 0,
            num_layers      = 2       # one encoder layer per “T”
        )
        models["General-TT (Formal)"] = GeneralHybridLM(cfg)
    # ─── 2-block “SM vs LM” mini-sweep ───────────────────────────────────────
    elif benchmark_type == "sm_vs_lm":
        models = {}
        for combo in ("SM", "LM"):
            cfg = GeneralHybridConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM,
                block_string=combo,
                context_length=SEQ_LEN,
                n_heads=4,
                dropout=DROPOUT,
                pad_idx=0,
                num_layers=1
            )
            models[f"General-{combo} (MQAR)"] = GeneralHybridLM(cfg)
    # ─── Pair-wise ablations (2-column heat-maps) ──────────────────────────────
    elif benchmark_type in (
            "ss_vs_ll",   # sLSTM→LSTM   (recurrence only)
            "ms_vs_ml",   #     "        (attn-first order)
            "sm_vs_lm",   # already present – keeps as is
            "sm_vs_st",   # mLSTM→Transformer (attn injected late)
            "ms_vs_ts",   #        "
            "mm_vs_tt"    #        "
    ):
        pair_lookup = {
            "ss_vs_ll": ("SS", "LL"),
            "ms_vs_ml": ("MS", "ML"),
            "sm_vs_lm": ("SM", "LM"),      # ← existing
            "sm_vs_st": ("SM", "ST"),
            "ms_vs_ts": ("MS", "TS"),
            "mm_vs_tt": ("MM", "TT"),
        }
        combos = pair_lookup[benchmark_type]
        models = {}
        for combo in combos:
            cfg = GeneralHybridConfig(
                vocab_size   = VOCAB_SIZE,
                embed_dim    = EMBED_DIM,
                hidden_dim   = EMBED_DIM,
                block_string = combo,
                context_length = SEQ_LEN,
                dropout      = DROPOUT,
                n_heads      = 4,
                pad_idx      = 0,
            )
            tag = "Formal" if "formal" in __file__ else "MQAR"
            models[f"General-{combo} ({tag})"] = GeneralHybridLM(cfg)
    # ─── Zoology-style baseline with TT (2 blocks, 1 head) ────────────────────
    elif benchmark_type == "zoology_tt":
        # Build Zoology data with paper schedule
        cfgZ = MQARZoologyConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=SEQ_LEN,                   # uses global set by sweep_mqar_capacity
            kv_pairs=kv_pairs,                 # also set by caller
            d_model=EMBED_DIM,                 # width from sweep
            n_heads=1,                         # Appendix E/E.2
            n_layers=2,                        # two T blocks total
            train_examples=train_size,
            val_examples=val_size,
        )
        cfgZ.finalize()

        train_ds, val_ds = _make_zoology_dataset(cfgZ)
        train_loader = DataLoader(train_ds, batch_size=cfgZ.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfgZ.batch_size, shuffle=False, pin_memory=True, drop_last=False)

        # one TT model (GeneralHybridLM with "TT", 1 head, no dropout)
        from utils.model_architectures import GeneralHybridLM, GeneralHybridConfig
        tt_cfg = GeneralHybridConfig(
            vocab_size=cfgZ.vocab_size,
            embed_dim=cfgZ.d_model,
            hidden_dim=cfgZ.d_model,
            block_string="TT",
            context_length=cfgZ.seq_len,
            n_heads=cfgZ.n_heads,     # 1
            dropout=0.0,              # no dropout per paper
            pad_idx=0,
            num_layers=1              # one encoder layer per “T”
        )
        model = GeneralHybridLM(tt_cfg).to(device)

        def run_one_lr(lr: float) -> float:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfgZ.weight_decay, betas=(0.9, 0.95))
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * cfgZ.epochs
            warmup = max(1, int(cfgZ.warmup_frac * total_steps))
            step = 0

            def set_lr(s: int):
                scale = (s + 1) / warmup if s < warmup else 1.0
                for pg in opt.param_groups:
                    pg["lr"] = lr * scale

            best = 0.0
            for ep in range(1, cfgZ.epochs + 1):
                model.train()
                run_loss = 0.0
                for xb, qmask, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    set_lr(step); step += 1
                    opt.zero_grad(set_to_none=True)
                    out = model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    opt.step()
                    run_loss += loss.item()
                val_acc = _zoology_mqar_acc(model, val_loader, device)
                print(f"[zoo lr={lr:.1e}] epoch {ep:02d}/{cfgZ.epochs}  "
                      f"train_loss={run_loss/max(1,steps_per_epoch):.4f}  val_mqar_acc={val_acc:.4f}")
                best = max(best, val_acc)
            return best

        best_acc, best_lr = 0.0, None
        for lr in (1e-4, 3e-4, 1e-3, 3e-3):  # Appendix E/E.2 grid
            # fresh model per LR:
            model = GeneralHybridLM(tt_cfg).to(device)
            acc = run_one_lr(lr)
            if acc > best_acc: best_acc, best_lr = acc, lr
        print(f"Zoology-TT done. best_acc={best_acc:.4f} @ lr={best_lr:.1e}")
        return {"Zoology-TT (MQAR)": best_acc}
    else:
        raise ValueError(f"Unknown type {benchmark_type}")

    peak_lrs = {n: LR_PEAK for n in models}

    results = {}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    for name, model in models.items():
        model.to(device)
        optimiser = optim.AdamW(model.parameters(),
                                lr=peak_lrs[name], betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
        sched = make_lm_scheduler(optimiser, TOTAL_STEPS,
                                  warmup_pct=0.1,
                                  final_lr=LR_FINAL,
                                  peak_lr=peak_lrs[name])

        best_acc, best_state, steps_no_improve = 0., None, 0
        step = 0
        early_stop = False  
        while step < TOTAL_STEPS and steps_no_improve < PATIENCE and not early_stop:
            for xb, yb in train_loader:
                if step >= TOTAL_STEPS: break
                model.train()
                xb = xb.to(device); yb = yb.to(device)
                optimiser.zero_grad()
                raw_out = model(xb)
                logits   = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                logits   = logits.transpose(1, 2)           # [B, V, L] for CE
                loss     = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               GRAD_CLIP_NORM)
                optimiser.step(); sched.step()
                step += 1

                if step % VALID_EVERY == 0:
                    acc = _sentinel_acc(model, val_loader, device)
                    print(f"[{name}] step {step}/{TOTAL_STEPS} "
                          f"train-loss {loss.item():.3f} "
                          f"val-acc {acc:.4f}")
                    if acc > best_acc:
                        best_acc, best_state = acc, copy.deepcopy(model.state_dict())
                        steps_no_improve = 0
                    else:
                        steps_no_improve += 1
                if best_acc >= 1.0:
                    print(f"[{name}] perfect accuracy reached – stopping early.")
                    early_stop = True
                    break
        # restore best
        if best_state: model.load_state_dict(best_state)
        results[name] = _sentinel_acc(model, val_loader, device)
        print(f"{name} final accuracy: {results[name]:.4f}")
        del model
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    # –– 1-D bar chart (kept, because it is still handy) ––––––––––––––––––––––––
    plot_results(results,
                "MQAR Validation Accuracy",
                "Accuracy (higher is better)",
                "mqar_benchmark.png")

    # –– NEW: row-normalised heat-map –––––––––––––––––––––––––––––––––––––––––––
    from utils.plotting import plot_mqar_heatmap
    plot_mqar_heatmap(results, filename="mqar_benchmark_heatmap.png")

    return results
# ─── Convenience wrapper: reproduce Figure-5 style plot ────────────────
def sweep_mqar_capacity(device_str="cpu",
                        benchmark_type="lt_combinations",
                        kv_settings=(16, 32),
                        width_settings=(64, 128, 256),
                        seq_len=128,
                        train_size=20_000,
                        val_size=4_000,
                        repetitions: int = 1):
    """
    Runs the MQAR benchmark *repetitions* times for every (kv, width) pair.
    Accuracy lists are stored; the plotting code now shows mean ± 95 % CI.
    """
    results_by_kv = {}
    for kv in kv_settings:
        results_by_kv[kv] = {}
        for width in width_settings:
            global EMBED_DIM, SEQ_LEN
            EMBED_DIM, SEQ_LEN = width, seq_len

            for rep in range(repetitions):
                run_out = run_mqar_benchmark(device_str=device_str,
                                             benchmark_type=benchmark_type,
                                             kv_pairs=kv,
                                             train_size=train_size,
                                             val_size=val_size)
                for model, acc in run_out.items():
                    results_by_kv[kv].setdefault(model, {}).setdefault(width, []).append(acc)

    plot_mqar_capacity_panels(results_by_kv, context_len=seq_len)
