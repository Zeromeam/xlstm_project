import torch, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xlstm_replica_old as xlstm_scratch
from xlstm import (
    xLSTMBlockStack as LibXLSTMBlockStack,
    xLSTMBlockStackConfig as LibXLSTMBlockStackConfig,
    sLSTMBlockConfig as LibSLSTMBlockConfig,
    sLSTMLayerConfig as LibSLSTMLayerConfig,
    FeedForwardConfig as LibFeedForwardConfig,
)
from utils.training_loops import make_lm_scheduler
from utils.plotting import plot_nns_results as plot_results
from utils.model_architectures import GeneralHybridLM, GeneralHybridConfig
from utils.plotting import plot_nns_results, plot_formal_heatmaps

import numpy as np                       # ← new

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
VOCAB = {"a":2, "b":3, "c":4, "(":5, ")":6, "#":1}  # 0=pad, 1=end-of-seq



SEQ_LEN = 64
# ── task toggles ────────────────────────────────────────────

PARITY_DISTRACTORS = True
EMBED_DIM    = 256
N_HEADS = 4
BATCH_SIZE   = 512
TOTAL_STEPS  = 12_000
VALID_EVERY  = 500
LR_PEAK      = 1e-3
LR_FINAL     = 1e-7
DROPOUT      = 0.10
WEIGHT_DECAY = 0.01
CLIP_NORM    = 1.0
PATIENCE     = 3

# ─── Dataset ──────────────────────────────────────────────────────────────────
class FormalLangDataset(Dataset):
    """
    Three formal-language classification tasks.
    Target label (0 / 1) is stored in the last token position (idx = SEQ_LEN-1).
    Returned tensors:
        inp  : LongTensor [SEQ_LEN]   – 0-padded, EOS='#' at position −1
        tgt  : LongTensor [SEQ_LEN]   – -1 everywhere except last pos (label+7)
    This version follows the set-up used in the original xLSTM paper:
        • Parity strings contain *only* ‘a’ symbols (no distractors).
        • Dataset is class-balanced: exactly 50 % positive, 50 % negative.
        • Dyck-1 and aⁿbⁿcⁿ tasks are unchanged (already faithful).
    """
    def __init__(self, n_samples: int, fixed_typ: int | None = None):
        super().__init__()
        self.samples = []

        # helper -------------------------------------------------------------
        def _pad_and_pack(seq: list[int], label: int):
            seq = seq[:SEQ_LEN - 1]                     # safety crop
            pad_len = SEQ_LEN - 1 - len(seq)
            inp = torch.tensor(seq + [0] * pad_len + [VOCAB["#"]])
            tgt = torch.full((SEQ_LEN,), -1, dtype=torch.long)
            tgt[-1] = label + 7                         # labels live at 7 / 8
            return inp, tgt

        for _ in range(n_samples):
            typ = fixed_typ if fixed_typ is not None else torch.randint(0, 3, ()).item()

            # ── 0. Parity (regular) ────────────────────────────────────────
            if typ == 0:
                # choose even / odd with equal probability ------------------
                want_odd = bool(torch.randint(0, 2, ()).item())
                # sample a length that matches the desired parity
                #   min length 4 avoids degenerate strings
                length = torch.randint(4, SEQ_LEN - 1, ()).item()
                if length % 2 != want_odd:           # flip if parity wrong
                    length += 1 if length < SEQ_LEN - 2 else -1
                if PARITY_DISTRACTORS:
                    # ~50 % of positions become distractors
                    seq = torch.randint(2, 5, (length,)).tolist()  # ids 2 = a, 3 = b, 4 = c
                else:
                    seq = [VOCAB["a"]] * length
                label = length % 2                   # 0 = even, 1 = odd
                inp, tgt = _pad_and_pack(seq, label)

            # ── 1. Dyck-1 (context-free) ───────────────────────────────────
            elif typ == 1:
                n_pairs = torch.randint(2, (SEQ_LEN - 2) // 4, ()).item()
                seq = []
                for _ in range(n_pairs):
                    if torch.rand(()) < 0.5:
                        seq += [VOCAB["("], VOCAB[")"]]
                    else:
                        seq = [VOCAB["("]] + seq + [VOCAB[")"]]

                label = 1
                if torch.rand(()) < 0.5:             # corrupt half the time
                    if torch.rand(()) < 0.5 and seq:                         # delete
                        seq.pop(torch.randint(0, len(seq), ()).item())
                    else:                                                   # swap
                        i, j = torch.randint(0, len(seq), (2,))
                        seq[i], seq[j] = seq[j], seq[i]
                    label = 0
                inp, tgt = _pad_and_pack(seq, label)

            # ── 2. aⁿbⁿcⁿ (context-sensitive) ──────────────────────────────
            else:
                n = torch.randint(2, (SEQ_LEN - 1) // 3, ()).item()
                seq = [VOCAB["a"]] * n + [VOCAB["b"]] * n + [VOCAB["c"]] * n
                label = 1
                if torch.rand(()) < 0.5:              # corrupt half the time
                    i, j = torch.randint(0, len(seq), (2,))
                    seq[i], seq[j] = seq[j], seq[i]
                    label = 0
                inp, tgt = _pad_and_pack(seq, label)

            self.samples.append((inp, tgt))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):  return self.samples[idx]

# ─── Models (tiny) ────────────────────────────────────────────────────────────
class _ScratchXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, EMBED_DIM)    
        layer = xlstm_scratch.sLSTMLayerConfig(
            embedding_dim=EMBED_DIM, context_length=SEQ_LEN,
            num_heads=4, dropout=DROPOUT, conv1d_kernel_size=0)
        block = xlstm_scratch.sLSTMBlockConfig(
            slstm_layer_config=layer,
            feedforward_config=xlstm_scratch.FeedForwardConfig(
                proj_factor=1.0, act_fn="gelu", dropout=DROPOUT))
        stack = xlstm_scratch.xLSTMBlockStackConfig(
            mlstm_block_template=None, slstm_block_template=block,
            num_blocks=2, embedding_dim=EMBED_DIM, context_length=SEQ_LEN,
            dropout=DROPOUT, add_post_blocks_norm=True, slstm_at="all")
        self.body = xlstm_scratch.xLSTMBlockStack(stack)
        self.proj = nn.Linear(EMBED_DIM, 9)

    def forward(self, x):
        return self.proj(self.body(self.emb(x)))

class _LibXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, EMBED_DIM)
        cfg = LibXLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=LibSLSTMBlockConfig(
                slstm=LibSLSTMLayerConfig(
                    num_heads=4, dropout=DROPOUT,
                    conv1d_kernel_size=0, backend="vanilla"),
                feedforward=LibFeedForwardConfig(
                    proj_factor=1.0, act_fn="gelu", dropout=DROPOUT)),
            context_length=SEQ_LEN, num_blocks=2,
            embedding_dim=EMBED_DIM, dropout=DROPOUT,
            add_post_blocks_norm=True)
        self.body = LibXLSTMBlockStack(cfg)
        self.proj = nn.Linear(EMBED_DIM, 9)

    def forward(self, x):
        return self.proj(self.body(self.emb(x)))

# ─── Accuracy helper ──────────────────────────────────────────────────────────
def _cls_acc(model, loader, device):
    model.eval(); hit=tot=0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            
            output = model(xb)
            
            if isinstance(output, tuple):
                logits = output[0]  # The logits are the first element
            else:
                logits = output
            
            pred = logits.argmax(-1)   # [B,L]
            # ------------------------------------

            hit += (pred[:,-1]==yb[:,-1]).sum().item()
            tot += xb.size(0)
    return hit/tot if tot else float("nan")

# ─── Driver ───────────────────────────────────────────────────────────────────
def run_formal_benchmark(device_str="cpu",
                         benchmark_type="xlstm_vs_lib",
                         train_size=20_000,
                         val_size=4_000,
                         repetitions: int = 1):
    
    import importlib
    import experiments.ch_formal_benchmark as chb  
    device=torch.device(device_str)
    agg_scalar: dict[str, list[float]] = {}
    agg_heat:   dict[str, list[list[float]]] = {}

    for _ in range(repetitions):        # ← start outer repetition loop
        print(f"\n--- Formal-Language benchmark ({benchmark_type}) on {device} ---")




        if benchmark_type=="xlstm_vs_lib":
            models={"xLSTM-Scratch":_ScratchXLSTM(),
                    "xLSTM-Lib":_LibXLSTM()}
            
        elif benchmark_type == "sm_combinations":
            models = {}
            sm_combinations = [
                'SSSS', 'SSSM', 'SSMS', 'SMSS', 'MSSS', 'SSMM', 'SMSM', 'SMMS', 
                'MSMS', 'MSSM', 'MMSS', 'SMMM', 'MSMM', 'MMSM', 'MMMS', 'MMMM'
            ]
            for combo in sm_combinations:
                cfg = GeneralHybridConfig(
                    vocab_size=9,
                    embed_dim=EMBED_DIM,
                    hidden_dim=EMBED_DIM,
                    block_string=combo,
                    context_length=SEQ_LEN,
                    n_heads=4,
                    dropout=DROPOUT,
                    pad_idx=0
                )
                models[f"General-{combo} (Formal)"] = GeneralHybridLM(cfg)
                
        elif benchmark_type == "lt_combinations":
            models = {}
            lt_combinations = [
                'LLLL', 'LLLT', 'LLTL', 'LTLL', 'TLLL', 'LLTT', 'LTLT', 'LTTL', 
                'TLTL', 'TLLT', 'TTLL', 'LTTT', 'TLTT', 'TTLT', 'TTTL', 'TTTT'
            ]
            for combo in lt_combinations:
                cfg = GeneralHybridConfig(
                    vocab_size=9,
                    embed_dim=EMBED_DIM,
                    hidden_dim=EMBED_DIM,
                    block_string=combo,
                    context_length=SEQ_LEN,
                    n_heads=4,
                    dropout=DROPOUT,
                    pad_idx=0
                )
                models[f"General-{combo} (Formal)"] = GeneralHybridLM(cfg)
        # ─── 2-block xLSTM (S/M) sweep ─────────────────────────────────────
        elif benchmark_type == "xlstm_two_block_combinations":
            models = {}
            for combo in (a + b for a in "SM" for b in "SM"):           # SS, SM, MS, MM
                cfg = GeneralHybridConfig(
                    vocab_size=9,
                    embed_dim=EMBED_DIM,
                    hidden_dim=EMBED_DIM,
                    block_string=combo,          # e.g. "SM"
                    context_length=SEQ_LEN,
                    n_heads=N_HEADS,
                    dropout=DROPOUT,
                    pad_idx=0,
                    num_layers= 1
                )
                models[f"General-{combo} (Formal)"] = GeneralHybridLM(cfg)

        # ─── 2-block TL (L/T) sweep ────────────────────────────────────────
        elif benchmark_type == "tl_two_block_combinations":
            models = {}
            for combo in (a + b for a in "LT" for b in "LT"):           # LL, LT, TL, TT
                cfg = GeneralHybridConfig(
                    vocab_size=9,
                    embed_dim=EMBED_DIM,
                    hidden_dim=EMBED_DIM,
                    block_string=combo,
                    context_length=SEQ_LEN,
                    n_heads=N_HEADS,
                    dropout=DROPOUT,
                    pad_idx=0
                )
                models[f"General-{combo} (Formal)"] = GeneralHybridLM(cfg)
        # ─── 2-block “SM vs LM” sweep ─────────────────────────────────────────
        elif benchmark_type == "only_tt":
            models = {}                 # S-M vs L-M
            cfg = GeneralHybridConfig(
                vocab_size=9,
                embed_dim=EMBED_DIM,
                hidden_dim=EMBED_DIM,
                block_string='TT',                     # e.g. "SM"
                context_length=SEQ_LEN,
                n_heads=N_HEADS,
                dropout=DROPOUT,
                pad_idx=0
            )
            models[f"General-{'TT'} (Formal)"] = GeneralHybridLM(cfg)
        elif benchmark_type == "sm_vs_lm":
            models = {}
            for combo in ("SM", "LM"):                      # S-M vs L-M
                cfg = GeneralHybridConfig(
                    vocab_size=9,
                    embed_dim=EMBED_DIM,
                    hidden_dim=EMBED_DIM,
                    block_string=combo,                     # e.g. "SM"
                    context_length=SEQ_LEN,
                    n_heads=N_HEADS,
                    dropout=DROPOUT,
                    pad_idx=0
                )
                models[f"General-{combo} (Formal)"] = GeneralHybridLM(cfg)
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
                    vocab_size   = 9 ,
                    embed_dim    = EMBED_DIM,
                    hidden_dim   = EMBED_DIM,
                    block_string = combo,
                    context_length = SEQ_LEN,
                    n_heads      = N_HEADS ,
                    dropout      = DROPOUT,
                    pad_idx      = 0,
                    num_layers= 1
                )
                tag = "Formal" if "formal" in __file__ else "MQAR"
                models[f"General-{combo} ({tag})"] = GeneralHybridLM(cfg)
        elif benchmark_type == "missing_models_only":
            # Only the “both-substitutions” variants that were missing:
            #   - LT: from base SM (S→L and M→T)
            #   - TL: from base MS (S→L and M→T)
            models = {}
            for combo in ("LT", "TL"):
                cfg = GeneralHybridConfig(
                    vocab_size=9,
                    embed_dim=EMBED_DIM,
                    hidden_dim=EMBED_DIM,
                    block_string=combo,          # "LT" or "TL"
                    context_length=SEQ_LEN,
                    n_heads=N_HEADS,
                    dropout=DROPOUT,
                    pad_idx=0,
                    num_layers=1                 # keep 2‑block parity with other tests
                )
                models[f"General-{combo} (Formal)"] = GeneralHybridLM(cfg)


        else:
            raise ValueError

        TASKS = {
            0: "Parity (regular)",
            1: "Dyck-1 (context-free)",
            2: "aⁿbⁿcⁿ (context-sensitive)"
        }

        results_heat   = {}   # 3 values per model (one per task)
        results_scalar = {}   # optional overall average, keeps old bar-plot working
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
        for name, model_proto in models.items():
            print(f"\n=== {name} ===")
            results_heat[name] = []

            for task_idx, task_lbl in TASKS.items():
                train_ds = FormalLangDataset(train_size)

                val_ds   = FormalLangDataset(val_size)

                         # reset to short strings for next task
                train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
                val_loader   = DataLoader(val_ds,  BATCH_SIZE)

                # ── fresh copy of the model (keeps original weights intact) ────
                model = copy.deepcopy(model_proto).to(device)
                for p in model.parameters():               # quick re-initialisation
                    if p.dim() > 1:  nn.init.xavier_uniform_(p)

                opt   = optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
                sched = make_lm_scheduler(opt, TOTAL_STEPS, warmup_pct=0.1,
                                        final_lr=LR_FINAL, peak_lr=LR_PEAK)

                best_acc = 0.0; best_state = None; stall = 0; step = 0
                while step < TOTAL_STEPS and stall < PATIENCE:
                    for xb, yb in train_loader:
                        if step >= TOTAL_STEPS:
                            break
                        model.train()
                        xb = xb.to(device); yb = yb.to(device)
                        opt.zero_grad()

                        out = model(xb)[0] if isinstance(model(xb), tuple) else model(xb)
                        loss = loss_fn(out.transpose(1, 2), yb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                        opt.step(); sched.step(); step += 1

                        if step % VALID_EVERY == 0:
                            acc = _cls_acc(model, val_loader, device)
                            print(f"  [{task_lbl}] step {step}/{TOTAL_STEPS}  "
                                f"loss {loss.item():.3f}  acc {acc:.3f}")
                            if acc > best_acc:
                                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
                                stall = 0
                            else:
                                stall += 1

                if best_state:
                    model.load_state_dict(best_state)

                final_acc = _cls_acc(model, val_loader, device)
                results_heat[name].append(final_acc)
                print(f"  → {task_lbl} final acc: {final_acc:.3f}")

            # overall mean keeps existing bar-plot compatible
            results_scalar[name] = sum(results_heat[name]) / len(results_heat[name])

        # --- aggregate --------------------------------------------------
        for m, overall in results_scalar.items():
            agg_scalar.setdefault(m, []).append(overall)

        for m, triple in results_heat.items():          # 3 tasks
            if m not in agg_heat:
                agg_heat[m] = [[], [], []]
            for i, val in enumerate(triple):
                agg_heat[m][i].append(val)
    plot_nns_results(
        agg_scalar,
        "Formal-Language Validation Accuracy (overall)",
        "Accuracy (higher is better)",
        "formal_benchmark_overall.png"
    )

    plot_formal_heatmaps(
        agg_heat,
        filename="formal_benchmark_heatmap.png"
    )
    agg_scalar_mean = {k: float(np.mean(v)) for k, v in agg_scalar.items()}
    return agg_scalar_mean    
