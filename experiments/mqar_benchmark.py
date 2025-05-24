
import torch, math, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xlstm_replica as xlstm_scratch
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
# ─── MQAR hyper-params ─────────────────────────────────────────────────────────
VOCAB_SIZE     = 256      
SEQ_LEN        = 64        
BATCH_SIZE     = 128
EMBED_DIM      = 128
TOTAL_STEPS    = 25_000
VALID_EVERY    = 500
LR_PEAK        = 1e-3
LR_FINAL       = 1e-7
DROPOUT        = 0.10
WEIGHT_DECAY   = 0.01
GRAD_CLIP_NORM = 1.0
PATIENCE       = 8          

# ─── Dataset ───────────────────────────────────────────────────────────────────
class MQARDataset(Dataset):
    """
    Each sample:
        [k1 v1 k2 v2 … kN vN   q1 ? q2 ? … qN ?]   (SEQ_LEN fixed with padding 0)
        target = same length, value token where “?” appears, else -1 (ignore)
    """
    def __init__(self, num_samples: int, kv_pairs: int):
        super().__init__()
        self.samples = []
        for _ in range(num_samples):
            ks = torch.randint(2, VOCAB_SIZE, (kv_pairs,))
            vs = torch.randint(2, VOCAB_SIZE, (kv_pairs,))
            q_perm = torch.randperm(kv_pairs)
            inp  = torch.full((SEQ_LEN,), 0, dtype=torch.long)
            tgt  = torch.full((SEQ_LEN,), -1, dtype=torch.long)
            # build sequence
            ptr = 0
            for k, v in zip(ks, vs):
                inp[ptr]   = k;   ptr += 1
                inp[ptr]   = v;   ptr += 1
            for qi in q_perm:
                inp[ptr]   = ks[qi]; ptr += 1
                inp[ptr]   = 1      
                tgt[ptr]   = vs[qi]  
                ptr += 1
            self.samples.append((inp, tgt))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]

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

# ── put this near the top of mqar_benchmark.py, alongside _LibXLSTM ──────────
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

        # ── mLSTM block template (same dim, same dropout) ───────────────────

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
            num_blocks=4,              # S M S M  (even ↔ odd)
            context_length=SEQ_LEN,
            embedding_dim=EMBED_DIM,
            dropout=DROPOUT,
            add_post_blocks_norm=True,
            slstm_at="even",
            mlstm_at="odd"
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
                       kv_pairs=10,
                       train_size=20_000,
                       val_size=4_000):
    device = torch.device(device_str)
    print(f"\n--- MQAR benchmark ({benchmark_type}) on {device} ---")

    train_ds = MQARDataset(train_size, kv_pairs)
    val_ds   = MQARDataset(val_size,   kv_pairs)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,  BATCH_SIZE)


    # Choose models
    if benchmark_type == "xlstm_vs_lib":
        
        models = {#"lstm_trans_hypered"   : lstm_trans_hypered,
                  "xLSTM-Scratch": _ScratchXLSTM(),
                  "xLSTM-Lib"   : _LibXLSTM()}
    elif benchmark_type == "hybrid_vs_xlstm":
        # ── Hybrid TLTL core (Transformer↔LSTM↔Transformer↔LSTM) ───────────
        hybrid_core = build_hybrid_core_lm(
            input_dim=EMBED_DIM,
            hidden_dim=EMBED_DIM,
            pattern="TLTL",      # 4 blocks, alternates like the xLSTM stack
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
    else:
        raise ValueError(f"Unknown type {benchmark_type}")

    peak_lrs = {n: LR_PEAK for n in models}

    results = {}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    for name, model in models.items():
        model.to(device)
        optimiser = optim.AdamW(model.parameters(),
                                lr=peak_lrs[name], weight_decay=WEIGHT_DECAY)
        sched = make_lm_scheduler(optimiser, TOTAL_STEPS,
                                  warmup_pct=0.1,
                                  final_lr=LR_FINAL,
                                  peak_lr=peak_lrs[name])

        best_acc, best_state, steps_no_improve = 0., None, 0
        step = 0
        while step < TOTAL_STEPS and steps_no_improve < PATIENCE:
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
                    acc = _accuracy(model, val_loader, device)
                    print(f"[{name}] step {step}/{TOTAL_STEPS} "
                          f"train-loss {loss.item():.3f} "
                          f"val-acc {acc:.4f}")
                    if acc > best_acc:
                        best_acc, best_state = acc, copy.deepcopy(model.state_dict())
                        steps_no_improve = 0
                    else:
                        steps_no_improve += 1
        # restore best
        if best_state: model.load_state_dict(best_state)
        results[name] = _accuracy(model, val_loader, device)
        print(f"{name} final accuracy: {results[name]:.4f}")

    plot_results(results, "MQAR Validation Accuracy",
                 "Accuracy (higher is better)",
                 "mqar_benchmark.png")
    return results
