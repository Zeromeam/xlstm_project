import torch, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xlstm_replica as xlstm_scratch
from xlstm import (
    xLSTMBlockStack as LibXLSTMBlockStack,
    xLSTMBlockStackConfig as LibXLSTMBlockStackConfig,
    sLSTMBlockConfig as LibSLSTMBlockConfig,
    sLSTMLayerConfig as LibSLSTMLayerConfig,
    FeedForwardConfig as LibFeedForwardConfig,
)
from utils.training_loops import make_lm_scheduler
from utils.plotting import plot_nns_results as plot_results

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
VOCAB = {"a":2, "b":3, "c":4, "(":5, ")":6, "#":1}  # 0=pad, 1=end-of-seq
SEQ_LEN      = 64
EMBED_DIM    = 128
BATCH_SIZE   = 128
TOTAL_STEPS  = 25_000
VALID_EVERY  = 500
LR_PEAK      = 1e-3
LR_FINAL     = 1e-7
DROPOUT      = 0.10
WEIGHT_DECAY = 0.01
CLIP_NORM    = 1.0
PATIENCE     = 8

# ─── Dataset ──────────────────────────────────────────────────────────────────
class FormalLangDataset(Dataset):
    """
    Mixture of three tasks – one for each Chomsky level we care about.
      • Parity (regular): # of 'a' chars mod 2  → label 0/1
      • Dyck-1  (context-free): balanced parentheses? → 0/1
      • a^n b^n c^n (context-sensitive): 0/1
    Target label is appended as last token (“0” or “1”)
    Loss is CE on that last position.
    """
    def __init__(self, n_samples:int):
        self.samples = []
        for _ in range(n_samples):
            typ = torch.randint(0,3,(1,)).item()
            if typ == 0:         
                length = torch.randint(8, SEQ_LEN-1, (1,)).item()
                seq = torch.randint(VOCAB["a"], VOCAB["b"]+1,
                                    (length,)).tolist()  # only 'a'
                label = sum(1 for t in seq if t==VOCAB["a"]) % 2
            elif typ == 1:          
                n_pairs = torch.randint(2, (SEQ_LEN-2)//4, (1,)).item()
                seq = []
                for _ in range(n_pairs):
                    if torch.rand(1).item()<0.5:
                        seq += [VOCAB["("]] + [VOCAB[")"]]
                    else:
                        seq = [VOCAB["("]] + seq + [VOCAB[")"]]
                label = int(self._is_balanced(seq))
            else:                   # a^n b^n c^n
                n = torch.randint(2, (SEQ_LEN-1)//3, (1,)).item()
                seq = [VOCAB["a"]]*n + [VOCAB["b"]]*n + [VOCAB["c"]]*n
                label = 1
                # corrupt 50 % of the time
                if torch.rand(1).item() < 0.5:
                    swap = torch.randint(0,len(seq),(2,))
                    seq[swap[0]], seq[swap[1]] = seq[swap[1]], seq[swap[0]]
                    label = 0
            seq = seq[:SEQ_LEN-1]                  
            pad_len = SEQ_LEN-1-len(seq)
            inp = torch.tensor(seq + [0]*pad_len + [VOCAB["#"]]) 
            tgt = torch.full((SEQ_LEN,), -1, dtype=torch.long)
            tgt[-1] = label+7    
            self.samples.append((inp, tgt))

    def _is_balanced(self, seq):
        bal = 0
        for t in seq:
            if t==VOCAB["("]: bal+=1
            elif t==VOCAB[")"]: bal-=1
            if bal<0: return False
        return bal==0

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):  return self.samples[i]

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
            pred = model(xb).argmax(-1)   # [B,L]
            hit += (pred[:,-1]==yb[:,-1]).sum().item()
            tot += xb.size(0)
    return hit/tot if tot else float("nan")

# ─── Driver ───────────────────────────────────────────────────────────────────
def run_formal_benchmark(device_str="cpu",
                         benchmark_type="xlstm_vs_lib",
                         train_size=20_000,
                         val_size=4_000):
    device=torch.device(device_str)
    print(f"\n--- Formal-Language benchmark ({benchmark_type}) on {device} ---")

    train_ds=FormalLangDataset(train_size)
    val_ds  =FormalLangDataset(val_size)
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    val_loader  =DataLoader(val_ds,BATCH_SIZE)

    if benchmark_type=="xlstm_vs_lib":
        models={"xLSTM-Scratch":_ScratchXLSTM(),
                "xLSTM-Lib":_LibXLSTM()}
    else:
        raise ValueError

    results={}
    loss_fn=nn.CrossEntropyLoss(ignore_index=-1)
    for name,model in models.items():
        model.to(device)
        opt=optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
        sched=make_lm_scheduler(opt, TOTAL_STEPS, warmup_pct=0.1,
                                final_lr=LR_FINAL, peak_lr=LR_PEAK)
        best_acc,best_state,stall=0.,None,0; step=0
        while step<TOTAL_STEPS and stall<PATIENCE:
            for xb,yb in train_loader:
                if step>=TOTAL_STEPS: break
                model.train(); xb=xb.to(device); yb=yb.to(device)
                opt.zero_grad()
                logits=model(xb).transpose(1,2)
                loss=loss_fn(logits,yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_NORM)
                opt.step(); sched.step(); step+=1
                if step%VALID_EVERY==0:
                    acc=_cls_acc(model,val_loader,device)
                    print(f"[{name}] {step}/{TOTAL_STEPS} "
                          f"loss {loss.item():.3f} acc {acc:.3f}")
                    if acc>best_acc:
                        best_acc,best_state=acc,copy.deepcopy(model.state_dict())
                        stall=0
                    else:
                        stall+=1
        if best_state: model.load_state_dict(best_state)
        results[name]=_cls_acc(model,val_loader,device)
        print(f"{name} final acc: {results[name]:.3f}")

    plot_results(results,"Formal-Language Validation Accuracy",
                 "Accuracy (higher is better)",
                 "formal_benchmark.png")
    return results
