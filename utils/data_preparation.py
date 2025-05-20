# utils/data_preparation.py
import torch
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset 
from collections import Counter
from typing import List, Dict

# --- NNS Data Generation  ---
def nns_unit_vector():
    v = torch.randn(2)
    return v / v.norm()

def nns_build_sample(max_len: int, input_dim: int):
    L = random.randint(2, max_len)
    ref = nns_unit_vector()
    seq = [torch.cat([ref, torch.zeros(1)])]
    best_val, best_d = None, 1e9
    for _ in range(L - 1):
        v = nns_unit_vector()
        val = torch.rand(1)
        seq.append(torch.cat([v, val]))
        d = (v - ref).pow(2).sum().item()
        if d < best_d:
            best_d, best_val = d, val.item()
    seq.extend([torch.zeros(input_dim)] * (max_len - L))
    return torch.stack(seq), torch.tensor([best_val])

class NNSDataset(Dataset):
    def __init__(self, n_samples: int, max_len: int, input_dim: int):
        self.max_len = max_len
        self.input_dim = input_dim
        self.data = [nns_build_sample(self.max_len, self.input_dim) for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# --- LM Data Utilities  ---
def lm_basic_tokenize(line: str) -> List[str]:
    return line.strip().split()

def load_wikitext2_data(cfg: Dict):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_tokens: List[str] = []
    for txt in raw["train"]["text"]:
        train_tokens.extend(lm_basic_tokenize(txt))

    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    counter = Counter(train_tokens)
    vocab = specials + [w for w, _ in counter.most_common()]
    word2idx = {w: i for i, w in enumerate(vocab)}
    unk_idx, pad_idx = word2idx["<unk>"], word2idx["<pad>"]
    vocab_size = len(vocab)
    print(f"Tokenising train corpus…\n→ Vocab size = {vocab_size:,}")

    def _encode(lines):
        ids: List[int] = []
        for ln in lines:
            toks = lm_basic_tokenize(ln)
            if toks:
                ids.extend(word2idx.get(t, unk_idx) for t in toks)
                ids.append(word2idx["<eos>"])
        return torch.tensor(ids, dtype=torch.long)

    train_ids = _encode(raw["train"]["text"])
    val_ids = _encode(raw["validation"]["text"])
    test_ids = _encode(raw["test"]["text"])

    def _batchify(data: torch.Tensor, bsz: int):
        nbatch = data.numel() // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        return data.view(bsz, -1).contiguous()

    train_data = _batchify(train_ids, cfg["batch_size"])
    val_data = _batchify(val_ids, cfg["batch_size"])
    test_data = _batchify(test_ids, cfg["batch_size"])
    freq_map = torch.bincount(train_ids, minlength=vocab_size)

    return vocab_size, pad_idx, freq_map, train_data, val_data, test_data

def get_lm_batch(source: torch.Tensor, idx: int, bptt: int):
    seq_len = min(bptt, source.size(1) - 1 - idx)
    if seq_len <= 0:
        return None, None
    return source[:, idx:idx + seq_len], source[:, idx + 1: idx + 1 + seq_len]