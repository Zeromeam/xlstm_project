
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict,Optional,Tuple
# -------------------------------------------------------------------
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig,
)
# --- Common Model Wrapper ---
class SimpleLMWrapper(nn.Module):
    """Wrap core stack with token embedding & untied LM head."""
    def __init__(self, core: nn.Module, vocab_size: int, model_dim: int, pad_idx: int, bias: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.core = core 
        self.lm_head = nn.Linear(core.out_dim, vocab_size, bias=bias)
        self.bias = bias 
        self._tie_or_init_weights()

    def _tie_or_init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
        
        
        if self.lm_head.in_features == self.embedding.embedding_dim and \
           self.lm_head.out_features == self.embedding.num_embeddings:
            print("Tying embedding and LM‑head weights ...")
            self.lm_head.weight = self.embedding.weight
        else:
            nn.init.uniform_(self.lm_head.weight, -0.1, 0.1)
        
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(self, ids: torch.Tensor, hidden=None):
        x = self.embedding(ids)
        if hasattr(self.core, "forward") and "hidden" in self.core.forward.__code__.co_varnames:
            x_core, hidden_out = self.core(x, hidden)
        else:
            x_core, hidden_out = self.core(x), None 
            if isinstance(x_core, tuple):
                 x_core, hidden_out_maybe = x_core
                 if hidden_out_maybe is not None:
                      hidden_out = hidden_out_maybe
           
        return self.lm_head(x_core), hidden_out


# --- Baseline Models for LM  ---


class LSTMCoreLM(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=dropout, batch_first=True)
        self.out_dim = hidden_dim 

    def forward(self, x: torch.Tensor, hidden=None):
        return self.lstm(x, hidden)

class TransformerCoreLM(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, n_heads: int, dropout: float, bptt: int):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, bptt) 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_dim = hidden_dim 

    def forward(self, x: torch.Tensor, hidden=None):  
        x_proj = self.proj_in(x) * math.sqrt(self.out_dim) 
        x_pos = self.pos_encoder(x_proj)
        
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        out = self.encoder(x_pos, mask=mask,is_causal=True) 
        return out, None


class PositionalEncoding(nn.Module): 
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (if batch_first)
               or [seq_len, batch_size, embedding_dim] (if not batch_first)
        """
        x = x + self.pe[:x.size(1)].transpose(0,1) 
        return self.dropout(x)
    



# --- Baseline Models for NNS ---
class TransformerModel_NNS(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, max_len: int): 
        super().__init__()
        self.inp = nn.Linear(input_dim, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len=max_len) 
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=4*embed_dim,
            batch_first=True,
            activation="gelu", 
            dropout=0.1 
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out = nn.Linear(embed_dim, 1)
        self.out_dim = embed_dim 

    def forward(self, x):
        mask = (x.abs().sum(dim=-1) == 0) 
        h = self.inp(x)
        h = self.pos(h) 
        h = self.enc(h, src_key_padding_mask=mask) 
        return self.out(h[:,-1]).squeeze(-1)

# --- Llama-Inspired Components for NNS ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) 
        return output * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) 
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, dim)
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))

class LlamaInspiredEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, ffn_intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, bias=False, batch_first=True)
        
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, ffn_intermediate_dim, dropout=dropout)
        self.dropout_res = nn.Dropout(dropout) 

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # x: (B, S, C)
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, 
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False) 
        x = x + self.dropout_res(attn_output)
        
        normed_x_ffn = self.norm2(x)
        ffn_output = self.ffn(normed_x_ffn)
        x = x + self.dropout_res(ffn_output)
        
        return x
    
# --- Llama-Inspired Transformer for NNS ---
class LlamaInspiredTransformerNNS(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, max_len: int, 
                 n_layers: int = 2, n_heads: int = 4, 
                 ffn_intermediate_dim: Optional[int] = None, 
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, embed_dim, bias=False) 
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout=dropout) 
        
        if ffn_intermediate_dim is None:
            ffn_intermediate_dim = embed_dim * 2 

        self.layers = nn.ModuleList()
        for _ in range(n_layers): 
            self.layers.append(
                LlamaInspiredEncoderBlock(embed_dim, n_heads, ffn_intermediate_dim, dropout)
            )
        
        self.final_norm = RMSNorm(embed_dim) 
        self.out = nn.Linear(embed_dim, 1, bias=False) 
        self.out_dim = embed_dim 

    def forward(self, x: torch.Tensor):
        key_padding_mask = (x.abs().sum(dim=-1) == 0) # (B, S)

        h = self.inp(x) 
        h = self.pos_encoder(h)

        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        
        h = self.final_norm(h)
        
        output_token_h = h[:, -1, :] 
        
        return self.out(output_token_h).squeeze(-1)
    
class LSTMModel_NNS(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int): 
        super().__init__()
        self.inp = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)
        self.out  = nn.Linear(embed_dim, 1)
        self.out_dim = embed_dim

    def forward(self, x):
        h_inp = self.inp(x)
        h_lstm, _ = self.lstm(h_inp)
        return self.out(h_lstm[:,-1]).squeeze(-1) 
    

    # the lstm trasnformers hypered models 


    # ──────────────────────────────────────────────────────────────────────────────
# HYBRID LSTM ↔ TRANSFORMER  CORE  (xLSTM-style configurable block stack)
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import List, Union

@dataclass
class LSTMBlockCfg:
    hidden_dim: int
    dropout: float = 0.1           
    bidirectional: bool = False   
    num_layers: int = 1 

@dataclass
class TransformerBlockCfg:
    hidden_dim: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

@dataclass
class HybridStackCfg:
    """Order of `blocks` defines the stack layout (e.g. [L,T,L])."""
    blocks: List[Union[LSTMBlockCfg, TransformerBlockCfg]]
    input_dim: int                 
    max_len: int = 512             

class _HybridLSTMBlock(nn.Module):
    def __init__(self, cfg: LSTMBlockCfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg.hidden_dim,
                            cfg.hidden_dim,
                            num_layers=cfg.num_layers,
                            batch_first=True,
                            dropout=cfg.dropout,
                            bidirectional=cfg.bidirectional)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, hidden=None):
        out, h = self.lstm(x, hidden)
        return self.dropout(self.norm(out)), h

class _ALiBiAttention(nn.Module):
    """Multi‑head self‑attention with ALiBi relative positional bias.
    Works in causal mode only (is_causal=True).
    """
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_slopes", self._build_slopes(n_heads), persistent=False)

    @staticmethod
    def _build_slopes(n_heads: int) -> torch.Tensor:
        """Slopes from the ALiBi paper (https://arxiv.org/abs/2108.12409)."""
        def get_pow2_slopes(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n_heads).is_integer():
            slopes = get_pow2_slopes(n_heads)
        else:  # handle non‑power‑of‑two num_heads
            closest_pow2 = 2 ** math.floor(math.log2(n_heads))
            slopes = get_pow2_slopes(closest_pow2)
            slopes += get_pow2_slopes(2 * closest_pow2)[0::2][: n_heads - closest_pow2]
        return torch.tensor(slopes).view(n_heads, 1, 1)  # (h,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.size()
        qkv = self.qkv(x)  # (b,s,3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)  # (b,h,s,hd)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (b,h,s,s)
        # build ALiBi bias on the fly (no large pre‑allocated tensor)
        pos = torch.arange(s, device=x.device)
        bias = pos.unsqueeze(0) - pos.unsqueeze(1)  # (s,s)
        bias = bias.unsqueeze(0).type_as(attn_scores)  # (1,1,s,s) via broadcasting later
        attn_scores = attn_scores - self._slopes * bias  # larger distance → more negative

        mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)

class _TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerBlockCfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = _ALiBiAttention(cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class RelativeTransformerBlock(nn.Module):
    """Drop‑in replacement for _HybridTransformerBlock.
    Returns (output, None) so that GeneralHybridLM can ignore the second value.
    """
    def __init__(self, cfg: TransformerBlockCfg, max_len: int):  # max_len kept for API compatibility
        super().__init__()
        self.layers = nn.ModuleList([_TransformerLayer(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x: torch.Tensor, *_):
        for layer in self.layers:
            x = layer(x)
        return x, None

class _HybridTransformerBlock(nn.Module):
    """
    This block is correct. It is designed to be self-contained
    and applies its own positional encoding, which is the proper
    design for a modular Transformer block.
    """
    def __init__(self, cfg: TransformerBlockCfg, max_len: int):
        super().__init__()
        self.pos = PositionalEncoding(cfg.hidden_dim, max_len, dropout=cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x, *_):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        # Applies its own positional encoding to the input it receives.
        #x = self.pos(x)
        #h = x 
        #h = self.enc(h, mask=mask, is_causal=True)
        x = self.enc(x, mask=mask, is_causal=True)
        # x = self.enc(x, is_causal=True)

        return self.norm(x), None
 

class HybridStackCoreLM(nn.Module):
    """
    xLSTM-inspired stack: an ordered list of blocks where each block
    is *either* an LSTM or a Transformer.
    """
    def __init__(self, cfg: HybridStackCfg):
        super().__init__()
        self.proj_in = nn.Linear(cfg.input_dim, cfg.blocks[0].hidden_dim)
        self.blocks = nn.ModuleList()
        for blk_cfg in cfg.blocks:
            if isinstance(blk_cfg, LSTMBlockCfg):
                self.blocks.append(_HybridLSTMBlock(blk_cfg))
            # elif isinstance(blk_cfg, TransformerBlockCfg):
            #     self.blocks.append(_HybridTransformerBlock(blk_cfg, cfg.max_len))
            elif isinstance(blk_cfg, TransformerBlockCfg):
                self.blocks.append(RelativeTransformerBlock(blk_cfg, cfg.max_len))
            else:
                raise ValueError(f"Unknown block cfg {type(blk_cfg)}")
        self.out_dim = cfg.blocks[0].hidden_dim  

    def forward(self, x: torch.Tensor, hidden=None):
        h = self.proj_in(x)
        next_hidden = []
        for blk in self.blocks:
            h, h_blk = blk(h, None)
            next_hidden.append(h_blk)
        return h, next_hidden  

def build_hybrid_core_lm(input_dim: int,
                         hidden_dim: int,
                         pattern: str = "LT",
                         n_heads: int = 4,
                         dropout: float = 0.1,
                         max_len: int = 512) -> HybridStackCoreLM:
    """
    pattern: string with 'L' (LSTM) and 'T' (Transformer) – e.g. "LLTT" or "TL".
    """
    blocks = []
    for ch in pattern.upper():
        if ch == "L":
            blocks.append(LSTMBlockCfg(hidden_dim=hidden_dim,
                                       dropout=dropout))
        elif ch == "T":
            blocks.append(TransformerBlockCfg(hidden_dim=hidden_dim,
                                              n_heads=n_heads,
                                              dropout=dropout))
        else:
            raise ValueError(f"Unsupported char {ch} in pattern; use L/T.")
    stack_cfg = HybridStackCfg(blocks=blocks,
                               input_dim=input_dim,
                               max_len=max_len)
    return HybridStackCoreLM(stack_cfg)



#####


from dataclasses import dataclass
from typing import List

@dataclass
class GeneralHybridConfig:
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    block_string: str         
    context_length: int
    n_heads: int = 4          
    dropout: float = 0.1
    pad_idx: int = 0
    num_layers: int = 1
    def __post_init__(self):
        allowed = set("tlms")
        if not set(self.block_string.lower()) <= allowed:
            raise ValueError(f"Unknown symbols in block_string: {self.block_string}")
        self.blocks: List[str] = list(self.block_string.lower())




class GeneralHybridLM(nn.Module):
    def __init__(self, cfg: GeneralHybridConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        
        # --- ADDED: The critical input projection layer. ---
        # It maps the embedding dimension to the model's hidden dimension.
        #self.proj_in = nn.Linear(cfg.embed_dim, cfg.hidden_dim)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([self._make_block(sym) for sym in cfg.blocks])

        # --- REMOVED: Final LayerNorm. The successful model does not have this. ---
        # self.norm = nn.LayerNorm(cfg.hidden_dim) 
        
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        # self.apply(self._init_weights)

        self.use_pos_enc = any(sym == "t" for sym in cfg.blocks)
        self.use_proj_in = (cfg.embed_dim != cfg.hidden_dim)


        if self.use_pos_enc:
            self.pos_enc_once = PositionalEncoding(
                cfg.hidden_dim, cfg.context_length, dropout=cfg.dropout
            )
        if self.use_proj_in:
            self.proj_in = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
            self._init_weights(self.proj_in)


        _init_emb(self.embed); _init_linear(self.lm_head)
    def _make_block(self, sym: str) -> nn.Module:
        cfg = self.cfg
        d = self.cfg.hidden_dim
        if sym == 'l':
            lstm_block_cfg = LSTMBlockCfg(hidden_dim=d, dropout=cfg.dropout,num_layers=cfg.num_layers)
            return _HybridLSTMBlock(lstm_block_cfg)
        if sym == 't':
            transformer_block_cfg = TransformerBlockCfg(hidden_dim=d, n_heads=cfg.n_heads, dropout=cfg.dropout, num_layers=cfg.num_layers  )
            return _HybridTransformerBlock(transformer_block_cfg, max_len=cfg.context_length)
        # if sym == 't':
        #     transformer_block_cfg = TransformerBlockCfg(
        #         hidden_dim=d, n_heads=cfg.n_heads,
        #         dropout=cfg.dropout, num_layers=cfg.num_layers)
        #     return RelativeTransformerBlock(transformer_block_cfg, max_len=cfg.context_length)
        if sym == 'm':
            mlstm_cfg = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    embedding_dim=d,
                    context_length=self.cfg.context_length,
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=self.cfg.n_heads,
                ),
            )
            stack = xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    mlstm_block=mlstm_cfg,
                    context_length=self.cfg.context_length,
                    num_blocks=cfg.num_layers,
                    embedding_dim=d,
                    add_post_blocks_norm=True,  
                )
            )
            return XLSTMWrapper(stack)
        if sym == 's':
            slstm_cfg = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla" ,
                    conv1d_kernel_size=4,
                    num_heads=self.cfg.n_heads,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            )
            stack = xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    slstm_block=slstm_cfg,
                    context_length=self.cfg.context_length,
                    num_blocks=cfg.num_layers,
                    embedding_dim=d,
                )
            )
            return XLSTMWrapper(stack)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, hidden=None):
        x = self.embed(idx)

        # --- Apply the projection and dropout ---
        if self.use_proj_in:
            x = self.proj_in(x) 
        # x = x * math.sqrt(self.cfg.hidden_dim)
        if self.use_pos_enc :
            x = x * math.sqrt(self.cfg.hidden_dim)
            x = self.pos_enc_once(x)
        x = self.dropout(x)

        for blk in self.blocks:
            x, hidden = blk(x, hidden)
            
        # --- The final LayerNorm is no longer here ---
        # x = self.norm(x)
        
        return self.lm_head(x), hidden



class XLSTMWrapper(nn.Module):
    def __init__(self, stack: nn.Module):
        super().__init__()
        self.stack = stack
    def forward(self, x, hidden=None):
        x_out = self.stack(x)
        return x_out, None
    

def _init_linear(m: nn.Linear):
    nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def _init_emb(e: nn.Embedding):
    nn.init.normal_(e.weight, mean=0.0, std=0.02)