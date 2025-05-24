
"""
Self‑contained re‑implementation of the **xLSTM** building blocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Literal, List, Dict 

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy 

# Helper initialisation utilities: bias_linspace_init_, small_init_init_, wang_init_
def bias_linspace_init_(bias: torch.Tensor, *, start: float = 3.4, end: float = 6.0):
    with torch.no_grad():
        source_tensor = torch.linspace(start, end, bias.numel(), dtype=bias.dtype, device=bias.device)
        bias.copy_(source_tensor.reshape(bias.shape))
    return bias

def small_init_init_(weight: torch.Tensor, *, dim: int):
    """Kaiming‑style *small* initialisation used in the official code.
    Matches Transformers without Tears: Improving the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019)
    and EleutherAI/gpt-neox.
    """
    std = math.sqrt(2 / (5 * dim)) #
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
    return weight

def wang_init_(weight: torch.Tensor, *, dim: int, num_blocks: int):
    """Scaled Xavier initialisation from the paper appendix (Wang et al.).
    Adopted from EleutherAI/gpt-neox.
    """
    safe_num_blocks = max(1, num_blocks)
    std = (2 / safe_num_blocks) / math.sqrt(dim) #
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
    return weight

# LayerNorm, MultiHeadLayerNorm, CausalConv1d, LinearHeadwiseExpand
class LayerNorm(nn.LayerNorm):
    def __init__(self, ndim: int, *, weight: bool = True, bias: bool = True, eps: float = 1e-6):
        super().__init__(ndim, eps=eps, elementwise_affine=weight or bias)
        if self.elementwise_affine:
            if not weight:
                if self.weight is not None:
                    nn.init.ones_(self.weight)
                    self.weight.requires_grad_(False)
            if not bias:
                if self.bias is not None:
                    nn.init.zeros_(self.bias)
                    self.bias.requires_grad_(False)
    def reset_parameters(self): 
        if self.elementwise_affine:
            if self.weight is not None and self.weight.requires_grad: nn.init.ones_(self.weight)
            if self.bias is not None and self.bias.requires_grad: nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, 
                 num_heads: int,       # NH
                 head_dim: int,        # DH
                 *,
                 weight: bool = True,  # Corresponds to elementwise_affine for weight
                 bias: bool = False, # Corresponds to elementwise_affine for bias (cell.py uses bias=False for outnorm)
                 eps: float = 1e-5,  # Default from library's LayerNorm
                 residual_weight: bool = True): # Mimics library's LayerNorm default
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.residual_weight = residual_weight
        
        num_channels = num_heads * head_dim

        if weight:
            self.affine_weight = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('affine_weight', None)

        if bias:
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('affine_bias', None)
        
 

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.affine_weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.affine_weight
        else:
            return self.affine_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, NH, S, DH)
        B, NH_in, S, DH_in = x.shape

        assert NH_in == self.num_heads
        assert DH_in == self.head_dim

        # Reshape for F.group_norm, as in library's MultiHeadLayerNorm
        # (B, NH, S, DH) -> (B, S, NH, DH)
        gn_in_1 = x.transpose(1, 2)
        # (B, S, NH, DH) -> (B * S, NH * DH) where NH*DH is num_channels
        gn_in_2 = gn_in_1.reshape(B * S, self.num_heads * self.head_dim)

        out_gn = F.group_norm(
            gn_in_2,
            num_groups=self.num_heads, 
            weight=self.weight_proxy,  
            bias=self.affine_bias,
            eps=self.eps,
        )
        
        # Reshape back: (B * S, NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        output = out_gn.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return output

    def reset_parameters(self):
        # Initialize affine_weight to zeros (so weight_proxy starts at 1.0 if residual_weight)
        if self.affine_weight is not None:
            nn.init.zeros_(self.affine_weight)
        # Initialize affine_bias to zeros
        if self.affine_bias is not None:
            nn.init.zeros_(self.affine_bias)




@dataclass
class CausalConv1dConfig:
    feature_dim: int; kernel_size: int = 4; bias: bool = False
class CausalConv1d(nn.Module):
    config_class = CausalConv1dConfig
    def __init__(self, config: CausalConv1dConfig):
        super().__init__()
        self.config = config
        self.padding_left = config.kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=config.feature_dim, out_channels=config.feature_dim,
            kernel_size=config.kernel_size, groups=config.feature_dim, bias=config.bias)
        self.reset_parameters()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        if self.padding_left > 0: x_pad = F.pad(x_t, (self.padding_left, 0))
        else: x_pad = x_t
        y = self.conv(x_pad)
        return y.transpose(1, 2)
    def reset_parameters(self):
        pass 


@dataclass
class LinearHeadwiseExpandConfig: 
    in_features: int
    num_heads: int
    bias: bool = False
    _out_features: int = -1

    def __post_init__(self):
        if self.num_heads <= 0: raise ValueError("num_heads must be > 0")
        if self.num_heads > self.in_features: raise ValueError("num_heads must be <= in_features")
        if self.in_features % self.num_heads != 0: raise ValueError("in_features must be multiple of num_heads")
        
        if self._out_features < 0: 
            
            self._out_features = self.in_features 
        
        if self._out_features % self.num_heads != 0:
            raise ValueError(f"_out_features ({self._out_features}) must be divisible by num_heads ({self.num_heads})")


class LinearHeadwiseExpand(nn.Module):
    config_class = LinearHeadwiseExpandConfig

    def __init__(self, config: LinearHeadwiseExpandConfig):
        super().__init__()
        self.config = config

        if config.in_features % config.num_heads != 0:
            raise ValueError(f"in_features ({config.in_features}) must be divisible by num_heads ({config.num_heads})")
        
        self.head_dim_in = config.in_features // config.num_heads

        self.head_dim_out = config._out_features // config.num_heads
        
        self.weight = nn.Parameter(torch.empty(config.num_heads, self.head_dim_out, self.head_dim_in))
        
        if config.bias:
            self.bias = nn.Parameter(torch.empty(config._out_features)) #
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        
        x_reshaped = x.view(B, S, self.config.num_heads, self.head_dim_in)
        
        projected_x = torch.einsum('bshi,hoi->bsho', x_reshaped, self.weight) 
            
        out_flat = projected_x.reshape(B, S, self.config._out_features) 
        if self.bias is not None:
            out_flat = out_flat + self.bias #
            
        return out_flat

    def reset_parameters(self):
        for i in range(self.config.num_heads):
            small_init_init_(self.weight[i], dim=self.head_dim_in) 
            
        if self.bias is not None:
            nn.init.zeros_(self.bias) #

################################################################################
# Backend maths                           #
################################################################################
EPS = 1e-8 


def _prepare_ltr(S: int, device, dtype_bool: torch.dtype = torch.bool):
    return torch.tril(torch.ones((S, S), device=device, dtype=torch.float32)).to(dtype_bool)

def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor, # (B, NH, S, 1)
    fgate_preact: torch.Tensor, # (B, NH, S, 1)
    lower_triangular_matrix: Optional[torch.Tensor] = None, # (S, S) boolean
    eps: float = EPS, 
) -> torch.Tensor:
    """
   

    Shapes
    -------
    queries, keys, values : (B, NH, S, DH)
    igate_preact, fgate_preact : (B, NH, S, 1)
    """
    B, NH, S, DH = queries.shape
    dev, dtype = queries.device, queries.dtype

    # ── 1. log-forget gate matrix  ──────────────────
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)

    if lower_triangular_matrix is None or S != lower_triangular_matrix.size(-1):
        ltr = _prepare_ltr(S, dev)
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"


    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=dtype, device=dev),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    _log_fg_matrix_full = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    
    
    log_fg_matrix = torch.where(
        ltr.unsqueeze(0).unsqueeze(0), 
        _log_fg_matrix_full[:, :, 1:, 1:], 
        torch.tensor(-float("inf"), device=dev, dtype=dtype) 
    ) 

    log_D = log_fg_matrix + igate_preact.transpose(-2, -1) 

    # ── 2. scale keys/values –
    max_log_D, _ = torch.max(log_D, dim=-1, keepdim=True) 
    max_log_D_f32 = max_log_D.to(torch.float32) 

    D_matrix = torch.exp(log_D.to(torch.float32) - max_log_D_f32) # fp32

    keys_scaled = keys / math.sqrt(DH) #

    qk = torch.matmul(
        queries.to(torch.float32),
        keys_scaled.transpose(-2, -1).to(torch.float32), # Use scaled keys
    ) 

    C = qk * D_matrix  

    sum_C_abs = C.sum(dim=-1, keepdim=True).abs() 
    exp_neg_max_log_D = torch.exp(-max_log_D_f32) # fp32
    
    normaliser = torch.maximum(sum_C_abs, exp_neg_max_log_D) # fp32
    C_norm = C / (normaliser + eps) # fp32

    # ── 3. weighted value sum and cast back ────────────────────────────────────
    h_tilde = torch.matmul(C_norm, values.to(torch.float32)) # fp32
    return h_tilde.to(dtype)




def recurrent_step_stabilized_simple(
    c_prev: torch.Tensor,            # (B, NH, DH, DH)
    n_prev: torch.Tensor,            # (B, NH, DH, 1)
    m_prev: torch.Tensor,            # (B, NH, 1, 1) MUST be fp32
    q_step: torch.Tensor,            # (B, NH, 1, DH)
    k_step: torch.Tensor,            # (B, NH, 1, DH)
    v_step: torch.Tensor,            # (B, NH, 1, DH)
    igate_preact_step: torch.Tensor, # (B, NH, 1, 1)
    fgate_preact_step: torch.Tensor, # (B, NH, 1, 1)
    eps: float = EPS, # Ensure EPS is defined
):
    """Single-time-step version – MODIFIED for shape handling and fg_scaled."""
    B, NH, S_dim, DH = q_step.shape # S_dim should be 1
    dtype = q_step.dtype

    # Reshape Q, K, V as in the library's backend
    # From (B, NH, 1, DH) to (B, NH, DH, 1)
    q_proj = q_step.squeeze(2).unsqueeze(-1)
    k_proj = k_step.squeeze(2).unsqueeze(-1)
    v_proj = v_step.squeeze(2).unsqueeze(-1)

    # 1. update running log-sum-exp (m)
    m_prev_f32 = m_prev.to(torch.float32)
    
    log_fg_act_step = F.logsigmoid(fgate_preact_step) # (B, NH, 1, 1)

    m_new_f32 = torch.maximum(
        log_fg_act_step.to(torch.float32) + m_prev_f32, 
        igate_preact_step.to(torch.float32)
    ) # (B, NH, 1, 1)

    # 2. gate scalars (still fp32)
    fg_scaled = torch.exp(log_fg_act_step.to(torch.float32) + m_prev_f32 - m_new_f32)
    ig_scaled = torch.exp(igate_preact_step.to(torch.float32) - m_new_f32)

    # 3. projected key/value
    k_proj_scaled = k_proj / math.sqrt(DH) 
    kv = torch.matmul(k_proj_scaled, v_proj.transpose(-1, -2)) 

    # 4. running C and N

    c_new = fg_scaled * c_prev + ig_scaled * kv # fg_scaled and ig_scaled broadcast correctly
    n_new = fg_scaled * n_prev + ig_scaled * k_proj_scaled #

    # 5. output
    h_num = torch.matmul(q_proj.transpose(-1, -2).to(torch.float32), c_new.to(torch.float32)) # fp32
    q_n = torch.matmul(q_proj.transpose(-1, -2).to(torch.float32), n_new.to(torch.float32))   # fp32
    
    denom_f32 = torch.maximum(q_n.abs(), torch.exp(-m_new_f32)) + eps # fp32
    h_step = (h_num / denom_f32).to(dtype) # h_step is (B, NH, 1, DH)

    return h_step, (c_new, n_new, m_new_f32)



################################################################################
# mLSTM Components                                                             #
################################################################################
@dataclass
class mLSTMCellConfig:
    embedding_dim: int 
    num_heads: int     
    qkv_proj_blocksize: int 
    context_length: int
    bias: bool = False

class mLSTMCell(nn.Module):
    config_class = mLSTMCellConfig
    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config
        self.inner_dim = config.embedding_dim 
        self.num_master_heads = config.num_heads
        assert self.inner_dim % self.num_master_heads == 0, \
            f"Cell's embedding_dim ({self.inner_dim}) must be divisible by num_master_heads ({self.num_master_heads})"
        self.master_head_dim = self.inner_dim // self.num_master_heads

        self.qkv_proj_blocksize = config.qkv_proj_blocksize
        assert self.inner_dim % self.qkv_proj_blocksize == 0, \
            f"Cell's embedding_dim ({self.inner_dim}) must be divisible by qkv_proj_blocksize ({self.qkv_proj_blocksize})"
        self.num_qkv_heads = round(self.inner_dim // self.qkv_proj_blocksize)


        qkv_linear_config = LinearHeadwiseExpandConfig(
            in_features=self.inner_dim, num_heads=self.num_qkv_heads, bias=config.bias)
        self.q_proj = LinearHeadwiseExpand(qkv_linear_config)
        self.k_proj = LinearHeadwiseExpand(qkv_linear_config)
        self.v_proj = LinearHeadwiseExpand(qkv_linear_config)

        self.igate = nn.Linear(3 * self.inner_dim, self.num_master_heads, bias=True) 
        self.fgate = nn.Linear(3 * self.inner_dim, self.num_master_heads, bias=True)

        self.outnorm = MultiHeadLayerNorm(
            num_heads=self.num_master_heads, 
            head_dim=self.master_head_dim, 
            weight=True,      
            bias=False,      
            eps=1e-5          
                            
        )
        
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self): 
            self.q_proj.reset_parameters() 
            self.k_proj.reset_parameters()
            self.v_proj.reset_parameters()
            
            nn.init.zeros_(self.fgate.weight)
            if self.fgate.bias is not None:
                bias_linspace_init_(self.fgate.bias, start=3.4, end=6.0) #
                
            nn.init.zeros_(self.igate.weight)
            if self.igate.bias is not None: 
                nn.init.normal_(self.igate.bias, mean=0.0, std=0.1) 
                
            self.outnorm.reset_parameters() 

    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor, v_input: torch.Tensor) -> torch.Tensor:
        B, S, _ = q_input.shape 
        q_sub_heads = self.q_proj(q_input)
        k_sub_heads = self.k_proj(k_input)
        v_sub_heads = self.v_proj(v_input)
        q_master_S_major = q_sub_heads.reshape(B, S, self.num_master_heads, self.master_head_dim)
        k_master_S_major = k_sub_heads.reshape(B, S, self.num_master_heads, self.master_head_dim)
        v_master_S_major = v_sub_heads.reshape(B, S, self.num_master_heads, self.master_head_dim)
        q_for_backend = q_master_S_major.transpose(1, 2)
        k_for_backend = k_master_S_major.transpose(1, 2)
        v_for_backend = v_master_S_major.transpose(1, 2)
        if_gate_inp = torch.cat([q_input, k_input, v_input], dim=-1)
        igate_preact_S_major = self.igate(if_gate_inp)
        fgate_preact_S_major = self.fgate(if_gate_inp)
        igate_for_backend = igate_preact_S_major.unsqueeze(-1).transpose(1, 2)
        fgate_for_backend = fgate_preact_S_major.unsqueeze(-1).transpose(1, 2)
        current_causal_mask = self.causal_mask[:S, :S] if S <= self.causal_mask.size(0) else _prepare_ltr(S, q_input.device, torch.bool)
        h_tilde_backend = parallel_stabilized_simple(
            q_for_backend, k_for_backend, v_for_backend,
            igate_for_backend, fgate_for_backend,
            lower_triangular_matrix=current_causal_mask)
        h_norm = self.outnorm(h_tilde_backend)
        h_final = h_norm.transpose(1, 2).contiguous().view(B, S, -1)
        return h_final
    
    def step(self, q_input_step: torch.Tensor, k_input_step: torch.Tensor, v_input_step: torch.Tensor,
               mlstm_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
              ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S, _ = q_input_step.shape; assert S == 1 
        q_sub_heads = self.q_proj(q_input_step)
        k_sub_heads = self.k_proj(k_input_step)
        v_sub_heads = self.v_proj(v_input_step)
        q_master_S_major = q_sub_heads.reshape(B, S, self.num_master_heads, self.master_head_dim)
        k_master_S_major = k_sub_heads.reshape(B, S, self.num_master_heads, self.master_head_dim)
        v_master_S_major = v_sub_heads.reshape(B, S, self.num_master_heads, self.master_head_dim)
        q_for_backend = q_master_S_major.transpose(1, 2)
        k_for_backend = k_master_S_major.transpose(1, 2)
        v_for_backend = v_master_S_major.transpose(1, 2)
        if_gate_inp = torch.cat([q_input_step, k_input_step, v_input_step], dim=-1)
        igate_preact_S_major = self.igate(if_gate_inp)
        fgate_preact_S_major = self.fgate(if_gate_inp)
        igate_for_backend = igate_preact_S_major.unsqueeze(-1).transpose(1, 2)
        fgate_for_backend = fgate_preact_S_major.unsqueeze(-1).transpose(1, 2)

        if mlstm_state is None:
            current_dtype_dev = q_for_backend.device
            current_dtype_val = q_for_backend.dtype
            c_prev = torch.zeros(B, self.num_master_heads, self.master_head_dim, self.master_head_dim, 
                                 device=current_dtype_dev, dtype=current_dtype_val)
            n_prev = torch.zeros(B, self.num_master_heads, self.master_head_dim, 1,
                                 device=current_dtype_dev, dtype=current_dtype_val)
            m_prev = torch.zeros(B, self.num_master_heads, 1, 1,
                                 device=current_dtype_dev, dtype=torch.float32)
        else: 
            c_prev, n_prev, m_prev = mlstm_state
            if m_prev.dtype != torch.float32:
                m_prev = m_prev.to(torch.float32)

        h_tilde_backend, new_state = recurrent_step_stabilized_simple(
            c_prev, n_prev, m_prev, 
            q_for_backend, k_for_backend, v_for_backend,
            igate_for_backend, fgate_for_backend) 
        
        h_norm = self.outnorm(h_tilde_backend)
        h_final_step = h_norm.transpose(1, 2).contiguous().view(B, S, -1)
        return h_final_step, new_state

@dataclass
class mLSTMLayerConfig:
    embedding_dim: int; context_length: int; num_heads: int = 4; conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4; proj_factor: float = 2.0; bias: bool = False 
    dropout: float = 0.0; _num_blocks: int = 1 





class mLSTMLayer(nn.Module):
    config_class = mLSTMLayerConfig

    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        raw_inner = config.embedding_dim * config.proj_factor
        self.inner_dim = int(round(raw_inner / 8.0)) * 8
        inner = self.inner_dim
        self.embedding_dim = config.embedding_dim

        self.proj_up = nn.Linear(self.embedding_dim, 2 * inner, bias=config.bias)


        self.conv1d = CausalConv1d(CausalConv1dConfig(
            feature_dim=inner,
            kernel_size=config.conv1d_kernel_size,
            bias=config.bias, 
        ))
        self.conv_act = nn.SiLU() 

        self.mlstm_cell = mLSTMCell(
            mLSTMCellConfig( 
                embedding_dim=inner, 
                num_heads=config.num_heads,
                qkv_proj_blocksize=config.qkv_proj_blocksize, 
                context_length=config.context_length,
                bias=config.bias, 
            )
        )

        self.proj_down = nn.Linear(inner, self.embedding_dim, bias=config.bias)
        self.dropout_layer = nn.Dropout(config.dropout)

        self.learnable_skip = nn.Parameter(torch.ones(inner)) # Vector of size inner_dim
        
        self.ogate_act = nn.SiLU() 
    def reset_parameters(self): 
        small_init_init_(self.proj_up.weight, dim=self.embedding_dim)
        if self.config.bias and self.proj_up.bias is not None: nn.init.zeros_(self.proj_up.bias)
        
  
        
        self.mlstm_cell.reset_parameters() # If mLSTMCell has it
        
   

        wang_init_(self.proj_down.weight, dim=self.inner_dim, num_blocks=self.config._num_blocks)
        if self.config.bias and self.proj_down.bias is not None: nn.init.zeros_(self.proj_down.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_projected = self.proj_up(x)
        x_for_mlstm_path, z_for_ogate = torch.split(x_projected, self.inner_dim, dim=-1)
        
        x_conv = self.conv1d(x_for_mlstm_path)
        x_conv_act = self.conv_act(x_conv) 
        
        h_tilde = self.mlstm_cell(q_input=x_conv_act, k_input=x_conv_act, v_input=x_for_mlstm_path)
        
 
        h_skip = h_tilde + self.learnable_skip * x_conv_act 
        

        h_gated = h_skip * self.ogate_act(z_for_ogate) 
        
        output = self.proj_down(h_gated)
        return self.dropout_layer(output)

    def step(self, x_step: torch.Tensor, mlstm_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:


        x_projected = self.proj_up(x_step)
        x_for_mlstm_path, z_for_ogate = torch.split(x_projected, self.inner_dim, dim=-1)
        
    
        x_conv = self.conv1d(x_for_mlstm_path) 
        x_conv_act = self.conv_act(x_conv)
        
        h_tilde_step, new_mlstm_state = self.mlstm_cell.step(
            q_input_step=x_conv_act, k_input_step=x_conv_act, v_input_step=x_for_mlstm_path,
            mlstm_state=mlstm_state) 
        
        h_skip = h_tilde_step + self.learnable_skip * x_conv_act
        h_gated = h_skip * self.ogate_act(z_for_ogate)
        
        output_step = self.proj_down(h_gated)
   
        return self.dropout_layer(output_step), new_mlstm_state
    
@dataclass
class mLSTMBlockConfig:
    mlstm_layer_config: mLSTMLayerConfig
class mLSTMBlock(nn.Module):
    config_class = mLSTMBlockConfig
    def __init__(self, config: mLSTMBlockConfig):
        super().__init__()
        self.config = config
        self.ln = LayerNorm(config.mlstm_layer_config.embedding_dim, weight=True, bias=False)
        self.core = mLSTMLayer(config.mlstm_layer_config)
        self.reset_parameters()
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor: # Added **kwargs for compatibility
        return x + self.core(self.ln(x))
    def reset_parameters(self): 
        self.ln.reset_parameters()
        self.core.reset_parameters()
    def step(self, x_step: torch.Tensor, state: Optional[Dict[str, Tuple]] = None, **kwargs):
        mlstm_layer_state_val = state.get("mlstm_layer_state") if state else None
        processed_x, new_mlstm_core_state = self.core.step(self.ln(x_step), mlstm_state=mlstm_layer_state_val)
        output_x = x_step + processed_x
        return output_x, {"mlstm_layer_state": new_mlstm_core_state}

###############################################################################
# sLSTM Components                          #
###############################################################################

def powerlaw_blockdependent_init_(bias_tensor: torch.Tensor, head_dim: int, block_idx: int, num_blocks: int):
    """
    Initializes the bias tensor using the powerlaw block-dependent scheme.
    The input bias_tensor is typically for one gate, shaped (num_heads, head_dim).
    """
    with torch.no_grad():
        if num_blocks > 1:
            ratio_0_to_1 = block_idx / (num_blocks - 1.0)
        else:
            ratio_0_to_1 = 0.0
        
        if head_dim == 1:
            term_val = 0.0 
        else:
            term_val = (torch.arange(head_dim, device=bias_tensor.device, dtype=bias_tensor.dtype) / (head_dim - 1.0))
            
        exponent = 0.3 + 1.3 * ratio_0_to_1
        init_values_raw = -(-5.0 + 12.0 * (term_val ** exponent))
        
        bias_tensor.copy_(init_values_raw.unsqueeze(0).expand_as(bias_tensor))
    return bias_tensor

@dataclass
class sLSTMCellConfig:
    embedding_dim: int      
    num_heads: int = 4
    bias: bool = True 
    recurrent_bias: bool = False 
    recurrent_transform_weight_init_std: Optional[float] = None 
    _block_idx: int = 0
    _num_blocks: int = 1

    @property
    def head_dim(self) -> int:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        return self.embedding_dim // self.num_heads

    def __post_init__(self):
        _ = self.head_dim


class sLSTMCell(nn.Module):
    config_class = sLSTMCellConfig
    def __init__(self, cfg: sLSTMCellConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        lhe_config_internal = LinearHeadwiseExpandConfig(
            in_features=cfg.embedding_dim, 
            num_heads=cfg.num_heads,
            bias=cfg.recurrent_bias 
        )
        self.gate_transforms = nn.ModuleList([
            LinearHeadwiseExpand(deepcopy(lhe_config_internal)) for _ in range(4) # z, i, f, o
        ])

        if cfg.bias: 
            self.gate_biases = nn.Parameter(torch.empty(4, cfg.num_heads, self.head_dim))
        else:
            self.register_parameter('gate_biases', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(4):
            transform_layer = self.gate_transforms[i]
            if self.cfg.recurrent_transform_weight_init_std is not None:
                std = self.cfg.recurrent_transform_weight_init_std
                if std == 0: 
                    with torch.no_grad():
                        transform_layer.weight.data.fill_(0.0)
                elif std > 0: 
                    with torch.no_grad():
                        for h_idx in range(self.cfg.num_heads):
                            transform_layer.weight.data[h_idx].normal_(mean=0.0, std=std)
                
                if self.cfg.recurrent_bias and transform_layer.bias is not None:
                    nn.init.zeros_(transform_layer.bias.data)
            else:
                transform_layer.reset_parameters()

        # Initialize final additive gate_biases
        if self.cfg.bias and self.gate_biases is not None:
            nn.init.zeros_(self.gate_biases[0, ...]) # z_bias
            nn.init.zeros_(self.gate_biases[1, ...]) # i_bias
            nn.init.zeros_(self.gate_biases[3, ...]) # o_bias
            
            powerlaw_blockdependent_init_(self.gate_biases[2, ...], 
                                          self.head_dim, 
                                          self.cfg._block_idx, 
                                          self.cfg._num_blocks)
    
    def _zeros_state(self, B: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_shape = (B, self.cfg.num_heads, self.head_dim)
        h0 = torch.zeros(state_shape, device=device, dtype=dtype)
        c0 = torch.zeros_like(h0)
        n0 = torch.zeros_like(h0)
        m0 = torch.zeros_like(h0, dtype=torch.float32) 
        return h0, c0, n0, m0

    def forward(self, x_gate_inputs: torch.Tensor, # (B, S, D_model * 4)
                slstm_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S, D_quad_model = x_gate_inputs.shape
        assert D_quad_model == self.cfg.embedding_dim * 4, "Input dimension mismatch"

        projs_from_layer = torch.split(x_gate_inputs, self.cfg.embedding_dim, dim=-1)
        
        transformed_signals_list = [
            self.gate_transforms[i](projs_from_layer[i]) for i in range(4)
        ]
        

        z_transformed = transformed_signals_list[0].view(B, S, self.cfg.num_heads, self.head_dim)
        i_transformed = transformed_signals_list[1].view(B, S, self.cfg.num_heads, self.head_dim)
        f_transformed = transformed_signals_list[2].view(B, S, self.cfg.num_heads, self.head_dim)
        o_transformed = transformed_signals_list[3].view(B, S, self.cfg.num_heads, self.head_dim)

        if slstm_cell_state is None:
            h_prev, c_prev, n_prev, m_prev = self._zeros_state(B, x_gate_inputs.device, x_gate_inputs.dtype)
        else:
            h_prev, c_prev, n_prev, m_prev = slstm_cell_state
            if m_prev.dtype != torch.float32: m_prev = m_prev.to(torch.float32)

        outs_h = [] 

        for t in range(S):
            z_t_curr = z_transformed[:, t, :, :]
            i_t_curr = i_transformed[:, t, :, :]
            f_t_curr = f_transformed[:, t, :, :]
            o_t_curr = o_transformed[:, t, :, :]

            if self.cfg.bias and self.gate_biases is not None:
                z_biased = z_t_curr + self.gate_biases[0].unsqueeze(0) 
                i_biased = i_t_curr + self.gate_biases[1].unsqueeze(0)
                f_biased = f_t_curr + self.gate_biases[2].unsqueeze(0)
                o_biased = o_t_curr + self.gate_biases[3].unsqueeze(0)
            else: 
                z_biased = z_t_curr
                i_biased = i_t_curr
                f_biased = f_t_curr
                o_biased = o_t_curr
            
            z_val = torch.tanh(z_biased)
            i_val = torch.exp(i_biased) 
            f_val = torch.exp(f_biased) 
            o_val = torch.sigmoid(o_biased)

            m_prev_fp32 = m_prev.to(torch.float32)
            log_f_val_fp32 = torch.log(f_val.to(torch.float32) + EPS) 
            log_i_val_fp32 = torch.log(i_val.to(torch.float32) + EPS)

            m_new_fp32 = torch.maximum(log_f_val_fp32 + m_prev_fp32, log_i_val_fp32)


            i_hat = torch.exp(log_i_val_fp32 - m_new_fp32).to(h_prev.dtype)
            f_hat = torch.exp(log_f_val_fp32 + m_prev_fp32 - m_new_fp32).to(h_prev.dtype)
            
            c_new = f_hat * c_prev + i_hat * z_val
            n_new = f_hat * n_prev + i_hat
            h_new = o_val * (c_new / (n_new + EPS))

            outs_h.append(h_new)
            h_prev, c_prev, n_prev, m_prev = h_new, c_new, n_new, m_new_fp32

        final_h_sequence = torch.stack(outs_h, dim=1) # (B, S, NH, DH)
        final_state_tuple = (h_prev, c_prev, n_prev, m_prev)
        
        return final_h_sequence, final_state_tuple

    def step(self, x_gate_inputs_step: torch.Tensor, # (B, 1, D_model * 4)
               slstm_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        h_sequence, final_state = self.forward(x_gate_inputs_step, slstm_cell_state=slstm_cell_state)
        return h_sequence, final_state


@dataclass
class sLSTMLayerConfig:
    embedding_dim: int
    context_length: int 
    num_heads: int = 4
    conv1d_kernel_size: int = 0 
    bias: bool = False 
    dropout: float = 0.0
    group_norm_weight: bool = True
    
    cell_final_bias: bool = True          
    cell_recurrent_transform_bias: bool = False 
    cell_recurrent_transform_weight_init_std: Optional[float] = None 

    _block_idx: int = 0
    _num_blocks: int = 1

    @property
    def head_dim(self) -> int: 
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        return self.embedding_dim // self.num_heads

    def __post_init__(self):
        _ = self.head_dim 

class sLSTMLayer(nn.Module):
    config_class = sLSTMLayerConfig
    def __init__(self, cfg: sLSTMLayerConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(CausalConv1dConfig(
                feature_dim=cfg.embedding_dim,
                kernel_size=cfg.conv1d_kernel_size,
                bias=cfg.bias
            ))
            self.conv_act = nn.SiLU()
        else:
            self.conv1d = None
            self.conv_act = None

        proj_cfg = LinearHeadwiseExpandConfig(
            in_features=cfg.embedding_dim, 
            num_heads=cfg.num_heads, 
            bias=cfg.bias 
        )
        self.z_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))
        self.i_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))
        self.f_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))
        self.o_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))

        
        cell_cfg_obj = sLSTMCellConfig(
            embedding_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            bias=cfg.cell_final_bias, 
            recurrent_bias=cfg.cell_recurrent_transform_bias,
            recurrent_transform_weight_init_std=cfg.cell_recurrent_transform_weight_init_std,
            _block_idx=cfg._block_idx,
            _num_blocks=cfg._num_blocks
        )
        self.cell = sLSTMCell(cell_cfg_obj)
        
        self.dropout_layer = nn.Dropout(cfg.dropout)
        
        head_dim_for_norm = cfg.embedding_dim // cfg.num_heads
        self.group_norm = MultiHeadLayerNorm(
            num_heads=cfg.num_heads,
            head_dim=head_dim_for_norm,
            weight=cfg.group_norm_weight,
            bias=False 
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv1d is not None:
            self.conv1d.reset_parameters()

        self.z_gate_proj.reset_parameters()
        self.i_gate_proj.reset_parameters()
        self.f_gate_proj.reset_parameters()
        self.o_gate_proj.reset_parameters()
        
        self.cell.reset_parameters()
        self.group_norm.reset_parameters()

    def forward(self, x: torch.Tensor, 
                slstm_cell_state_init: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
               ) -> torch.Tensor: 
        B, S, D_embed = x.shape

        if self.conv1d is not None and self.conv_act is not None:
            x_conv = self.conv1d(x)
            x_conv_activated = self.conv_act(x_conv)
        else:
            x_conv_activated = x 

        # Projections from this layer: i, f from x_conv_activated; z, o from x
        i_projected = self.i_gate_proj(x_conv_activated)
        f_projected = self.f_gate_proj(x_conv_activated)
        z_projected = self.z_gate_proj(x)
        o_projected = self.o_gate_proj(x)

        # Concatenate gate inputs for the cell: (B, S, D_model * 4)
        # Order is z, i, f, o, which sLSTMCell expects for its gate_transforms and gate_biases
        gate_inputs_concat = torch.cat([z_projected, i_projected, f_projected, o_projected], dim=-1)
        
        # Cell processes concatenated inputs. Output h_cell_output is (B, S, NH, DH)
        h_cell_output, _ = self.cell(gate_inputs_concat, slstm_cell_state=slstm_cell_state_init)
        
        h_dropped = self.dropout_layer(h_cell_output) # Still (B, S, NH, DH)
        
        # Group norm expects (B, NH, S, DH)
        h_norm_input = h_dropped.transpose(1, 2) 
        h_norm_output = self.group_norm(h_norm_input) 
        
        # Transpose back and reshape to (B, S, D_layer_embed)
        h_final = h_norm_output.transpose(1, 2).contiguous()
        return h_final.view(B, S, D_embed)

    def step(self, x_step: torch.Tensor, # (B, 1, D_layer_embed)
               slstm_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
              ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S_one, D_embed = x_step.shape
        assert S_one == 1, "Step method expects sequence length of 1"

        if self.conv1d is not None and self.conv_act is not None:
            x_conv_step = self.conv1d(x_step) 
            x_conv_activated_step = self.conv_act(x_conv_step)
        else:
            x_conv_activated_step = x_step

        i_projected_step = self.i_gate_proj(x_conv_activated_step)
        f_projected_step = self.f_gate_proj(x_conv_activated_step)
        z_projected_step = self.z_gate_proj(x_step)
        o_projected_step = self.o_gate_proj(x_step)

        gate_inputs_concat_step = torch.cat([z_projected_step, i_projected_step, f_projected_step, o_projected_step], dim=-1)
        
        h_cell_step_output, new_slstm_cell_state_tuple = self.cell.step(
            gate_inputs_concat_step, slstm_cell_state=slstm_cell_state
        )
        
        h_dropped_step = self.dropout_layer(h_cell_step_output) # (B, 1, NH, DH)
        
        h_norm_input_step = h_dropped_step.transpose(1, 2) # (B, NH, 1, DH)
        h_norm_output_step = self.group_norm(h_norm_input_step) # (B, NH, 1, DH)
        
        h_final_step = h_norm_output_step.transpose(1, 2).contiguous()
        output_step = h_final_step.view(B, S_one, D_embed) # (B, 1, D_embed)
        
        return output_step, new_slstm_cell_state_tuple


@dataclass
class sLSTMBlockConfig:
    slstm_layer_config: sLSTMLayerConfig 
    feedforward_config: Optional[FeedForwardConfig] = None
    _block_idx: int = 0 
    _num_blocks: int = 1 

    def __post_init__(self):
        if self.slstm_layer_config is not None:
            self.slstm_layer_config._block_idx = self._block_idx
            self.slstm_layer_config._num_blocks = self._num_blocks

class sLSTMBlock(nn.Module):
    config_class = sLSTMBlockConfig
    def __init__(self, config: sLSTMBlockConfig):
        super().__init__()
        self.config = config
        
        block_embedding_dim = config.slstm_layer_config.embedding_dim
        
        self.ln1 = LayerNorm(block_embedding_dim, weight=True, bias=False) 
        self.core = sLSTMLayer(config.slstm_layer_config) 
        
        if config.feedforward_config is not None:
            self.ln2 = LayerNorm(block_embedding_dim, weight=True, bias=False) 
            self.ffn = FeedForward(config.feedforward_config, embedding_dim=block_embedding_dim)
        else:
            self.ln2 = None
            self.ffn = None
        self.reset_parameters()

    def reset_parameters(self):
        self.ln1.reset_parameters()
        self.core.reset_parameters()
        if self.ffn is not None and self.ln2 is not None:
            self.ln2.reset_parameters()
            self.ffn.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        identity = x
        x = self.ln1(x)
        x = self.core(x) # sLSTMLayer.forward
        x = identity + x
        
        if self.ffn is not None and self.ln2 is not None:
            identity_ffn = x
            x = self.ln2(x)
            x = self.ffn(x)
            x = identity_ffn + x
        return x

    def step(self, x_step: torch.Tensor, state: Optional[Dict[str, Tuple]] = None, **kwargs):
        # state is expected to be like {"slstm_layer_state": (h,c,n,m)} or None
        # where (h,c,n,m) is the tuple for sLSTMCell's state.
        slstm_core_state_val = state.get("slstm_layer_state") if state else None
        
        identity_step = x_step
        x_ln1_step = self.ln1(x_step)
        
        core_out_step, new_slstm_core_state_tuple = self.core.step(
            x_ln1_step, slstm_cell_state=slstm_core_state_val
        )
        res_x_step = identity_step + core_out_step

        if self.ffn is not None and self.ln2 is not None:
            identity_ffn_step = res_x_step
            x_ln2_step = self.ln2(res_x_step)
            ffn_out_step = self.ffn(x_ln2_step) 
            final_x_step = identity_ffn_step + ffn_out_step
        else:
            final_x_step = res_x_step
            
        # The new state dict should match the expected input format for the next step
        new_state_dict = {"slstm_layer_state": new_slstm_core_state_tuple}
        return final_x_step, new_state_dict

@dataclass
class FeedForwardConfig:
    proj_factor: float = 1.0 
    act_fn: str = "gelu"    # Activation function name 
    dropout: float = 0.0
    bias: bool = False      # Bias for linear layers

class FeedForward(nn.Module):
    config_class = FeedForwardConfig
    def __init__(self, config: FeedForwardConfig, embedding_dim: int): # embedding_dim is D_model for FFN
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        inner_dim = int(config.proj_factor * embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim, inner_dim, bias=config.bias)
        
        if config.act_fn == "gelu": self.act = nn.GELU()
        elif config.act_fn == "silu": self.act = nn.SiLU()
        else: self.act = getattr(F, config.act_fn) # Fallback for other F.activations
            
        self.fc2 = nn.Linear(inner_dim, embedding_dim, bias=config.bias)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        small_init_init_(self.fc1.weight, dim=self.embedding_dim)
        if self.config.bias and self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        
        inner_dim = int(self.config.proj_factor * self.embedding_dim)
        small_init_init_(self.fc2.weight, dim=inner_dim)
        if self.config.bias and self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout_layer(x)




################################################################################
# xLSTM Block Stack (Combined mLSTM and sLSTM)                                 #
################################################################################

@dataclass
class xLSTMBlockStackConfig:
    # Configurations for the block types. If a config is None, that block type won't be used.
    mlstm_block_template: Optional[mLSTMBlockConfig] = None
    slstm_block_template: Optional[sLSTMBlockConfig] = None

    # Overall stack parameters
    num_blocks: int = 6
    embedding_dim: int = 512
    context_length: int = 96 # Used by mLSTM and sLSTM layer configs
    
    # Parameters to be propagated to block configs if they are not None
    bias: bool = False
    dropout: float = 0.0
    
    add_post_blocks_norm: bool = True

    # Specifies the indices at which sLSTM blocks are placed. Other indices use mLSTM.
    # Indexing starts from 0. If 'mlstm_block_template' is None, all blocks must be sLSTM.
    # If 'slstm_block_template' is None, all blocks must be mLSTM.
    slstm_at: Union[List[int], Literal["all"], Literal["none"]] = field(default_factory=list)

    # Internal map: 0 for mLSTM, 1 for sLSTM
    _block_map_str: str = field(init=False, repr=False) # Renamed from _block_map

    @property
    def block_map(self) -> List[int]:
        return list(map(int, self._block_map_str.split(",")))

    def _create_block_map_str(self) -> str:
        """Creates the block map string, specifying which block type is used at which position."""
        if self.slstm_at == "all":
            if self.slstm_block_template is None:
                raise ValueError("sLSTM blocks specified ('all') but 'slstm_block_template' is None.")
            if self.mlstm_block_template is not None: # If mLSTM also available, "all" means all are sLSTM
                 resolved_slstm_at = list(range(self.num_blocks))
            else: # Only sLSTM is available
                 resolved_slstm_at = list(range(self.num_blocks))

        elif self.slstm_at == "none":
            resolved_slstm_at = []
        else: # List of indices
            resolved_slstm_at = self.slstm_at

        block_map_list = [0] * self.num_blocks # Default to mLSTM
        for slstm_idx in resolved_slstm_at:
            if not (0 <= slstm_idx < self.num_blocks):
                raise ValueError(f"Invalid sLSTM position index {slstm_idx} for {self.num_blocks} blocks.")
            if self.slstm_block_template is None:
                raise ValueError(f"sLSTM block specified at index {slstm_idx} but 'slstm_block_template' is None.")
            block_map_list[slstm_idx] = 1 # Set to sLSTM

        # Validate that if a block type is 0 (mLSTM), mlstm_block_template is available
        if any(b_type == 0 for b_type in block_map_list) and self.mlstm_block_template is None:
            raise ValueError("mLSTM block type inferred in block_map, but 'mlstm_block_template' is None.")
        
        return ",".join(map(str, block_map_list))

    def __post_init__(self):
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        if self.mlstm_block_template is None and self.slstm_block_template is None:
            raise ValueError("At least one of 'mlstm_block_template' or 'slstm_block_template' must be provided.")

        # Determine default if slstm_at is empty and both templates are available
        if not self.slstm_at and isinstance(self.slstm_at, list): 
            if self.mlstm_block_template is not None and self.slstm_block_template is None:
                self.slstm_at = "none"
            elif self.mlstm_block_template is None and self.slstm_block_template is not None:
                self.slstm_at = "all" 
        if self.mlstm_block_template is not None:
            layer_cfg = self.mlstm_block_template.mlstm_layer_config
            layer_cfg.embedding_dim = self.embedding_dim
            layer_cfg.context_length = self.context_length
            layer_cfg.bias = self.bias
            layer_cfg.dropout = self.dropout
            layer_cfg._num_blocks = self.num_blocks 
        if self.slstm_block_template is not None:
            layer_cfg = self.slstm_block_template.slstm_layer_config
            layer_cfg.embedding_dim = self.embedding_dim
            layer_cfg.context_length = self.context_length 
            layer_cfg.bias = self.bias
            layer_cfg.dropout = self.dropout
            layer_cfg._num_blocks = self.num_blocks
            if self.slstm_block_template.feedforward_config is not None:
                ffn_cfg = self.slstm_block_template.feedforward_config
                ffn_cfg.dropout = self.dropout 
                ffn_cfg.bias = self.bias       

        self._block_map_str = self._create_block_map_str()


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config
        self.blocks = self._create_blocks()

        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(config.embedding_dim, weight=True, bias=True)
        else:
            self.post_blocks_norm = nn.Identity()
        
        self.reset_parameters() # Initialize weights after all modules are created

    def _create_blocks(self) -> nn.ModuleList:
        module_blocks = []
        for block_type_int in self.config.block_map:
            if block_type_int == 0: # mLSTMBlock
                if self.config.mlstm_block_template is None:
                    raise RuntimeError("Trying to create mLSTMBlock but mlstm_block_template is None.")

                block_cfg = deepcopy(self.config.mlstm_block_template)
                module_blocks.append(mLSTMBlock(config=block_cfg))
            elif block_type_int == 1: # sLSTMBlock
                if self.config.slstm_block_template is None:
                    raise RuntimeError("Trying to create sLSTMBlock but slstm_block_template is None.")
                block_cfg = deepcopy(self.config.slstm_block_template)
                module_blocks.append(sLSTMBlock(config=block_cfg))
            else:
                raise ValueError(f"Invalid block type {block_type_int} in block_map.")
        return nn.ModuleList(module_blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if isinstance(self.post_blocks_norm, LayerNorm):
            self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, **kwargs) 
        x = self.post_blocks_norm(x)
        return x

    def step(self, x_step: torch.Tensor, states: Optional[List[Optional[Dict[str, Tuple]]]] = None
            ) -> Tuple[torch.Tensor, List[Optional[Dict[str, Tuple]]]]:
        
        if states is None:
            states = [None] * len(self.blocks)
        
        if len(states) != len(self.blocks):
            raise ValueError(f"Number of states ({len(states)}) must match number of blocks ({len(self.blocks)}).")

        new_states_list: List[Optional[Dict[str, Tuple]]] = []
        current_x_step = x_step

        for i, block in enumerate(self.blocks):

            block_input_state_dict = states[i] 
            
            current_x_step, returned_block_state_dict = block.step(current_x_step, state=block_input_state_dict)
            new_states_list.append(returned_block_state_dict)
            
        current_x_step = self.post_blocks_norm(current_x_step)
        return current_x_step, new_states_list