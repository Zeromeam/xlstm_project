# -*- coding: utf-8 -*-
"""
Self‑contained re‑implementation of the **xLSTM** building blocks.
Based on xlstm_replica_old.py, with added stabilization toggles 
for experimentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Literal, List, Dict # Added Union, Literal, List, Dict
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy # Added deepcopy

# Helper initialisation utilities: bias_linspace_init_, small_init_init_, wang_init_ (FROM OLD FILE)
def bias_linspace_init_(bias: torch.Tensor, *, start: float = 3.4, end: float = 6.0): # OLD DEFAULT
    with torch.no_grad():
        source_tensor = torch.linspace(start, end, bias.numel(), dtype=bias.dtype, device=bias.device)
        bias.copy_(source_tensor.reshape(bias.shape))
    return bias

def small_init_init_(weight: torch.Tensor, *, dim: int): # OLD DEFAULT
    """Kaiming‑style *small* initialisation used in the official code.
    Matches Transformers without Tears: Improving the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019)
    and EleutherAI/gpt-neox.
    """
    std = math.sqrt(2 / (5 * dim)) # OLD DEFAULT
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
    return weight

def wang_init_(weight: torch.Tensor, *, dim: int, num_blocks: int): # OLD DEFAULT
    """Scaled Xavier initialisation from the paper appendix (Wang et al.).
    Adopted from EleutherAI/gpt-neox.
    """
    safe_num_blocks = max(1, num_blocks)
    std = (2 / safe_num_blocks) / math.sqrt(dim) # OLD DEFAULT
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
    return weight

# LayerNorm, MultiHeadLayerNorm, CausalConv1d, LinearHeadwiseExpand (FROM OLD FILE)
class LayerNorm(nn.LayerNorm): # FROM OLD
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

class MultiHeadLayerNorm(nn.Module): # FROM OLD
    def __init__(self,
                 num_heads: int,       # NH
                 head_dim: int,        # DH
                 *,
                 weight: bool = True,
                 bias: bool = False,
                 eps: float = 1e-5, # Old default
                 residual_weight: bool = True):
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
        B, NH_in, S, DH_in = x.shape
        assert NH_in == self.num_heads
        assert DH_in == self.head_dim
        gn_in_1 = x.transpose(1, 2)
        gn_in_2 = gn_in_1.reshape(B * S, self.num_heads * self.head_dim)
        out_gn = F.group_norm(
            gn_in_2,
            num_groups=self.num_heads,
            weight=self.weight_proxy,
            bias=self.affine_bias,
            eps=self.eps,
        )
        output = out_gn.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return output

    def reset_parameters(self):
        if self.affine_weight is not None:
            nn.init.zeros_(self.affine_weight)
        if self.affine_bias is not None:
            nn.init.zeros_(self.affine_bias)

@dataclass
class CausalConv1dConfig: # FROM OLD (Identical to new)
    feature_dim: int; kernel_size: int = 4; bias: bool = False
class CausalConv1d(nn.Module): # FROM OLD (Identical to new apart from reset_parameters content)
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
    def reset_parameters(self): # Old version was pass
        pass

@dataclass
class LinearHeadwiseExpandConfig: # FROM OLD
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

class LinearHeadwiseExpand(nn.Module): # FROM OLD
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
            self.bias = nn.Parameter(torch.empty(config._out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        x_reshaped = x.view(B, S, self.config.num_heads, self.head_dim_in)
        projected_x = torch.einsum('bshi,hoi->bsho', x_reshaped, self.weight)
        out_flat = projected_x.reshape(B, S, self.config._out_features)
        if self.bias is not None:
            out_flat = out_flat + self.bias
        return out_flat

    def reset_parameters(self):
        for i in range(self.config.num_heads):
            small_init_init_(self.weight[i], dim=self.head_dim_in)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

################################################################################
# Backend maths – MODIFIED with toggles                                        #
################################################################################
EPS = 1e-8 # FROM OLD FILE

def _prepare_ltr(S: int, device, dtype_bool: torch.dtype = torch.bool):
    return torch.tril(torch.ones((S, S), device=device, dtype=torch.float32)).to(dtype_bool)

def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: Optional[torch.Tensor] = None,
    eps: float = EPS,
    # --- Experimental Toggles ---
    use_max_log_d_subtraction: bool = True,
    max_log_d_computation_dtype: Optional[torch.dtype] = torch.float32, # Old behavior was float32
    use_robust_normalizer_parallel: bool = True,
    use_eps_in_parallel_norm_denom: bool = True, # Old behavior effectively had eps
) -> torch.Tensor:
    B, NH, S, DH = queries.shape
    dev, dtype = queries.device, queries.dtype

    log_fgates = F.logsigmoid(fgate_preact)

    if lower_triangular_matrix is None or S > lower_triangular_matrix.size(-1): # Adjusted S condition from old
        ltr = _prepare_ltr(S, dev, torch.bool) # Ensure boolean
    else:
        ltr = lower_triangular_matrix[:S,:S].to(torch.bool) # Ensure boolean and correct slicing
    
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"


    log_fgates_cumsum = torch.cat(
        [torch.zeros((B, NH, 1, 1), dtype=log_fgates.dtype, device=dev), # use log_fgates.dtype
         torch.cumsum(log_fgates, dim=-2)], dim=-2,
    )
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)
    _log_fg_matrix_full = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)
    
    log_fg_matrix = torch.where(
        ltr.unsqueeze(0).unsqueeze(0),
        _log_fg_matrix_full[:, :, 1:, 1:],
        torch.tensor(-float("inf"), device=dev, dtype=log_fgates.dtype) # use log_fgates.dtype
    )

    log_D = log_fg_matrix + igate_preact.transpose(-2, -1)

    # Determine dtype for max_log_D computation
    _max_log_d_dtype = max_log_d_computation_dtype if max_log_d_computation_dtype is not None else log_D.dtype
    
    max_log_D_uncasted, _ = torch.max(log_D, dim=-1, keepdim=True)
    max_log_D_casted = max_log_D_uncasted.to(_max_log_d_dtype)

    # Determine dtype for D_matrix and QK computation (mimic old file's float32 usage)
    # This can be considered part of max_log_d_computation_dtype's role
    _computation_dtype_for_exp = _max_log_d_dtype

    if use_max_log_d_subtraction:
        D_matrix = torch.exp(log_D.to(_computation_dtype_for_exp) - max_log_D_casted)
    else:
        D_matrix = torch.exp(log_D.to(_computation_dtype_for_exp))
    
    D_matrix = D_matrix.to(dtype) # Cast back to original type if intermediate was different for exp

    keys_scaled = keys / math.sqrt(DH)

    # Use a consistent computation type for QK and C, then cast back
    # Old file used float32 for qk, C, sum_C_abs, normaliser, C_norm, h_tilde
    # Let's make this configurable or stick to _computation_dtype_for_exp
    _matmult_dtype = _computation_dtype_for_exp

    qk = torch.matmul(
        queries.to(_matmult_dtype),
        keys_scaled.transpose(-2, -1).to(_matmult_dtype),
    )
    C = qk * D_matrix.to(_matmult_dtype) # D_matrix might have been casted back

    sum_C_abs = C.sum(dim=-1, keepdim=True).abs()
    exp_neg_max_log_D = torch.exp(-max_log_D_casted)

    if use_robust_normalizer_parallel:
        normaliser = torch.maximum(sum_C_abs, exp_neg_max_log_D)
    else:
        normaliser = sum_C_abs # Less robust

    if use_eps_in_parallel_norm_denom:
        C_norm = C / (normaliser + eps)
    else:
        C_norm = C / normaliser # Potentially unstable

    h_tilde = torch.matmul(C_norm, values.to(_matmult_dtype))
    return h_tilde.to(dtype)


def recurrent_step_stabilized_simple(
    c_prev: torch.Tensor,
    n_prev: torch.Tensor,
    m_prev: torch.Tensor, # Original m_prev dtype (could be float32 or input dtype)
    q_step: torch.Tensor,
    k_step: torch.Tensor,
    v_step: torch.Tensor,
    igate_preact_step: torch.Tensor,
    fgate_preact_step: torch.Tensor,
    eps: float = EPS,
    # --- Experimental Toggles ---
    m_computation_dtype_recurrent: Optional[torch.dtype] = torch.float32, # Old behavior was float32
    use_robust_m_new_recurrent: bool = True,
    use_m_new_subtraction_recurrent_gates: bool = True,
    use_robust_h_denom_recurrent: bool = True,
    use_eps_in_recurrent_h_denom: bool = True, # Old behavior had eps
):
    B, NH, S_dim, DH = q_step.shape
    input_dtype = q_step.dtype

    q_proj = q_step.squeeze(2).unsqueeze(-1)
    k_proj = k_step.squeeze(2).unsqueeze(-1)
    v_proj = v_step.squeeze(2).unsqueeze(-1)

    _m_dtype = m_computation_dtype_recurrent if m_computation_dtype_recurrent is not None else m_prev.dtype
    m_prev_casted = m_prev.to(_m_dtype)
    
    log_fg_act_step = F.logsigmoid(fgate_preact_step).to(_m_dtype)
    igate_preact_casted = igate_preact_step.to(_m_dtype)

    if use_robust_m_new_recurrent:
        m_new_casted = torch.maximum(log_fg_act_step + m_prev_casted, igate_preact_casted)
    else:
        m_new_casted = log_fg_act_step + m_prev_casted # Less robust

    if use_m_new_subtraction_recurrent_gates:
        fg_scaled = torch.exp(log_fg_act_step + m_prev_casted - m_new_casted)
        ig_scaled = torch.exp(igate_preact_casted - m_new_casted)
    else:
        fg_scaled = torch.exp(log_fg_act_step + m_prev_casted) # Potentially unstable
        ig_scaled = torch.exp(igate_preact_casted)           # Potentially unstable
    
    # Cast gates back to a type compatible with c_prev, n_prev, kv for accumulation
    # Typically input_dtype or a common high precision type like float32 if inputs are lower
    # For safety, can promote c_prev, n_prev to _m_dtype if different
    _accum_dtype = _m_dtype # Perform accumulation in the m_state's precision

    k_proj_scaled = k_proj / math.sqrt(DH)
    kv = torch.matmul(k_proj_scaled.to(_accum_dtype), v_proj.transpose(-1, -2).to(_accum_dtype))

    c_new = fg_scaled.to(_accum_dtype) * c_prev.to(_accum_dtype) + ig_scaled.to(_accum_dtype) * kv
    n_new = fg_scaled.to(_accum_dtype) * n_prev.to(_accum_dtype) + ig_scaled.to(_accum_dtype) * k_proj_scaled.to(_accum_dtype)

    # Output computation, also in _m_dtype for stability before final cast
    h_num = torch.matmul(q_proj.transpose(-1, -2).to(_accum_dtype), c_new)
    q_n = torch.matmul(q_proj.transpose(-1, -2).to(_accum_dtype), n_new)
    
    exp_neg_m_new = torch.exp(-m_new_casted)

    if use_robust_h_denom_recurrent:
        denom_base = torch.maximum(q_n.abs(), exp_neg_m_new)
    else:
        denom_base = q_n.abs() # Less robust

    if use_eps_in_recurrent_h_denom:
        denom = denom_base + eps
    else:
        denom = denom_base # Potentially unstable
        
    h_step = (h_num / denom).to(input_dtype)

    return h_step, (c_new.to(c_prev.dtype), n_new.to(n_prev.dtype), m_new_casted.to(m_prev.dtype))


################################################################################
# mLSTM Components (largely from OLD, with config for toggles)                #
################################################################################
@dataclass
class mLSTMCellConfig: # Based on OLD, adding toggle fields
    embedding_dim: int
    num_heads: int
    qkv_proj_blocksize: int
    context_length: int
    bias: bool = False
    # --- Experimental Toggles for mLSTM Backend ---
    mlstm_m_computation_dtype_recurrent: Optional[torch.dtype] = field(default=torch.float32, repr=False)
    mlstm_use_robust_m_new_recurrent: bool = field(default=True, repr=False)
    mlstm_use_m_new_subtraction_recurrent_gates: bool = field(default=True, repr=False)
    mlstm_use_robust_h_denom_recurrent: bool = field(default=True, repr=False)
    mlstm_use_eps_in_recurrent_h_denom: bool = field(default=True, repr=False) # Old behavior had eps
    # Parallel path toggles
    mlstm_use_max_log_d_subtraction: bool = field(default=True, repr=False)
    mlstm_max_log_d_computation_dtype: Optional[torch.dtype] = field(default=torch.float32, repr=False)
    mlstm_use_robust_normalizer_parallel: bool = field(default=True, repr=False)
    mlstm_use_eps_in_parallel_norm_denom: bool = field(default=True, repr=False) # Old behavior had eps

class mLSTMCell(nn.Module): # Based on OLD
    config_class = mLSTMCellConfig
    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config # Store the config
        self.inner_dim = config.embedding_dim
        self.num_master_heads = config.num_heads
        assert self.inner_dim % self.num_master_heads == 0, \
            f"Cell's embedding_dim ({self.inner_dim}) must be divisible by num_master_heads ({self.num_master_heads})"
        self.master_head_dim = self.inner_dim // self.num_master_heads

        self.qkv_proj_blocksize = config.qkv_proj_blocksize
        assert self.inner_dim % self.qkv_proj_blocksize == 0, \
            f"Cell's embedding_dim ({self.inner_dim}) must be divisible by qkv_proj_blocksize ({self.qkv_proj_blocksize})"
        self.num_qkv_heads = round(self.inner_dim // self.qkv_proj_blocksize) # OLD behavior

        qkv_linear_config = LinearHeadwiseExpandConfig( # Uses OLD LinearHeadwiseExpand
            in_features=self.inner_dim, num_heads=self.num_qkv_heads, bias=config.bias)
        self.q_proj = LinearHeadwiseExpand(qkv_linear_config)
        self.k_proj = LinearHeadwiseExpand(qkv_linear_config)
        self.v_proj = LinearHeadwiseExpand(qkv_linear_config)

        self.igate = nn.Linear(3 * self.inner_dim, self.num_master_heads, bias=True)
        self.fgate = nn.Linear(3 * self.inner_dim, self.num_master_heads, bias=True)

        self.outnorm = MultiHeadLayerNorm( # Uses OLD MultiHeadLayerNorm
            num_heads=self.num_master_heads,
            head_dim=self.master_head_dim,
            weight=True, bias=False, eps=1e-5 # OLD eps
        )
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self): # FROM OLD
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        nn.init.zeros_(self.fgate.weight)
        if self.fgate.bias is not None:
            bias_linspace_init_(self.fgate.bias, start=3.4, end=6.0) # OLD default
        nn.init.zeros_(self.igate.weight)
        if self.igate.bias is not None:
            nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
        self.outnorm.reset_parameters()

    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor, v_input: torch.Tensor) -> torch.Tensor:
        B, S, _ = q_input.shape
        q_sub_heads = self.q_proj(q_input) # OLD LinearHeadwiseExpand returns (B,S,D)
        k_sub_heads = self.k_proj(k_input)
        v_sub_heads = self.v_proj(v_input)
        
        # Reshape to (B, S, num_master_heads, master_head_dim)
        q_master_S_major = q_sub_heads.view(B, S, self.num_master_heads, self.master_head_dim)
        k_master_S_major = k_sub_heads.view(B, S, self.num_master_heads, self.master_head_dim)
        v_master_S_major = v_sub_heads.view(B, S, self.num_master_heads, self.master_head_dim)
        
        q_for_backend = q_master_S_major.transpose(1, 2) # (B, num_master_heads, S, master_head_dim)
        k_for_backend = k_master_S_major.transpose(1, 2)
        v_for_backend = v_master_S_major.transpose(1, 2)
        
        if_gate_inp = torch.cat([q_input, k_input, v_input], dim=-1)
        igate_preact_S_major = self.igate(if_gate_inp) # (B, S, num_master_heads)
        fgate_preact_S_major = self.fgate(if_gate_inp) # (B, S, num_master_heads)
        
        igate_for_backend = igate_preact_S_major.unsqueeze(-1).transpose(1, 2) # (B, num_master_heads, S, 1)
        fgate_for_backend = fgate_preact_S_major.unsqueeze(-1).transpose(1, 2) # (B, num_master_heads, S, 1)
        
        current_causal_mask = self.causal_mask[:S, :S] if S <= self.causal_mask.size(0) else _prepare_ltr(S, q_input.device, torch.bool)
        
        h_tilde_backend = parallel_stabilized_simple(
            q_for_backend, k_for_backend, v_for_backend,
            igate_for_backend, fgate_for_backend,
            lower_triangular_matrix=current_causal_mask,
            eps=EPS, # Use global EPS from old file
            use_max_log_d_subtraction=self.config.mlstm_use_max_log_d_subtraction,
            max_log_d_computation_dtype=self.config.mlstm_max_log_d_computation_dtype,
            use_robust_normalizer_parallel=self.config.mlstm_use_robust_normalizer_parallel,
            use_eps_in_parallel_norm_denom=self.config.mlstm_use_eps_in_parallel_norm_denom
        )
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

        q_master_S_major = q_sub_heads.view(B, S, self.num_master_heads, self.master_head_dim)
        k_master_S_major = k_sub_heads.view(B, S, self.num_master_heads, self.master_head_dim)
        v_master_S_major = v_sub_heads.view(B, S, self.num_master_heads, self.master_head_dim)

        q_for_backend = q_master_S_major.transpose(1, 2)
        k_for_backend = k_master_S_major.transpose(1, 2)
        v_for_backend = v_master_S_major.transpose(1, 2)

        if_gate_inp = torch.cat([q_input_step, k_input_step, v_input_step], dim=-1)
        igate_preact_S_major = self.igate(if_gate_inp)
        fgate_preact_S_major = self.fgate(if_gate_inp)
        igate_for_backend = igate_preact_S_major.unsqueeze(-1).transpose(1, 2)
        fgate_for_backend = fgate_preact_S_major.unsqueeze(-1).transpose(1, 2)

        # Determine m_prev_dtype for state initialization
        # Old file always initialized m_prev to float32.
        # New file initializes based on config or input. Let's use config.
        _m_dtype_init = self.config.mlstm_m_computation_dtype_recurrent \
            if self.config.mlstm_m_computation_dtype_recurrent is not None \
            else torch.float32 # Fallback to old behavior if None

        if mlstm_state is None:
            current_dtype_dev = q_for_backend.device
            current_dtype_val = q_for_backend.dtype # for c, n
            c_prev = torch.zeros(B, self.num_master_heads, self.master_head_dim, self.master_head_dim,
                                 device=current_dtype_dev, dtype=current_dtype_val)
            n_prev = torch.zeros(B, self.num_master_heads, self.master_head_dim, 1,
                                 device=current_dtype_dev, dtype=current_dtype_val)
            m_prev = torch.zeros(B, self.num_master_heads, 1, 1, # Initial m_prev can be in target m_dtype
                                 device=current_dtype_dev, dtype=_m_dtype_init)
        else:
            c_prev, n_prev, m_prev = mlstm_state
            # Ensure m_prev is in the computation dtype for the recurrent step if specified
            # Or keep its current dtype if mlstm_m_computation_dtype_recurrent is None (pass-through)
            if self.config.mlstm_m_computation_dtype_recurrent is not None and \
               m_prev.dtype != self.config.mlstm_m_computation_dtype_recurrent:
                m_prev = m_prev.to(self.config.mlstm_m_computation_dtype_recurrent)
            elif self.config.mlstm_m_computation_dtype_recurrent is None and m_prev.dtype != _m_dtype_init:
                # This case implies m_prev should be _m_dtype_init (e.g. float32) if config is None
                 m_prev = m_prev.to(_m_dtype_init)


        h_tilde_backend, new_state = recurrent_step_stabilized_simple(
            c_prev, n_prev, m_prev,
            q_for_backend, k_for_backend, v_for_backend,
            igate_for_backend, fgate_for_backend,
            eps=EPS, # Use global EPS from old file
            m_computation_dtype_recurrent=self.config.mlstm_m_computation_dtype_recurrent,
            use_robust_m_new_recurrent=self.config.mlstm_use_robust_m_new_recurrent,
            use_m_new_subtraction_recurrent_gates=self.config.mlstm_use_m_new_subtraction_recurrent_gates,
            use_robust_h_denom_recurrent=self.config.mlstm_use_robust_h_denom_recurrent,
            use_eps_in_recurrent_h_denom=self.config.mlstm_use_eps_in_recurrent_h_denom
        )
        h_norm = self.outnorm(h_tilde_backend)
        h_final_step = h_norm.transpose(1, 2).contiguous().view(B, S, -1)
        return h_final_step, new_state

@dataclass
class mLSTMLayerConfig: # Based on OLD, adding toggle fields for cell
    embedding_dim: int; context_length: int; num_heads: int = 4; conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4; proj_factor: float = 2.0; bias: bool = False
    dropout: float = 0.0; _num_blocks: int = 1
    # Fields to pass to mLSTMCellConfig
    mlstm_m_computation_dtype_recurrent: Optional[torch.dtype] = field(default=torch.float32, repr=False)
    mlstm_use_robust_m_new_recurrent: bool = field(default=True, repr=False)
    mlstm_use_m_new_subtraction_recurrent_gates: bool = field(default=True, repr=False)
    mlstm_use_robust_h_denom_recurrent: bool = field(default=True, repr=False)
    mlstm_use_eps_in_recurrent_h_denom: bool = field(default=True, repr=False)
    mlstm_use_max_log_d_subtraction: bool = field(default=True, repr=False)
    mlstm_max_log_d_computation_dtype: Optional[torch.dtype] = field(default=torch.float32, repr=False)
    mlstm_use_robust_normalizer_parallel: bool = field(default=True, repr=False)
    mlstm_use_eps_in_parallel_norm_denom: bool = field(default=True, repr=False)


class mLSTMLayer(nn.Module): # Based on OLD
    config_class = mLSTMLayerConfig
    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config
        raw_inner = config.embedding_dim * config.proj_factor
        self.inner_dim = int(round(raw_inner / 8.0)) * 8 # OLD inner_dim calculation
        inner = self.inner_dim
        self.embedding_dim = config.embedding_dim
        self.proj_up = nn.Linear(self.embedding_dim, 2 * inner, bias=config.bias)
        self.conv1d = CausalConv1d(CausalConv1dConfig(
            feature_dim=inner, kernel_size=config.conv1d_kernel_size, bias=config.bias,
        ))
        self.conv_act = nn.SiLU()
        
        cell_cfg = mLSTMCellConfig(
            embedding_dim=inner, num_heads=config.num_heads,
            qkv_proj_blocksize=config.qkv_proj_blocksize,
            context_length=config.context_length, bias=config.bias,
            # Pass toggles from layer config
            mlstm_m_computation_dtype_recurrent=config.mlstm_m_computation_dtype_recurrent,
            mlstm_use_robust_m_new_recurrent=config.mlstm_use_robust_m_new_recurrent,
            mlstm_use_m_new_subtraction_recurrent_gates=config.mlstm_use_m_new_subtraction_recurrent_gates,
            mlstm_use_robust_h_denom_recurrent=config.mlstm_use_robust_h_denom_recurrent,
            mlstm_use_eps_in_recurrent_h_denom=config.mlstm_use_eps_in_recurrent_h_denom,
            mlstm_use_max_log_d_subtraction=config.mlstm_use_max_log_d_subtraction,
            mlstm_max_log_d_computation_dtype=config.mlstm_max_log_d_computation_dtype,
            mlstm_use_robust_normalizer_parallel=config.mlstm_use_robust_normalizer_parallel,
            mlstm_use_eps_in_parallel_norm_denom=config.mlstm_use_eps_in_parallel_norm_denom
        )
        self.mlstm_cell = mLSTMCell(cell_cfg)

        self.proj_down = nn.Linear(inner, self.embedding_dim, bias=config.bias)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.learnable_skip = nn.Parameter(torch.ones(inner))
        self.ogate_act = nn.SiLU()
        self.reset_parameters() # Call reset_parameters from OLD

    def reset_parameters(self): # FROM OLD
        small_init_init_(self.proj_up.weight, dim=self.embedding_dim)
        if self.config.bias and self.proj_up.bias is not None: nn.init.zeros_(self.proj_up.bias)
        # self.conv1d.reset_parameters() # conv1d from old has pass
        self.mlstm_cell.reset_parameters()
        # nn.init.ones_(self.learnable_skip) # learnable_skip is initialized to ones by default
        wang_init_(self.proj_down.weight, dim=self.inner_dim, num_blocks=self.config._num_blocks)
        if self.config.bias and self.proj_down.bias is not None: nn.init.zeros_(self.proj_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # FROM OLD
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
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: # FROM OLD
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
class mLSTMBlockConfig: # FROM OLD
    mlstm_layer_config: mLSTMLayerConfig
class mLSTMBlock(nn.Module): # FROM OLD
    config_class = mLSTMBlockConfig
    def __init__(self, config: mLSTMBlockConfig):
        super().__init__()
        self.config = config
        self.ln = LayerNorm(config.mlstm_layer_config.embedding_dim, weight=True, bias=False)
        self.core = mLSTMLayer(config.mlstm_layer_config)
        self.reset_parameters()
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
# sLSTM Components (Based on OLD, with Toggles in Cell)                     #
###############################################################################
def powerlaw_blockdependent_init_(bias_tensor: torch.Tensor, head_dim: int, block_idx: int, num_blocks: int): # FROM OLD
    with torch.no_grad():
        if num_blocks > 1: ratio_0_to_1 = block_idx / (num_blocks - 1.0)
        else: ratio_0_to_1 = 0.0
        if head_dim == 1: term_val = 0.0
        else: term_val = (torch.arange(head_dim, device=bias_tensor.device, dtype=bias_tensor.dtype) / (head_dim - 1.0))
        exponent = 0.3 + 1.3 * ratio_0_to_1
        init_values_raw = -(-5.0 + 12.0 * (term_val ** exponent))
        bias_tensor.copy_(init_values_raw.unsqueeze(0).expand_as(bias_tensor))
    return bias_tensor

@dataclass
class sLSTMCellConfig: # Based on OLD, adding toggle fields
    embedding_dim: int
    num_heads: int = 4
    bias: bool = True # Corresponds to cell_final_bias in old sLSTMLayerConfig
    recurrent_bias: bool = False # Corresponds to cell_recurrent_transform_bias
    recurrent_transform_weight_init_std: Optional[float] = None
    _block_idx: int = 0 # For powerlaw_blockdependent_init_
    _num_blocks: int = 1 # For powerlaw_blockdependent_init_
    # --- Experimental Toggles for sLSTMCell ---
    slstm_m_state_dtype: Optional[torch.dtype] = field(default=torch.float32, repr=False) # Old behavior was float32
    slstm_use_robust_m_new: bool = field(default=True, repr=False)
    slstm_use_m_new_subtraction_gates: bool = field(default=True, repr=False)
    slstm_use_eps_in_h_denom: bool = field(default=True, repr=False) # Old behavior had eps

    @property
    def head_dim(self) -> int:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        return self.embedding_dim // self.num_heads
    def __post_init__(self):
        _ = self.head_dim

class sLSTMCell(nn.Module): # Based on OLD, with Toggles
    config_class = sLSTMCellConfig
    def __init__(self, cfg: sLSTMCellConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        lhe_config_internal = LinearHeadwiseExpandConfig( # Uses OLD LinearHeadwiseExpand
            in_features=cfg.embedding_dim, num_heads=cfg.num_heads, bias=cfg.recurrent_bias
        )
        self.gate_transforms = nn.ModuleList([
            LinearHeadwiseExpand(deepcopy(lhe_config_internal)) for _ in range(4) # z, i, f, o
        ])
        if cfg.bias:
            self.gate_biases = nn.Parameter(torch.empty(4, cfg.num_heads, self.head_dim))
        else:
            self.register_parameter('gate_biases', None)
        self.reset_parameters()

    def reset_parameters(self): # FROM OLD
        for i in range(4):
            transform_layer = self.gate_transforms[i]
            if self.cfg.recurrent_transform_weight_init_std is not None:
                std = self.cfg.recurrent_transform_weight_init_std
                if std == 0:
                    with torch.no_grad(): transform_layer.weight.data.fill_(0.0)
                elif std > 0:
                    with torch.no_grad():
                        for h_idx in range(self.cfg.num_heads):
                            transform_layer.weight.data[h_idx].normal_(mean=0.0, std=std)
                if self.cfg.recurrent_bias and transform_layer.bias is not None:
                    nn.init.zeros_(transform_layer.bias.data)
            else:
                transform_layer.reset_parameters()
        if self.cfg.bias and self.gate_biases is not None:
            nn.init.zeros_(self.gate_biases[0, ...]) # z_bias
            nn.init.zeros_(self.gate_biases[1, ...]) # i_bias
            nn.init.zeros_(self.gate_biases[3, ...]) # o_bias
            powerlaw_blockdependent_init_(self.gate_biases[2, ...],
                                          self.head_dim, self.cfg._block_idx, self.cfg._num_blocks)

    def _zeros_state(self, B: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_shape = (B, self.cfg.num_heads, self.head_dim)
        h0 = torch.zeros(state_shape, device=device, dtype=dtype)
        c0 = torch.zeros_like(h0)
        n0 = torch.zeros_like(h0)
        _m_dtype_init = self.cfg.slstm_m_state_dtype if self.cfg.slstm_m_state_dtype is not None else torch.float32 # Fallback for init
        m0 = torch.zeros(state_shape, device=device, dtype=_m_dtype_init)
        return h0, c0, n0, m0

    def forward(self, x_gate_inputs: torch.Tensor,
                slstm_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S, D_quad_model = x_gate_inputs.shape
        assert D_quad_model == self.cfg.embedding_dim * 4, "Input dimension mismatch"
        
        input_dtype = x_gate_inputs.dtype # Original dtype of inputs/outputs like h, c, n

        projs_from_layer = torch.split(x_gate_inputs, self.cfg.embedding_dim, dim=-1)
        transformed_signals_list = [
            self.gate_transforms[i](projs_from_layer[i]) for i in range(4)
        ]
        z_transformed = transformed_signals_list[0].view(B, S, self.cfg.num_heads, self.head_dim)
        i_transformed = transformed_signals_list[1].view(B, S, self.cfg.num_heads, self.head_dim)
        f_transformed = transformed_signals_list[2].view(B, S, self.cfg.num_heads, self.head_dim)
        o_transformed = transformed_signals_list[3].view(B, S, self.cfg.num_heads, self.head_dim)

        if slstm_cell_state is None:
            h_prev, c_prev, n_prev, m_prev = self._zeros_state(B, x_gate_inputs.device, input_dtype)
        else:
            h_prev, c_prev, n_prev, m_prev = slstm_cell_state
        
        # Determine m_state computation dtype
        _m_dtype = self.cfg.slstm_m_state_dtype if self.cfg.slstm_m_state_dtype is not None else m_prev.dtype
        m_prev_casted = m_prev.to(_m_dtype)

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
                z_biased, i_biased, f_biased, o_biased = z_t_curr, i_t_curr, f_t_curr, o_t_curr
            
            z_val = torch.tanh(z_biased)    # Operates on input_dtype
            i_val = torch.exp(i_biased)     # Operates on input_dtype
            f_val = torch.exp(f_biased)     # Operates on input_dtype
            o_val = torch.sigmoid(o_biased) # Operates on input_dtype

            # Cast pre-activations to _m_dtype for m_new and scaled gates computation
            log_f_val_casted = torch.log(f_val.to(_m_dtype) + EPS) # Add EPS before log as in old
            log_i_val_casted = torch.log(i_val.to(_m_dtype) + EPS) # Add EPS before log as in old

            if self.cfg.slstm_use_robust_m_new:
                m_new_casted = torch.maximum(log_f_val_casted + m_prev_casted, log_i_val_casted)
            else:
                m_new_casted = log_f_val_casted + m_prev_casted # Less robust

            if self.cfg.slstm_use_m_new_subtraction_gates:
                i_hat_casted = torch.exp(log_i_val_casted - m_new_casted)
                f_hat_casted = torch.exp(log_f_val_casted + m_prev_casted - m_new_casted)
            else:
                i_hat_casted = torch.exp(log_i_val_casted) # Potentially unstable
                f_hat_casted = torch.exp(log_f_val_casted + m_prev_casted) # Potentially unstable
            
            # Cast scaled gates back to input_dtype for accumulation with c_prev, n_prev, z_val
            i_hat = i_hat_casted.to(input_dtype)
            f_hat = f_hat_casted.to(input_dtype)
            
            c_new = f_hat * c_prev + i_hat * z_val
            n_new = f_hat * n_prev + i_hat
            
            if self.cfg.slstm_use_eps_in_h_denom:
                h_new = o_val * (c_new / (n_new + EPS))
            else:
                h_new = o_val * (c_new / n_new) # Potentially unstable

            outs_h.append(h_new)
            h_prev, c_prev, n_prev, m_prev_casted = h_new, c_new, n_new, m_new_casted
        
        final_h_sequence = torch.stack(outs_h, dim=1)
        final_state_tuple = (h_prev, c_prev, n_prev, m_prev_casted.to(m_prev.dtype)) # Ensure m_state returns in original m_prev dtype
        return final_h_sequence, final_state_tuple

    def step(self, x_gate_inputs_step: torch.Tensor,
               slstm_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_sequence, final_state = self.forward(x_gate_inputs_step, slstm_cell_state=slstm_cell_state)
        return h_sequence, final_state

@dataclass
class sLSTMLayerConfig: # Based on OLD, with toggle fields for cell
    embedding_dim: int
    context_length: int
    num_heads: int = 4
    conv1d_kernel_size: int = 0
    bias: bool = False
    dropout: float = 0.0
    group_norm_weight: bool = True
    # Fields to pass to sLSTMCellConfig (from OLD config structure)
    cell_final_bias: bool = True # sLSTMCellConfig.bias
    cell_recurrent_transform_bias: bool = False # sLSTMCellConfig.recurrent_bias
    cell_recurrent_transform_weight_init_std: Optional[float] = None # sLSTMCellConfig.recurrent_transform_weight_init_std
    # Block info for powerlaw init
    _block_idx: int = 0
    _num_blocks: int = 1
    # New toggle fields to pass to sLSTMCellConfig
    slstm_m_state_dtype: Optional[torch.dtype] = field(default=torch.float32, repr=False)
    slstm_use_robust_m_new: bool = field(default=True, repr=False)
    slstm_use_m_new_subtraction_gates: bool = field(default=True, repr=False)
    slstm_use_eps_in_h_denom: bool = field(default=True, repr=False)

    @property
    def head_dim(self) -> int:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        return self.embedding_dim // self.num_heads
    def __post_init__(self):
        _ = self.head_dim

class sLSTMLayer(nn.Module): # Based on OLD
    config_class = sLSTMLayerConfig
    def __init__(self, cfg: sLSTMLayerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(CausalConv1dConfig(
                feature_dim=cfg.embedding_dim, kernel_size=cfg.conv1d_kernel_size, bias=cfg.bias
            ))
            self.conv_act = nn.SiLU()
        else:
            self.conv1d, self.conv_act = None, None

        proj_cfg = LinearHeadwiseExpandConfig( # Uses OLD LinearHeadwiseExpand
            in_features=cfg.embedding_dim, num_heads=cfg.num_heads, bias=cfg.bias
        )
        self.z_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))
        self.i_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))
        self.f_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))
        self.o_gate_proj = LinearHeadwiseExpand(deepcopy(proj_cfg))

        cell_cfg_obj = sLSTMCellConfig( # sLSTMCellConfig now includes old + new toggles
            embedding_dim=cfg.embedding_dim, num_heads=cfg.num_heads,
            bias=cfg.cell_final_bias, # from old sLSTMLayerConfig
            recurrent_bias=cfg.cell_recurrent_transform_bias, # from old
            recurrent_transform_weight_init_std=cfg.cell_recurrent_transform_weight_init_std, # from old
            _block_idx=cfg._block_idx, _num_blocks=cfg._num_blocks, # from old
            # Pass new toggles
            slstm_m_state_dtype=cfg.slstm_m_state_dtype,
            slstm_use_robust_m_new=cfg.slstm_use_robust_m_new,
            slstm_use_m_new_subtraction_gates=cfg.slstm_use_m_new_subtraction_gates,
            slstm_use_eps_in_h_denom=cfg.slstm_use_eps_in_h_denom
        )
        self.cell = sLSTMCell(cell_cfg_obj)
        self.dropout_layer = nn.Dropout(cfg.dropout)
        head_dim_for_norm = cfg.embedding_dim // cfg.num_heads
        self.group_norm = MultiHeadLayerNorm( # Uses OLD MultiHeadLayerNorm
            num_heads=cfg.num_heads, head_dim=head_dim_for_norm,
            weight=cfg.group_norm_weight, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self): # FROM OLD
        if self.conv1d is not None: self.conv1d.reset_parameters() # old was pass
        self.z_gate_proj.reset_parameters()
        self.i_gate_proj.reset_parameters()
        self.f_gate_proj.reset_parameters()
        self.o_gate_proj.reset_parameters()
        self.cell.reset_parameters()
        self.group_norm.reset_parameters()

    def forward(self, x: torch.Tensor,
                slstm_cell_state_init: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
               ) -> torch.Tensor: # FROM OLD
        B, S, D_embed = x.shape
        if self.conv1d is not None and self.conv_act is not None:
            x_conv, x_conv_activated = self.conv1d(x), self.conv_act(self.conv1d(x)) # Corrected: apply conv once
        else:
            x_conv_activated = x
        i_projected, f_projected = self.i_gate_proj(x_conv_activated), self.f_gate_proj(x_conv_activated)
        z_projected, o_projected = self.z_gate_proj(x), self.o_gate_proj(x)
        gate_inputs_concat = torch.cat([z_projected, i_projected, f_projected, o_projected], dim=-1)
        h_cell_output, _ = self.cell(gate_inputs_concat, slstm_cell_state=slstm_cell_state_init)
        h_dropped = self.dropout_layer(h_cell_output)
        h_norm_input = h_dropped.transpose(1, 2)
        h_norm_output = self.group_norm(h_norm_input)
        h_final = h_norm_output.transpose(1, 2).contiguous()
        return h_final.view(B, S, D_embed)

    def step(self, x_step: torch.Tensor,
               slstm_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
              ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: # FROM OLD
        B, S_one, D_embed = x_step.shape; assert S_one == 1
        if self.conv1d is not None and self.conv_act is not None:
            x_conv_step, x_conv_activated_step = self.conv1d(x_step), self.conv_act(self.conv1d(x_step)) # Corrected
        else:
            x_conv_activated_step = x_step
        i_projected_step, f_projected_step = self.i_gate_proj(x_conv_activated_step), self.f_gate_proj(x_conv_activated_step)
        z_projected_step, o_projected_step = self.z_gate_proj(x_step), self.o_gate_proj(x_step)
        gate_inputs_concat_step = torch.cat([z_projected_step, i_projected_step, f_projected_step, o_projected_step], dim=-1)
        h_cell_step_output, new_slstm_cell_state_tuple = self.cell.step(
            gate_inputs_concat_step, slstm_cell_state=slstm_cell_state
        )
        h_dropped_step = self.dropout_layer(h_cell_step_output)
        h_norm_input_step = h_dropped_step.transpose(1, 2)
        h_norm_output_step = self.group_norm(h_norm_input_step)
        h_final_step = h_norm_output_step.transpose(1, 2).contiguous()
        output_step = h_final_step.view(B, S_one, D_embed)
        return output_step, new_slstm_cell_state_tuple

@dataclass
class FeedForwardConfig: # FROM OLD (Identical)
    proj_factor: float = 1.0; act_fn: str = "gelu"; dropout: float = 0.0; bias: bool = False
class FeedForward(nn.Module): # FROM OLD (Identical logic, uses old small_init_init_)
    config_class = FeedForwardConfig
    def __init__(self, config: FeedForwardConfig, embedding_dim: int):
        super().__init__()
        self.config = config; self.embedding_dim = embedding_dim
        inner_dim = int(config.proj_factor * embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, inner_dim, bias=config.bias)
        if config.act_fn == "gelu": self.act = nn.GELU()
        elif config.act_fn == "silu": self.act = nn.SiLU()
        else: self.act = getattr(F, config.act_fn)
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
        x = self.fc1(x); x = self.act(x); x = self.fc2(x)
        return self.dropout_layer(x)

@dataclass
class sLSTMBlockConfig: # Based on OLD
    slstm_layer_config: sLSTMLayerConfig
    feedforward_config: Optional[FeedForwardConfig] = None
    _block_idx: int = 0 # For sLSTMLayerConfig
    _num_blocks: int = 1 # For sLSTMLayerConfig
    def __post_init__(self): # FROM OLD
        if self.slstm_layer_config is not None:
            self.slstm_layer_config._block_idx = self._block_idx
            self.slstm_layer_config._num_blocks = self.num_blocks

class sLSTMBlock(nn.Module): # Based on OLD
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
            self.ln2, self.ffn = None, None
        self.reset_parameters()
    def reset_parameters(self): # FROM OLD
        self.ln1.reset_parameters()
        self.core.reset_parameters()
        if self.ffn is not None and self.ln2 is not None:
            self.ln2.reset_parameters()
            self.ffn.reset_parameters()
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor: # FROM OLD
        identity = x; x = self.ln1(x); x = self.core(x); x = identity + x
        if self.ffn is not None and self.ln2 is not None:
            identity_ffn = x; x = self.ln2(x); x = self.ffn(x); x = identity_ffn + x
        return x
    def step(self, x_step: torch.Tensor, state: Optional[Dict[str, Tuple]] = None, **kwargs): # FROM OLD
        slstm_core_state_val = state.get("slstm_layer_state") if state else None
        identity_step = x_step; x_ln1_step = self.ln1(x_step)
        core_out_step, new_slstm_core_state_tuple = self.core.step(
            x_ln1_step, slstm_cell_state=slstm_core_state_val
        )
        res_x_step = identity_step + core_out_step
        if self.ffn is not None and self.ln2 is not None:
            identity_ffn_step = res_x_step; x_ln2_step = self.ln2(res_x_step)
            ffn_out_step = self.ffn(x_ln2_step); final_x_step = identity_ffn_step + ffn_out_step
        else:
            final_x_step = res_x_step
        new_state_dict = {"slstm_layer_state": new_slstm_core_state_tuple}
        return final_x_step, new_state_dict

################################################################################
# xLSTM Block Stack (Based on OLD, with Toggles in underlying Configs)         #
################################################################################
@dataclass
class xLSTMBlockStackConfig: # Based on OLD, adding top-level toggle defaults
    mlstm_block_template: Optional[mLSTMBlockConfig] = None
    slstm_block_template: Optional[sLSTMBlockConfig] = None
    num_blocks: int = 6; embedding_dim: int = 512; context_length: int = 96
    bias: bool = False; dropout: float = 0.0
    add_post_blocks_norm: bool = True
    slstm_at: Union[List[int], Literal["all"], Literal["none"]] = field(default_factory=list)
    _block_map_str: str = field(init=False, repr=False)

    # --- Top-level experimental toggles defaults ---
    # For mLSTM cells:
    default_mlstm_m_computation_dtype_recurrent: Optional[torch.dtype] = torch.float32
    default_mlstm_use_robust_m_new_recurrent: bool = True
    default_mlstm_use_m_new_subtraction_recurrent_gates: bool = True
    default_mlstm_use_robust_h_denom_recurrent: bool = True
    default_mlstm_use_eps_in_recurrent_h_denom: bool = True
    default_mlstm_use_max_log_d_subtraction: bool = True
    default_mlstm_max_log_d_computation_dtype: Optional[torch.dtype] = torch.float32
    default_mlstm_use_robust_normalizer_parallel: bool = True
    default_mlstm_use_eps_in_parallel_norm_denom: bool = True
    # For sLSTM cells:
    default_slstm_m_state_dtype: Optional[torch.dtype] = torch.float32
    default_slstm_use_robust_m_new: bool = True
    default_slstm_use_m_new_subtraction_gates: bool = True
    default_slstm_use_eps_in_h_denom: bool = True

    @property
    def block_map(self) -> List[int]: return list(map(int, self._block_map_str.split(",")))
    def _create_block_map_str(self) -> str: # FROM OLD
        if self.slstm_at == "all":
            if self.slstm_block_template is None: raise ValueError("sLSTM blocks specified ('all') but 'slstm_block_template' is None.")
            resolved_slstm_at = list(range(self.num_blocks))
        elif self.slstm_at == "none": resolved_slstm_at = []
        else: resolved_slstm_at = self.slstm_at
        block_map_list = [0] * self.num_blocks
        for slstm_idx in resolved_slstm_at:
            if not (0 <= slstm_idx < self.num_blocks): raise ValueError(f"Invalid sLSTM position index {slstm_idx} for {self.num_blocks} blocks.")
            if self.slstm_block_template is None: raise ValueError(f"sLSTM block specified at index {slstm_idx} but 'slstm_block_template' is None.")
            block_map_list[slstm_idx] = 1
        if any(b_type == 0 for b_type in block_map_list) and self.mlstm_block_template is None:
            raise ValueError("mLSTM block type inferred in block_map, but 'mlstm_block_template' is None.")
        return ",".join(map(str, block_map_list))

    def __post_init__(self): # Modified to pass down new defaults
        if self.num_blocks <= 0: raise ValueError("num_blocks must be positive.")
        if self.mlstm_block_template is None and self.slstm_block_template is None:
            raise ValueError("At least one of 'mlstm_block_template' or 'slstm_block_template' must be provided.")
        if not self.slstm_at and isinstance(self.slstm_at, list):
            if self.mlstm_block_template is not None and self.slstm_block_template is None: self.slstm_at = "none"
            elif self.mlstm_block_template is None and self.slstm_block_template is not None: self.slstm_at = "all"

        if self.mlstm_block_template is not None:
            layer_cfg = self.mlstm_block_template.mlstm_layer_config
            layer_cfg.embedding_dim, layer_cfg.context_length = self.embedding_dim, self.context_length
            layer_cfg.bias, layer_cfg.dropout, layer_cfg._num_blocks = self.bias, self.dropout, self.num_blocks
            # Pass mLSTM toggle defaults
            layer_cfg.mlstm_m_computation_dtype_recurrent = self.default_mlstm_m_computation_dtype_recurrent
            layer_cfg.mlstm_use_robust_m_new_recurrent = self.default_mlstm_use_robust_m_new_recurrent
            layer_cfg.mlstm_use_m_new_subtraction_recurrent_gates = self.default_mlstm_use_m_new_subtraction_recurrent_gates
            layer_cfg.mlstm_use_robust_h_denom_recurrent = self.default_mlstm_use_robust_h_denom_recurrent
            layer_cfg.mlstm_use_eps_in_recurrent_h_denom = self.default_mlstm_use_eps_in_recurrent_h_denom
            layer_cfg.mlstm_use_max_log_d_subtraction = self.default_mlstm_use_max_log_d_subtraction
            layer_cfg.mlstm_max_log_d_computation_dtype = self.default_mlstm_max_log_d_computation_dtype
            layer_cfg.mlstm_use_robust_normalizer_parallel = self.default_mlstm_use_robust_normalizer_parallel
            layer_cfg.mlstm_use_eps_in_parallel_norm_denom = self.default_mlstm_use_eps_in_parallel_norm_denom

        if self.slstm_block_template is not None:
            layer_cfg = self.slstm_block_template.slstm_layer_config
            layer_cfg.embedding_dim, layer_cfg.context_length = self.embedding_dim, self.context_length
            layer_cfg.bias, layer_cfg.dropout, layer_cfg._num_blocks = self.bias, self.dropout, self.num_blocks
            if self.slstm_block_template.feedforward_config is not None:
                ffn_cfg = self.slstm_block_template.feedforward_config
                ffn_cfg.dropout, ffn_cfg.bias = self.dropout, self.bias
            # Pass sLSTM toggle defaults
            layer_cfg.slstm_m_state_dtype = self.default_slstm_m_state_dtype
            layer_cfg.slstm_use_robust_m_new = self.default_slstm_use_robust_m_new
            layer_cfg.slstm_use_m_new_subtraction_gates = self.default_slstm_use_m_new_subtraction_gates
            layer_cfg.slstm_use_eps_in_h_denom = self.default_slstm_use_eps_in_h_denom
            # _block_idx is handled by sLSTMBlockConfig post_init

        self._block_map_str = self._create_block_map_str()

class xLSTMBlockStack(nn.Module): # Based on OLD
    config_class = xLSTMBlockStackConfig
    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config
        self.blocks = self._create_blocks()
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(config.embedding_dim, weight=True, bias=True)
        else:
            self.post_blocks_norm = nn.Identity()
        self.reset_parameters()

    def _create_blocks(self) -> nn.ModuleList: # FROM OLD (relies on config to have toggles)
        module_blocks = []
        for i, block_type_int in enumerate(self.config.block_map):
            if block_type_int == 0:
                if self.config.mlstm_block_template is None: raise RuntimeError("Trying to create mLSTMBlock but mlstm_block_template is None.")
                block_cfg = deepcopy(self.config.mlstm_block_template)
                # Ensure _num_blocks is correctly set on the layer_config if not already from stack config
                block_cfg.mlstm_layer_config._num_blocks = self.config.num_blocks
                module_blocks.append(mLSTMBlock(config=block_cfg))
            elif block_type_int == 1:
                if self.config.slstm_block_template is None: raise RuntimeError("Trying to create sLSTMBlock but slstm_block_template is None.")
                block_cfg = deepcopy(self.config.slstm_block_template)
                # Pass block index and total to the sLSTM block config for powerlaw_init
                block_cfg._block_idx = i
                block_cfg._num_blocks = self.config.num_blocks
                # This will be propagated to sLSTMLayerConfig by sLSTMBlockConfig.__post_init__
                module_blocks.append(sLSTMBlock(config=block_cfg))
            else:
                raise ValueError(f"Invalid block type {block_type_int} in block_map.")
        return nn.ModuleList(module_blocks)

    def reset_parameters(self) -> None: # FROM OLD
        for block in self.blocks: block.reset_parameters()
        if isinstance(self.post_blocks_norm, LayerNorm): self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor: # FROM OLD
        for block in self.blocks: x = block(x, **kwargs)
        x = self.post_blocks_norm(x); return x

    def step(self, x_step: torch.Tensor, states: Optional[List[Optional[Dict[str, Tuple]]]] = None
            ) -> Tuple[torch.Tensor, List[Optional[Dict[str, Tuple]]]]: # FROM OLD
        if states is None: states = [None] * len(self.blocks)
        if len(states) != len(self.blocks): raise ValueError(f"Number of states ({len(states)}) must match number of blocks ({len(self.blocks)}).")
        new_states_list: List[Optional[Dict[str, Tuple]]] = []
        current_x_step = x_step
        for i, block in enumerate(self.blocks):
            block_input_state_dict = states[i]
            current_x_step, returned_block_state_dict = block.step(current_x_step, state=block_input_state_dict)
            new_states_list.append(returned_block_state_dict)
        current_x_step = self.post_blocks_norm(current_x_step)
        return current_x_step, new_states_list