# experiments/lm_benchmark.py
import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict,List

import xlstm_replica as xlstm_scratch 
from xlstm import (
    xLSTMBlockStack as LibXLSTMBlockStack,
    xLSTMBlockStackConfig as LibXLSTMBlockStackConfig,
    mLSTMBlockConfig as LibMLSTMBlockConfig,
    mLSTMLayerConfig as LibMLSTMLayerConfig
)

from utils.data_preparation import load_wikitext2_data
from utils.model_architectures import SimpleLMWrapper, LSTMCoreLM, TransformerCoreLM 
from utils.training_loops import train_one_lm_model, evaluate_bins_ppl, evaluate_overall_ppl
from utils.plotting import plot_lm_results

# LM Specific Hyperparameters 
LM_CFG = {
    "batch_size": 64, "bptt": 96, "hidden_dim": 512, "embed_dim": 512, 
    "n_layers": 6, "n_heads": 8,
    "max_steps": 10_000, "lr_peak": 1e-3, "final_lr": 1e-7, "warmup_pct": 0.1,
    "grad_clip": 0.5, "patience": 6, "bin_edges": (100, 1000),
    "plot_scale_first_bin": 100.0,
    "conv1d_kernel_size": 4, "qkv_proj_blocksize": 4,
    "mlstm_proj_factor": 2.0, "bias": False, 
    "dropout": 0.1, "weight_decay": 0.01 
}

# --- LM Model Factories ---
def make_lib_xlstm_lm(vocab_size: int, pad_idx: int):
    cfg_layer = LibMLSTMLayerConfig(
        embedding_dim=LM_CFG["hidden_dim"], context_length=LM_CFG["bptt"],
        num_heads=LM_CFG["n_heads"], conv1d_kernel_size=LM_CFG["conv1d_kernel_size"],
        qkv_proj_blocksize=LM_CFG["qkv_proj_blocksize"], proj_factor=LM_CFG["mlstm_proj_factor"],
        bias=LM_CFG["bias"], dropout=LM_CFG["dropout"], _num_blocks=LM_CFG["n_layers"]
    )
    cfg_block = LibMLSTMBlockConfig(mlstm=cfg_layer)
    cfg_stack = LibXLSTMBlockStackConfig(
        mlstm_block=cfg_block, slstm_block=None, context_length=LM_CFG["bptt"],
        num_blocks=LM_CFG["n_layers"], embedding_dim=LM_CFG["hidden_dim"],
        add_post_blocks_norm=True, dropout=LM_CFG["dropout"], bias=LM_CFG["bias"]
    )
    core = LibXLSTMBlockStack(cfg_stack)
    core.out_dim = LM_CFG["hidden_dim"] 
    return SimpleLMWrapper(core, vocab_size, LM_CFG["hidden_dim"], pad_idx, bias=LM_CFG["bias"])

def make_scratch_xlstm_lm(vocab_size: int, pad_idx: int):
    scratch_mlstm_layer_cfg = xlstm_scratch.mLSTMLayerConfig(
        embedding_dim=LM_CFG["hidden_dim"], context_length=LM_CFG["bptt"],
        num_heads=LM_CFG["n_heads"], conv1d_kernel_size=LM_CFG["conv1d_kernel_size"],
        qkv_proj_blocksize=LM_CFG["qkv_proj_blocksize"], proj_factor=LM_CFG["mlstm_proj_factor"],
    )
    scratch_mlstm_block_template = xlstm_scratch.mLSTMBlockConfig(
        mlstm_layer_config=scratch_mlstm_layer_cfg
    )
    cfg_stack_scratch = xlstm_scratch.xLSTMBlockStackConfig(
        num_blocks=LM_CFG["n_layers"], embedding_dim=LM_CFG["hidden_dim"],
        context_length=LM_CFG["bptt"], bias=LM_CFG["bias"], dropout=LM_CFG["dropout"],
        mlstm_block_template=scratch_mlstm_block_template, slstm_block_template=None,
        slstm_at="none", add_post_blocks_norm=True,
    )
    core = xlstm_scratch.xLSTMBlockStack(cfg_stack_scratch)
    core.out_dim = LM_CFG["hidden_dim"] # Ensured this line is present
    return SimpleLMWrapper(core, vocab_size, LM_CFG["hidden_dim"], pad_idx, bias=LM_CFG["bias"])

def make_lstm_lm(vocab_size: int, pad_idx: int):
    core = LSTMCoreLM(input_dim=LM_CFG["embed_dim"], hidden_dim=LM_CFG["hidden_dim"], 
                      layers=LM_CFG["n_layers"], dropout=LM_CFG["dropout"])
    return SimpleLMWrapper(core, vocab_size, LM_CFG["embed_dim"], pad_idx, bias=LM_CFG["bias"])

def make_transformer_lm(vocab_size: int, pad_idx: int):
    core = TransformerCoreLM(input_dim=LM_CFG["embed_dim"], hidden_dim=LM_CFG["hidden_dim"],
                             layers=LM_CFG["n_layers"], n_heads=LM_CFG["n_heads"], 
                             dropout=LM_CFG["dropout"], bptt=LM_CFG["bptt"])
    return SimpleLMWrapper(core, vocab_size, LM_CFG["embed_dim"], pad_idx, bias=LM_CFG["bias"])

def run_lm_benchmark(device_str: str, benchmark_type: str = "xlstm_vs_lib"):
    print(f"\n--- Running LM Benchmark ({benchmark_type}) on WikiText-2 on {device_str} ---")
    current_lm_cfg = LM_CFG.copy()
    current_lm_cfg["device"] = device_str
    device = torch.device(device_str)

    vocab_size, pad_idx, freq_map, train_data, val_data, test_data = load_wikitext2_data(current_lm_cfg)

    models_to_run_lm: Dict[str, nn.Module] = {}
    plot_title = ""
    plot_filename_suffix = ""

    if benchmark_type == "xlstm_vs_lib":
        models_to_run_lm = {
            "Scratch-xLSTM (LM)": make_scratch_xlstm_lm(vocab_size, pad_idx),
            "Library-xLSTM (LM)": make_lib_xlstm_lm(vocab_size, pad_idx),
        }
        plot_title = "LM: Scratch xLSTM vs. Library xLSTM (PPL)"
        plot_filename_suffix = "xlstm_vs_lib"
    elif benchmark_type == "baselines_vs_xlstm":
        models_to_run_lm = {
            "LSTM (LM)": make_lstm_lm(vocab_size, pad_idx),
            "Library-xLSTM (LM)": make_lib_xlstm_lm(vocab_size, pad_idx),
            "Transformer (LM)": make_transformer_lm(vocab_size, pad_idx),
        }
        plot_title = "LM: Baselines vs. xLSTM (PPL)"
        plot_filename_suffix = "baselines_vs_xlstm"
    else:
        raise ValueError(f"Unknown LM benchmark_type: {benchmark_type}")

    trained_models_lm: Dict[str, torch.nn.Module] = {}
    for name, model_instance in models_to_run_lm.items():
        print(f"\nTraining {name} for LM task ({benchmark_type})...")
        params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        print(f"Param count: {params/1e6:.2f}M")
        
        trained_model, best_ppl = train_one_lm_model(name, model_instance, train_data, val_data, current_lm_cfg, pad_idx, device_str)
        if trained_model is not None:
            trained_models_lm[name] = trained_model
        else:
            print(f"Training failed for {name}. It will be excluded from evaluation.")


    ppl_results_lm: Dict[str, List[float]] = {}
    if not trained_models_lm:
        print(f"No LM models finished training successfully for benchmark '{benchmark_type}'. Skipping LM evaluation.")
        return {}
        
    for name, model_instance in trained_models_lm.items():
        print(f"\nEvaluating {name} for LM task ({benchmark_type})...")
        model_instance.to(device)
        
        bin_ppls = evaluate_bins_ppl(model_instance, test_data, current_lm_cfg["bptt"], device_str, 
                                     pad_idx=pad_idx, freq_vec=freq_map, # freq_map will be moved to device in func
                                     bin_edges=current_lm_cfg["bin_edges"])
        overall_ppl = evaluate_overall_ppl(model_instance, test_data, current_lm_cfg["bptt"], device_str, pad_idx)
        
        ppl_results_lm[name] = bin_ppls + [overall_ppl]
        
        overall_str = f"{overall_ppl:.2f}" if np.isfinite(overall_ppl) else "N/A"
        print(f"{name} - Bin PPLs: "
            f"{[f'{p:.2f}' if np.isfinite(p) else 'N/A' for p in bin_ppls]}, "
            f"Overall PPL: {overall_str}")
    plot_lm_results(ppl_results_lm, 
                    current_lm_cfg["bin_edges"], 
                    plot_scale_first_bin=current_lm_cfg["plot_scale_first_bin"],
                    filename=f"lm_benchmark_{plot_filename_suffix}_results.png")
    
    results_path = f"lm_benchmark_{plot_filename_suffix}_results.json"
    with open(results_path, "w") as f:
        json_compatible_results = {
            k: [None if np.isnan(val) else (float('inf') if np.isinf(val) else val) for val in v] 
            for k, v in ppl_results_lm.items()
        }
        json.dump(json_compatible_results, f, indent=2)
    print(f"LM PPL results saved to {results_path}")
    
    return ppl_results_lm