
import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict,List, Tuple, Optional

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
from utils.plotting import plot_lm_results,plot_lm_training_dynamics
from utils.model_architectures import GeneralHybridLM, GeneralHybridConfig

# LM Specific Hyperparameters 
LM_CFG = {
    "batch_size": 64, "bptt": 96, "hidden_dim": 512, "embed_dim": 512, 
    "n_layers": 6, "n_heads": 8,
    "max_steps": 10_000, "lr_peak": 1e-3, "final_lr": 1e-7, "warmup_pct": 0.1,
    "grad_clip": 0.5, "patience": 3, "bin_edges": (100, 1000),
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
def make_scratch_xlstm_lm(
    vocab_size: int,
    pad_idx: int,
    arg_mlstm_use_eps_in_parallel_norm_denom: bool = True,
    arg_mlstm_use_eps_in_recurrent_h_denom: bool = True,
    arg_mlstm_use_max_log_d_subtraction: bool = True,
    arg_mlstm_use_m_new_subtraction_recurrent_gates: bool = True,
    arg_mlstm_use_robust_normalizer_parallel: bool = True,
    arg_mlstm_use_robust_h_denom_recurrent: bool = True,
    arg_mlstm_use_robust_m_new_recurrent: bool = True,
    arg_mlstm_max_log_d_computation_dtype: Optional[torch.dtype] = torch.float32,
    arg_mlstm_m_computation_dtype_recurrent: Optional[torch.dtype] = torch.float32
):
    base_mlstm_layer_cfg = xlstm_scratch.mLSTMLayerConfig(
        embedding_dim=LM_CFG["hidden_dim"],
        context_length=LM_CFG["bptt"],
        num_heads=LM_CFG["n_heads"],
        conv1d_kernel_size=LM_CFG["conv1d_kernel_size"],
        qkv_proj_blocksize=LM_CFG["qkv_proj_blocksize"],
        proj_factor=LM_CFG["mlstm_proj_factor"],
        bias=LM_CFG["bias"],
        dropout=LM_CFG["dropout"],
        _num_blocks=LM_CFG["n_layers"]
    )
    scratch_mlstm_block_template = xlstm_scratch.mLSTMBlockConfig(
        mlstm_layer_config=base_mlstm_layer_cfg
    )

    cfg_stack_scratch = xlstm_scratch.xLSTMBlockStackConfig(
        num_blocks=LM_CFG["n_layers"],
        embedding_dim=LM_CFG["hidden_dim"],
        context_length=LM_CFG["bptt"],
        bias=LM_CFG["bias"],
        dropout=LM_CFG["dropout"],
        mlstm_block_template=scratch_mlstm_block_template, 
        slstm_block_template=None,
        slstm_at="none",
        add_post_blocks_norm=True,

        default_mlstm_use_eps_in_parallel_norm_denom=arg_mlstm_use_eps_in_parallel_norm_denom,
        default_mlstm_use_eps_in_recurrent_h_denom=arg_mlstm_use_eps_in_recurrent_h_denom,
        default_mlstm_use_max_log_d_subtraction=arg_mlstm_use_max_log_d_subtraction,
        default_mlstm_use_m_new_subtraction_recurrent_gates=arg_mlstm_use_m_new_subtraction_recurrent_gates,
        default_mlstm_use_robust_normalizer_parallel=arg_mlstm_use_robust_normalizer_parallel,
        default_mlstm_use_robust_h_denom_recurrent=arg_mlstm_use_robust_h_denom_recurrent,
        default_mlstm_use_robust_m_new_recurrent=arg_mlstm_use_robust_m_new_recurrent,
        default_mlstm_max_log_d_computation_dtype=arg_mlstm_max_log_d_computation_dtype,
        default_mlstm_m_computation_dtype_recurrent=arg_mlstm_m_computation_dtype_recurrent
    )
    core = xlstm_scratch.xLSTMBlockStack(cfg_stack_scratch)
    core.out_dim = LM_CFG["hidden_dim"]
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
def make_general_hybrid_lm(vocab_size: int, pad_idx: int, block_string: str):
    cfg = GeneralHybridConfig(
        vocab_size=vocab_size,
        embed_dim=LM_CFG["embed_dim"],
        hidden_dim=LM_CFG["hidden_dim"], 
        block_string=block_string,
        context_length=LM_CFG["bptt"],
        n_heads=LM_CFG["n_heads"],
        dropout=LM_CFG["dropout"],
        pad_idx=pad_idx
    )
    return GeneralHybridLM(cfg)

def _train_and_eval_model(
    model_name: str,
    model_instance: nn.Module,
    train_data, val_data, test_data,
    current_lm_cfg, pad_idx, freq_map, device_str, device
) -> Tuple[Optional[List[float]], Optional[Dict[str, List[float]]]]: # Return PPLs and History
    """Helper function to train and evaluate a single model, returns PPLs and training history."""
    print(f"\nProcessing {model_name} for LM task...")
    params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    print(f"Param count: {params/1e6:.2f}M")

    trained_model, best_val_ppl, training_history = train_one_lm_model(
        model_name, model_instance, train_data, val_data, current_lm_cfg, pad_idx, device_str
    )

    if trained_model is None:

        num_final_metrics = len(current_lm_cfg["bin_edges"]) + 1 + 1 
        return [float('nan')] * num_final_metrics, training_history


    print(f"\nEvaluating {model_name} with its best state...")
    trained_model.to(device) 

    bin_ppls = evaluate_bins_ppl(trained_model, test_data, current_lm_cfg["bptt"], device_str,
                                 pad_idx=pad_idx, freq_vec=freq_map,
                                 bin_edges=current_lm_cfg["bin_edges"])

    overall_ppl_test = evaluate_overall_ppl(trained_model, test_data, current_lm_cfg["bptt"], device_str, pad_idx)


    model_results_ppl = bin_ppls + [overall_ppl_test]

    overall_str_test = f"{overall_ppl_test:.2f}" if np.isfinite(overall_ppl_test) else "N/A"
    print(f"{model_name} - Test Bin PPLs: "
        f"{[f'{p:.2f}' if np.isfinite(p) else 'N/A' for p in bin_ppls]}, "
        f"Test Overall PPL: {overall_str_test}")
    return model_results_ppl, training_history


# def run_lm_benchmark(device_str: str, benchmark_type: str = "xlstm_vs_lib"):
#     print(f"\n--- Running LM Benchmark ({benchmark_type}) on WikiText-2 on {device_str} ---")
#     current_lm_cfg = LM_CFG.copy()
#     current_lm_cfg["device"] = device_str
#     device = torch.device(device_str)

#     vocab_size, pad_idx, freq_map, train_data, val_data, test_data = load_wikitext2_data(current_lm_cfg)

#     models_to_run_lm: Dict[str, nn.Module] = {}
#     plot_title = ""
#     plot_filename_suffix = ""

#     if benchmark_type == "xlstm_vs_lib":
#         models_to_run_lm = {
#             "Scratch-xLSTM (LM)": make_scratch_xlstm_lm(vocab_size, pad_idx),
#             "Library-xLSTM (LM)": make_lib_xlstm_lm(vocab_size, pad_idx),
#         }
#         plot_title = "LM: Scratch xLSTM vs. Library xLSTM (PPL)"
#         plot_filename_suffix = "xlstm_vs_lib"
#     elif benchmark_type == "baselines_vs_xlstm":
#         models_to_run_lm = {
#             "LSTM (LM)": make_lstm_lm(vocab_size, pad_idx),
#             "Library-xLSTM (LM)": make_lib_xlstm_lm(vocab_size, pad_idx),
#             "Transformer (LM)": make_transformer_lm(vocab_size, pad_idx),
#         }
#         plot_title = "LM: Baselines vs. xLSTM (PPL)"
#         plot_filename_suffix = "baselines_vs_xlstm"
#     else:
#         raise ValueError(f"Unknown LM benchmark_type: {benchmark_type}")

#     trained_models_lm: Dict[str, torch.nn.Module] = {}
#     for name, model_instance in models_to_run_lm.items():
#         print(f"\nTraining {name} for LM task ({benchmark_type})...")
#         params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
#         print(f"Param count: {params/1e6:.2f}M")
        
#         trained_model, best_ppl = train_one_lm_model(name, model_instance, train_data, val_data, current_lm_cfg, pad_idx, device_str)
#         if trained_model is not None:
#             trained_models_lm[name] = trained_model
#         else:
#             print(f"Training failed for {name}. It will be excluded from evaluation.")


#     ppl_results_lm: Dict[str, List[float]] = {}
#     if not trained_models_lm:
#         print(f"No LM models finished training successfully for benchmark '{benchmark_type}'. Skipping LM evaluation.")
#         return {}
        
#     for name, model_instance in trained_models_lm.items():
#         print(f"\nEvaluating {name} for LM task ({benchmark_type})...")
#         model_instance.to(device)
        
#         bin_ppls = evaluate_bins_ppl(model_instance, test_data, current_lm_cfg["bptt"], device_str, 
#                                      pad_idx=pad_idx, freq_vec=freq_map, # freq_map will be moved to device in func
#                                      bin_edges=current_lm_cfg["bin_edges"])
#         overall_ppl = evaluate_overall_ppl(model_instance, test_data, current_lm_cfg["bptt"], device_str, pad_idx)
        
#         ppl_results_lm[name] = bin_ppls + [overall_ppl]
        
#         overall_str = f"{overall_ppl:.2f}" if np.isfinite(overall_ppl) else "N/A"
#         print(f"{name} - Bin PPLs: "
#             f"{[f'{p:.2f}' if np.isfinite(p) else 'N/A' for p in bin_ppls]}, "
#             f"Overall PPL: {overall_str}")
#     plot_lm_results(ppl_results_lm, 
#                     current_lm_cfg["bin_edges"], 
#                     plot_scale_first_bin=current_lm_cfg["plot_scale_first_bin"],
#                     filename=f"lm_benchmark_{plot_filename_suffix}_results.png")
    
#     results_path = f"lm_benchmark_{plot_filename_suffix}_results.json"
#     with open(results_path, "w") as f:
#         json_compatible_results = {
#             k: [None if np.isnan(val) else (float('inf') if np.isinf(val) else val) for val in v] 
#             for k, v in ppl_results_lm.items()
#         }
#         json.dump(json_compatible_results, f, indent=2)
#     print(f"LM PPL results saved to {results_path}")
    
#     return ppl_results_lm



def run_lm_benchmark(device_str: str, benchmark_type: str = "xlstm_vs_lib"):
    print(f"\n--- Running LM Benchmark ({benchmark_type}) on WikiText-2 on {device_str} ---")
    current_lm_cfg = LM_CFG.copy()
    current_lm_cfg.setdefault("log_interval", 100) 
    current_lm_cfg["device"] = device_str
    device = torch.device(device_str)

    vocab_size, pad_idx, freq_map, train_data, val_data, test_data = load_wikitext2_data(current_lm_cfg)

    models_to_run_lm: Dict[str, nn.Module] = {}
    plot_title_template_final_ppl = "LM Ablation: Baseline vs. {variant_name} (Final PPL)"
    plot_title_template_training = "LM Training: Baseline vs. {variant_name}"
    final_results_for_json = {}
    all_training_histories = {} 

    if benchmark_type == "xlstm_vs_lib" or benchmark_type == "baselines_vs_xlstm":

        current_plot_filename_suffix = benchmark_type
        if benchmark_type == "xlstm_vs_lib":
            models_to_run_lm = {
                "Scratch-xLSTM (LM)": make_scratch_xlstm_lm(vocab_size, pad_idx),
                "Library-xLSTM (LM)": make_lib_xlstm_lm(vocab_size, pad_idx),
            }
            plot_title_template_final_ppl = "LM: Scratch xLSTM vs. Library xLSTM (Final PPL)"
        elif benchmark_type == "baselines_vs_xlstm":
            models_to_run_lm = {
                "LSTM (LM)": make_lstm_lm(vocab_size, pad_idx),
                "Library-xLSTM (LM)": make_lib_xlstm_lm(vocab_size, pad_idx),
                "Transformer (LM)": make_transformer_lm(vocab_size, pad_idx),
            }
            plot_title_template_final_ppl = "LM: Baselines vs. xLSTM (Final PPL)"

        temp_histories_for_plot = {}
        for name, model_instance in models_to_run_lm.items():
            model_ppls, model_history = _train_and_eval_model(name, model_instance, train_data, val_data, test_data, current_lm_cfg, pad_idx, freq_map, device_str, device)
            if model_ppls:
                final_results_for_json[name] = model_ppls
            if model_history: 
                all_training_histories[name] = model_history
                temp_histories_for_plot[name] = model_history 

        if final_results_for_json:
            plot_lm_results(final_results_for_json,
                            current_lm_cfg["bin_edges"],
                            plot_scale_first_bin=current_lm_cfg["plot_scale_first_bin"],
                            filename=f"lm_benchmark_{current_plot_filename_suffix}_final_ppl.png",
                            title_override=plot_title_template_final_ppl)
        if temp_histories_for_plot:
             plot_lm_training_dynamics(temp_histories_for_plot,
                                      title=f"LM Training Dynamics: {benchmark_type.replace('_', ' ').title()}",
                                      filename=f"lm_benchmark_{current_plot_filename_suffix}_training.png",
                                      max_steps_display=current_lm_cfg["max_steps"])


    elif benchmark_type == "lm_stabilization_ablation":
        print("\n--- Running LM Stabilization Ablation Experiments ---")
        current_plot_filename_suffix = "stabilization_ablation"

        baseline_model_name = "Scratch-xLSTM (Stabilized)"

        model_training_repetition   = 4

        ####
        baseline_model_instance = make_scratch_xlstm_lm(vocab_size, pad_idx)
        baseline_ppl_results, baseline_history = _train_and_eval_model(
            baseline_model_name, baseline_model_instance,
            train_data, val_data, test_data, current_lm_cfg, pad_idx, freq_map, device_str, device
        )
        if baseline_ppl_results:
            final_results_for_json[baseline_model_name] = baseline_ppl_results
        if baseline_history: 
            all_training_histories[baseline_model_name] = baseline_history
            print(f"Baseline model {baseline_model_name} failed to produce training history. Skipping ablation studies.")
            return {}

        if baseline_ppl_results is None: 
            print(f"Baseline model {baseline_model_name} failed to train (no PPLs). Skipping ablation studies.")
            return {}


        stabilization_ablations = [
            {
                "name": "No Epsilon Stabilization",
                "suffix": "no_eps",
                "flags": {
                    "arg_mlstm_use_eps_in_parallel_norm_denom": False,
                    "arg_mlstm_use_eps_in_recurrent_h_denom": False
                }
            },
            {
                "name": "No MaxLogD Subtraction",
                "suffix": "no_maxlogd",
                "flags": {
                    "arg_mlstm_use_max_log_d_subtraction": False
                }
            },
            {
                "name": "No m_new Subtractions (Recurrent Gates)",
                "suffix": "no_mnew_sub",
                "flags": {
                    "arg_mlstm_use_m_new_subtraction_recurrent_gates": False
                }
            },
            {
                "name": "Simplified Normalization",
                "suffix": "no_robust_norms",
                "flags": {
                    "arg_mlstm_use_robust_normalizer_parallel": False,
                    "arg_mlstm_use_robust_h_denom_recurrent": False,
                    "arg_mlstm_use_robust_m_new_recurrent": False
                }
            },
            {
                "name": "No Dtype Casts",
                "suffix": "no_dtype_casts",
                "flags": {
                    "arg_mlstm_max_log_d_computation_dtype": None,
                    "arg_mlstm_m_computation_dtype_recurrent": None
                }
            }
        ]


        for ablation in stabilization_ablations:
            variant_model_name = f"Scratch-xLSTM ({ablation['name']})"
            variant_ppl_results, variant_history = [],[]
            for repetition_count in range(model_training_repetition):
                variant_model_instance = make_scratch_xlstm_lm(vocab_size, pad_idx, **ablation["flags"])
                one_time_variant_ppl_results, one_time_variant_history =  _train_and_eval_model(
                    variant_model_name, variant_model_instance,
                    train_data, val_data, test_data, current_lm_cfg, pad_idx, freq_map, device_str, device
                )
                
                variant_ppl_results.append(one_time_variant_ppl_results)
                variant_history.append(one_time_variant_history)

                if variant_ppl_results:
                    final_results_for_json[variant_model_name+f'_{repetition_count}'] = one_time_variant_ppl_results
                if variant_history: 
                    all_training_histories[variant_model_name+f'_{repetition_count}'] = one_time_variant_history
            if baseline_ppl_results and variant_ppl_results:
                mean_variant_ppl_results = []
                if variant_ppl_results:
                    variant_ppl_array = np.array(variant_ppl_results, dtype=float)
                    if variant_ppl_array.ndim == 2 and variant_ppl_array.shape[0] > 0:
                        mean_variant_ppl_results = np.nanmean(variant_ppl_array, axis=0).tolist()
                    elif baseline_ppl_results:
                        mean_variant_ppl_results = [np.nan] * len(baseline_ppl_results)
                else:
                    if baseline_ppl_results:
                        mean_variant_ppl_results = [np.nan] * len(baseline_ppl_results)

                current_final_ppl_plot_data = {
                    baseline_model_name: baseline_ppl_results,
                    variant_model_name: mean_variant_ppl_results
                }
                
                plot_lm_results(current_final_ppl_plot_data,
                                current_lm_cfg["bin_edges"],
                                plot_scale_first_bin=current_lm_cfg["plot_scale_first_bin"],
                                filename=f"lm_benchmark_ablation_{ablation['suffix']}_final_ppl.png",
                                title_override=plot_title_template_final_ppl.format(variant_name=ablation['name']))

            if baseline_history and variant_history:  # Using 'variant_history'
                
                variant_actual_max_step = 0
                if variant_history:
                    all_variant_run_steps = []
                    for history_run in variant_history: # history_run is a Dict
                        if history_run and history_run.get("steps"):
                            all_variant_run_steps.extend(history_run.get("steps"))
                    if all_variant_run_steps:
                        variant_actual_max_step = np.max(all_variant_run_steps)

                dynamic_plot_limit = 0
                if variant_actual_max_step > 0:
                    padding_steps = max(200, int(variant_actual_max_step * 0.20)) 
                    dynamic_plot_limit = variant_actual_max_step + padding_steps
                else: 
                    dynamic_plot_limit = max(500, current_lm_cfg.get("log_interval", 100) * 5)

                dynamic_plot_limit = min(dynamic_plot_limit, current_lm_cfg["max_steps"])

                current_training_plot_data_for_dynamics = {
                    baseline_model_name: baseline_history 
                }
                for i, single_run_history in enumerate(variant_history): 
                    current_training_plot_data_for_dynamics[f"{variant_model_name}_{i}"] = single_run_history
                
                plot_lm_training_dynamics(
                    current_training_plot_data_for_dynamics,
                    title=plot_title_template_training.format(variant_name=ablation['name']),
                    filename=f"lm_benchmark_ablation_{ablation['suffix']}_training.png",
                    max_steps_display=dynamic_plot_limit, 
                    baseline_model_name=baseline_model_name, 
                    total_max_steps_configured=None 
                )
    
    else:
        raise ValueError(f"Unknown LM benchmark_type: {benchmark_type}")
    def _is_nan_scalar(x):
        try:
            return bool(np.isnan(x))
        except TypeError:
            return True
    if final_results_for_json:
        results_path = f"lm_benchmark_{current_plot_filename_suffix}_results.json"
        with open(results_path, "w") as f:
            json_compatible_results = {
                k: [None if v_list is None
                        or idx >= len(v_list)
                        or _is_nan_scalar(v_list[idx])
                    else (float('inf') if np.isinf(v_list[idx]) else float(v_list[idx]))
                    for idx in range(len(LM_CFG["bin_edges"]) + 1 + 1)]
                for k, v_list in final_results_for_json.items()
            }

            json.dump(json_compatible_results, f, indent=2)
        print(f"All LM PPL results for '{benchmark_type}' saved to {results_path}")
