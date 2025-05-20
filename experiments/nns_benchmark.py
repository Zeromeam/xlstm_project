# experiments/nns_benchmark.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import xlstm_replica as xlstm_scratch 
from xlstm import (
    xLSTMBlockStack as LibXLSTMBlockStack,
    xLSTMBlockStackConfig as LibXLSTMBlockStackConfig,
    sLSTMBlockConfig as LibSLSTMBlockConfig,
    sLSTMLayerConfig as LibSLSTMLayerConfig,
    FeedForwardConfig as LibFeedForwardConfig
)
from utils.data_preparation import NNSDataset
from utils.plotting import plot_nns_results
from utils.model_architectures import TransformerModel_NNS, LSTMModel_NNS ,LlamaInspiredTransformerNNS
from utils.training_loops import make_lm_scheduler # Reusing the LM scheduler logic
from typing import Dict, Optional 
import copy
import math 

# NNS Specific Hyperparameters
NNS_MAX_LEN = 64
NNS_INPUT_DIM = 3
NNS_EMBED_DIM = 128
NNS_BATCH_SIZE = 64 
# NNS_EPOCHS = 10 
NNS_TOTAL_STEPS = 100_000 
NNS_VALID_STEPS = 500  
NNS_LR_FINAL = 1e-7    
NNS_LR_WARMUP_PCT = 0.1 
NNS_DROPOUT = 0.1 
NNS_WEIGHT_DECAY = 0.01 
NNS_GRAD_CLIP_NORM = 1.0 
NNS_PATIENCE = 10 

# --- NNS Model Definitions (ScratchxLSTM_NNS, LibxLSTM_NNS - remain the same) ---
class ScratchxLSTM_NNS(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(NNS_INPUT_DIM, NNS_EMBED_DIM)
        scratch_slstm_layer_cfg = xlstm_scratch.sLSTMLayerConfig(
            embedding_dim=NNS_EMBED_DIM, context_length=NNS_MAX_LEN,
            num_heads=4, conv1d_kernel_size=0, dropout=NNS_DROPOUT 
        )
        scratch_ffn_cfg = xlstm_scratch.FeedForwardConfig(
            proj_factor=1.0, act_fn="gelu", dropout=NNS_DROPOUT 
        )
        scratch_slstm_block_template = xlstm_scratch.sLSTMBlockConfig(
            slstm_layer_config=scratch_slstm_layer_cfg,
            feedforward_config=scratch_ffn_cfg
        )
        scratch_stack_cfg = xlstm_scratch.xLSTMBlockStackConfig(
            mlstm_block_template=None, slstm_block_template=scratch_slstm_block_template,
            num_blocks=2, embedding_dim=NNS_EMBED_DIM, context_length=NNS_MAX_LEN,
            dropout=NNS_DROPOUT, bias=False, add_post_blocks_norm=True, slstm_at="all"
        )
        self.body = xlstm_scratch.xLSTMBlockStack(config=scratch_stack_cfg)
        self.out = nn.Linear(NNS_EMBED_DIM, 1)

    def forward(self, x):
        h = self.inp(x)
        h = self.body(h)
        h_last = h[:, -1, :]
        output = self.out(h_last)
        return output.squeeze(-1)

class LibxLSTM_NNS(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(NNS_INPUT_DIM, NNS_EMBED_DIM)
        cfg = LibXLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=LibSLSTMBlockConfig(
                slstm=LibSLSTMLayerConfig(
                    num_heads=4, conv1d_kernel_size=0, dropout=NNS_DROPOUT, backend="vanilla"
                ),
                feedforward=LibFeedForwardConfig(proj_factor=1.0, act_fn="gelu", dropout=NNS_DROPOUT)
            ),
            context_length=NNS_MAX_LEN, num_blocks=2, embedding_dim=NNS_EMBED_DIM,
            add_post_blocks_norm=True, dropout=NNS_DROPOUT # Pass dropout to stack
        )
        self.body = LibXLSTMBlockStack(cfg)
        self.out = nn.Linear(NNS_EMBED_DIM, 1)

    def forward(self, x):
        h = self.body(self.inp(x))
        return self.out(h[:, -1]).squeeze(-1)

# Helper for NNS validation 
def evaluate_nns_mse(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    total_loss_sum = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.squeeze(-1).to(device)
            output = model(xb)
            loss = criterion(output, yb)
            total_loss_sum += loss.item() * xb.size(0)
            total_samples += xb.size(0)
    if total_samples == 0: return float('nan')
    return total_loss_sum / total_samples


def run_nns_benchmark(device_str: str, benchmark_type: str = "xlstm_vs_lib"):
    print(f"\n--- Running NNS Benchmark ({benchmark_type}) on {device_str} ---")
    device = torch.device(device_str)

    train_dataset = NNSDataset(8192, NNS_MAX_LEN, NNS_INPUT_DIM) 
    val_dataset = NNSDataset(8192, NNS_MAX_LEN, NNS_INPUT_DIM)

    train_loader = DataLoader(train_dataset, NNS_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if device_str.startswith("cuda") else False)
    val_loader = DataLoader(val_dataset, NNS_BATCH_SIZE, num_workers=2, pin_memory=True if device_str.startswith("cuda") else False)

    models_to_run: Dict[str, nn.Module] = {}
    model_specific_peak_lrs: Dict[str, float] = {}

    plot_title = ""
    plot_filename = ""

    if benchmark_type == "xlstm_vs_lib":
        models_to_run = {
            "xLSTM-Scratch (NNS)": ScratchxLSTM_NNS(),
            "xLSTM-Lib (NNS)": LibxLSTM_NNS(),
        }
        model_specific_peak_lrs = {
            "xLSTM-Scratch (NNS)": 1e-3,
            "xLSTM-Lib (NNS)": 1e-3,
        }
        plot_title = "NNS: Scratch xLSTM vs. Library xLSTM (Validation MSE)"
        plot_filename = "nns_benchmark_xlstm_vs_lib.png"
    elif benchmark_type == "baselines_vs_xlstm":
        models_to_run = {
            "LSTM (NNS)": LSTMModel_NNS(NNS_INPUT_DIM, NNS_EMBED_DIM),
            "xLSTM-Lib (NNS)": LibxLSTM_NNS(),
            #"Transformer (NNS)": TransformerModel_NNS(NNS_INPUT_DIM, NNS_EMBED_DIM, NNS_MAX_LEN),
            "LlamaInspiredTransformerNNS (NNS)": LlamaInspiredTransformerNNS(NNS_INPUT_DIM, NNS_EMBED_DIM, NNS_MAX_LEN),
        }
        model_specific_peak_lrs = {
            "LSTM (NNS)": 1e-2,  
            "xLSTM-Lib (NNS)": 1e-3, 
            "Transformer (NNS)": 1e-2, 
            "LlamaInspiredTransformerNNS (NNS)": 1e-3,
        }
        plot_title = "NNS: Baselines vs. xLSTM (Validation MSE)"
        plot_filename = "nns_benchmark_baselines_vs_xlstm.png"
    else:
        raise ValueError(f"Unknown NNS benchmark_type: {benchmark_type}")
    
    results = {}
    criterion = nn.MSELoss()

    for name, model in models_to_run.items():
        current_model_peak_lr = model_specific_peak_lrs.get(name)
        if current_model_peak_lr is None:
            raise ValueError(f"Peak LR not defined for model: {name}")

        print(f"\nTraining {name} for NNS task for {NNS_TOTAL_STEPS} steps with Peak LR: {current_model_peak_lr:.0e}...")
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=current_model_peak_lr, weight_decay=NNS_WEIGHT_DECAY) # Use current_model_peak_lr
        scheduler = make_lm_scheduler(optimizer, NNS_TOTAL_STEPS, 
                                      warmup_pct=NNS_LR_WARMUP_PCT, 
                                      final_lr=NNS_LR_FINAL, 
                                      peak_lr=current_model_peak_lr) # Use current_model_peak_lr

        best_val_mse = float('inf')
        epochs_no_improve = 0 
        best_model_state = None
        
        current_step = 0
        epoch = 0 
        done_training = False

        # --- Training loop ---
        while not done_training:
            epoch += 1
            

            for xb, yb in train_loader:
                if current_step >= NNS_TOTAL_STEPS:
                    done_training = True
                    break
                model.train()
                xb, yb = xb.to(device), yb.squeeze(-1).to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                if NNS_GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), NNS_GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step() 
                current_step += 1

                if current_step % NNS_VALID_STEPS == 0:
                    val_mse = evaluate_nns_mse(model, val_loader, criterion, device)
                    train_mse_approx = loss.item() 
                    print(f"Step {current_step}/{NNS_TOTAL_STEPS}, Train MSE (batch): {train_mse_approx:.6f}, Val MSE: {val_mse:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_model_state = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= NNS_PATIENCE:
                            print(f"Early stopping for {name} triggered at step {current_step}.")
                            done_training = True
                            break
            if done_training:
                break
        
        print(f"Finished training {name}.")
        if best_model_state:
            model.load_state_dict(best_model_state)
            final_val_mse = evaluate_nns_mse(model, val_loader, criterion, device) 
            results[name] = final_val_mse 
            print(f"{name} - Best Validation MSE: {final_val_mse:.6f}")
        else: 
            final_val_mse = evaluate_nns_mse(model, val_loader, criterion, device)
            results[name] = final_val_mse
            print(f"{name} - Final Validation MSE (no best state loaded, or training ended): {final_val_mse:.6f}")


    plot_nns_results(results, 
                     title=plot_title, 
                     ylabel="Validation MSE (lower is better)",
                     filename=plot_filename)
    return results