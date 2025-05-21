import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader 

from .data_preparation import get_lm_batch 

# --- NNS Evaluation  ---
def evaluate_nns_mse(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    total_loss_sum = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.squeeze(-1).to(device)
            output = model(xb)
            loss = criterion(output, yb) # MSELoss calculates mean by default
            total_loss_sum += loss.item() * xb.size(0) # Sum of losses
            total_samples += xb.size(0)
    if total_samples == 0: return float('nan')
    return total_loss_sum / total_samples


# --- LM Training and Evaluation ---
def make_lm_scheduler(optimizer: optim.Optimizer, total_steps: int, 
                      warmup_pct: float, final_lr: float, peak_lr: float):
    warmup_steps = int(total_steps * warmup_pct)
    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps) 
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps -1) 
        progress = min(1.0, max(0.0, progress)) 
        scale = (1 - final_lr / peak_lr)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return final_lr / peak_lr + scale * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def train_one_lm_model(model_name: str, model: nn.Module, 
                       train_data: torch.Tensor, val_data: torch.Tensor, 
                       cfg: Dict, pad_idx: int, device: str):
    bptt = cfg["bptt"]
    vocab_size = model.lm_head.out_features 

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr_peak"], weight_decay=cfg.get("weight_decay", 0.01), eps=1e-6)
    scheduler = make_lm_scheduler(optimizer, cfg["max_steps"], 
                                  warmup_pct=cfg["warmup_pct"], 
                                  final_lr=cfg["final_lr"], 
                                  peak_lr=cfg["lr_peak"])
    scaler_enabled = device.startswith("cuda") and hasattr(torch.cuda.amp, 'GradScaler')
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled) if scaler_enabled else None

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")

    step = 0
    epoch = 0
    best_val_ppl = float("inf")
    best_state = None
    epochs_no_improve = 0
    
    pbar = tqdm(total=cfg["max_steps"], desc=f"[{model_name}] Training", unit="step", leave=True)

    while step < cfg["max_steps"]:
        model.train()
        epoch_total_loss_sum = 0.0
        epoch_total_tokens = 0
        hidden_state = None 

        for i in range(0, train_data.size(1) - 1, bptt):
            if step >= cfg["max_steps"]: break
            
            t0 = time.perf_counter()
            x_batch, y_batch = get_lm_batch(train_data, i, bptt)
            if x_batch is None: continue
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            autocast_enabled = device.startswith("cuda") and scaler is not None
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits, hidden_state_next = model(x_batch, hidden_state)
                loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            
            if hidden_state_next is not None:
                if isinstance(hidden_state_next, tuple): 
                    hidden_state = tuple(h.detach() for h in hidden_state_next)
                elif isinstance(hidden_state_next, list) and all(isinstance(s, dict) for s in hidden_state_next): 
                    hidden_state = [{k: v.detach() for k, v in s_dict.items()} for s_dict in hidden_state_next]
                elif hasattr(hidden_state_next, 'detach'):
                    hidden_state = hidden_state_next.detach()
            
            num_tokens_batch = (y_batch != pad_idx).sum().item()
            batch_loss_sum = loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                #print(f"\nWarning: NaN or Inf loss at step {step} for {model_name}. Loss: {loss.item()}. Skipping update.")
                pbar.update()
                step += 1
                if step >= cfg["max_steps"]: break
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else: 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
            
            scheduler.step()
            
            batch_time = time.perf_counter() - t0
            current_lr = scheduler.get_last_lr()[0]
            
            pbar.update()
            display_loss = batch_loss_sum / max(1, num_tokens_batch) if num_tokens_batch > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{display_loss:.4f}", "lr": f"{current_lr:.2e}", "b_time": f"{batch_time:.3f}s"
            })

            epoch_total_loss_sum += batch_loss_sum
            epoch_total_tokens += num_tokens_batch
            step += 1
            if step >= cfg["max_steps"]: break
        
        avg_epoch_train_loss = epoch_total_loss_sum / max(1, epoch_total_tokens) if epoch_total_tokens > 0 else float('inf')
        train_ppl = math.exp(min(avg_epoch_train_loss, 700))

        val_ppl_overall = evaluate_overall_ppl(model, val_data, bptt, device, pad_idx)
        
        if np.isnan(val_ppl_overall):
             #print(f"\nWarning: {model_name} Val PPL is NaN at end of epoch {epoch+1}, step {step}.")
             val_ppl_overall = float('inf')

        tqdm.write(f"[{model_name}] Ep {epoch+1:2} Step {step}/{cfg['max_steps']} | Train PPL {train_ppl:7.2f} | Val PPL {val_ppl_overall:7.2f} | LR {current_lr:.2e}")
        epoch += 1

        if val_ppl_overall < best_val_ppl:
            best_val_ppl = val_ppl_overall
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            if step > cfg["max_steps"]*.1 :
                epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                tqdm.write(f"[{model_name}] Early stopping after {cfg['patience']} epochs of no Val PPL improvement.")
                break
    pbar.close()
    
    if best_state is not None:
        model.load_state_dict(best_state)
        final_eval_ppl = evaluate_overall_ppl(model, val_data, bptt, device, pad_idx)
        tqdm.write(f"[{model_name}] Loaded best model state with Val PPL: {final_eval_ppl:.2f}")
        return model, final_eval_ppl
    elif (best_val_ppl == float('inf') or np.isnan(best_val_ppl)) and step >= cfg["max_steps"]:
         tqdm.write(f"[{model_name}] Training finished, but no valid best state (Val PPL: {best_val_ppl}). Using last state.")
         return model, evaluate_overall_ppl(model, val_data, bptt, device, pad_idx)
    else:
         tqdm.write(f"[{model_name}] Training did not converge to a valid state (best Val PPL: {best_val_ppl}).")
         return None, best_val_ppl

def assign_frequency_buckets(token_ids: torch.Tensor, freq_vec: torch.Tensor, bin_edges_tensor: torch.Tensor):
    freq = freq_vec.to(token_ids.device)[token_ids] 
    buckets = torch.zeros_like(token_ids, dtype=torch.long) 
    mask_bin1 = (freq >= bin_edges_tensor[0]) & (freq < bin_edges_tensor[1])
    mask_bin2 = freq >= bin_edges_tensor[1]
    buckets[mask_bin1] = 1
    buckets[mask_bin2] = 2
    return buckets

def evaluate_bins_ppl(model: nn.Module, data: torch.Tensor, bptt: int, device: str, 
                      pad_idx: int, freq_vec: torch.Tensor, bin_edges: Tuple[int, int]):
    if model is None: return [float('nan')] * (len(bin_edges) + 1)
    
    model.eval()
    vocab_size = model.lm_head.out_features
    num_bins = len(bin_edges) + 1
    loss_sums = torch.zeros(num_bins, dtype=torch.double, device=device)
    counts = torch.zeros(num_bins, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)
    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.long, device=device) 
    
    hidden_state = None
    with torch.no_grad():
        for i in range(0, data.size(1) - 1, bptt):
            x_batch, y_batch = get_lm_batch(data, i, bptt)
            if x_batch is None: continue
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            autocast_enabled = device.startswith("cuda") and hasattr(torch.cuda.amp, 'autocast')
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                try:
                    logits, hidden_state_next = model(x_batch, hidden_state)
                    if hidden_state_next is not None: 
                        if isinstance(hidden_state_next, tuple): hidden_state = tuple(h.detach() for h in hidden_state_next)
                        elif isinstance(hidden_state_next, list) and all(isinstance(s, dict) for s in hidden_state_next): hidden_state = [{k: v.detach() for k, v in s_dict.items()} for s_dict in hidden_state_next]
                        elif hasattr(hidden_state_next, 'detach'): hidden_state = hidden_state_next.detach()
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        #print(f"Warning: NaN/Inf in model output during eval_bins (batch {i}). Skipping.")
                        continue
                    
                    token_losses = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
                    if torch.isnan(token_losses).any() or torch.isinf(token_losses).any():
                        #print(f"Warning: NaN/Inf in token_losses during eval_bins (batch {i}). Skipping.")
                        continue
                    token_losses = token_losses.view_as(y_batch)
                except Exception as e:
                    print(f"Error during model forward/loss in eval_bins (batch {i}): {e}")
                    continue

            bucket_ids = assign_frequency_buckets(y_batch, freq_vec, bin_edges_tensor)
            
            for k_bin in range(num_bins):
                mask = (bucket_ids == k_bin) & (y_batch != pad_idx)
                loss_sums[k_bin] += token_losses[mask].sum()
                counts[k_bin] += mask.sum().long()
                
    valid_counts = counts.clamp_min(1) 
    avg_losses = loss_sums / valid_counts
    avg_losses = torch.clamp(avg_losses, max=700) 
    ppls = torch.exp(avg_losses)
    ppls[counts == 0] = float('nan') 
    return ppls.tolist()

def evaluate_overall_ppl(model: nn.Module, data: torch.Tensor, bptt: int, device: str, pad_idx: int):
    if model is None: return float('nan')
    
    model.eval()
    vocab_size = model.lm_head.out_features
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_idx)
    total_loss_sum = 0.0
    total_tokens = 0
    hidden_state = None
    
    with torch.no_grad():
        for i in range(0, data.size(1) - 1, bptt):
            x_batch, y_batch = get_lm_batch(data, i, bptt)
            if x_batch is None: continue
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            autocast_enabled = device.startswith("cuda") and hasattr(torch.cuda.amp, 'autocast')
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                try:
                    logits, hidden_state_next = model(x_batch, hidden_state)
                    if hidden_state_next is not None: 
                        if isinstance(hidden_state_next, tuple): hidden_state = tuple(h.detach() for h in hidden_state_next)
                        elif isinstance(hidden_state_next, list) and all(isinstance(s, dict) for s in hidden_state_next): hidden_state = [{k: v.detach() for k, v in s_dict.items()} for s_dict in hidden_state_next]
                        elif hasattr(hidden_state_next, 'detach'): hidden_state = hidden_state_next.detach()
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        #print(f"Warning: NaN/Inf in model output during evaluate_overall (batch {i}).")
                        return float('nan') 
                        
                    loss_val = criterion(logits.view(-1, vocab_size), y_batch.view(-1)).item()
                    if np.isnan(loss_val) or np.isinf(loss_val):
                        #print(f"Warning: NaN/Inf loss during evaluate_overall (batch {i}).")
                        return float('nan')
                    total_loss_sum += loss_val
                except Exception as e:
                    print(f"Error during model forward/loss in evaluate_overall (batch {i}): {e}")
                    return float('nan')
                    
            total_tokens += (y_batch != pad_idx).sum().item()
            
    if total_tokens == 0: return float('nan')
    avg_loss = total_loss_sum / total_tokens
    
    if np.isnan(avg_loss) or np.isinf(avg_loss) or avg_loss > 700:
        return float('inf') if avg_loss > 0 else float('nan')
        
    return math.exp(avg_loss)