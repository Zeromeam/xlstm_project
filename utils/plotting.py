import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union



from typing import Dict, List, Tuple, Optional


def _to_percent(arr: np.ndarray) -> np.ndarray:
    """Return values in 0-100 range (if they were 0-1)."""
    return arr*100 if arr.max() <= 1.0 else arr




def _mean_ci(x: List[float]) -> tuple[float, float]:
    """return mean and 95 % half-CI for a list of numbers"""
    a = np.asarray(x, dtype=float)
    m = float(np.nanmean(a))
    n = np.count_nonzero(~np.isnan(a))
    if n < 2:
        return m, 0.0
    s = float(np.nanstd(a, ddof=1))
    return m, 1.96 * s / np.sqrt(n)

def plot_nns_results(results: Dict[str, Union[float, List[float]]],
                     title: str,
                     ylabel: str,
                     filename: str = "nns_results.png"):
    """
    Accepts either a single float per model OR a list of floats (repetitions).
    When a list is given the mean is shown and a 95 % CI error-bar is drawn.
    """
    plt.figure(figsize=(8, 6))
    models = list(results.keys())
    means, errors = [], []

    for v in results.values():
        if isinstance(v, (list, tuple)):
            m, e = _mean_ci(v)
        else:                           # single run
            m, e = float(v), 0.0
        means.append(m)
        errors.append(e)

    bar_container = plt.bar(models, means, yerr=errors,
                            capsize=6, alpha=0.9, error_kw=dict(lw=1.2))
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # write numbers above bars (mean ± ci if available)
    ylim = plt.gca().get_ylim()
    yspan = ylim[1] - ylim[0]
    for bar, m, e in zip(bar_container, means, errors):
        txt = f"{m:.3f}" if e == 0 else f"{m:.3f}±{e:.3f}"
        plt.text(bar.get_x() + bar.get_width()/2, m + 0.02*yspan,
                 txt, ha="center", va="bottom", fontsize=9)

    plt.tight_layout(pad=1.0)
    plt.savefig(filename, dpi=150)
    plt.show()
    print("NNS plot saved to", filename)




def plot_lm_results(ppl_by_model: Dict[str, List[float]],
                    bin_edges: Tuple[int, int],
                    plot_scale_first_bin: float,
                    filename: str = "lm_results.png",
                    title_override: Optional[str] = None): # <-- ADD title_override HERE
    labels = [f"Freq <{bin_edges[0]}", f"Freq {bin_edges[0]}-{bin_edges[1]}", f"Freq >{bin_edges[1]}", "Overall PPL"]
    model_names = list(ppl_by_model.keys())
    num_models = len(model_names)
    bar_width = 0.8 / max(1, num_models)
    x_indices = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        colors = plt.cm.get_cmap('viridis', max(1, num_models))
    except AttributeError: 
        cm = plt.colormaps.get_cmap('viridis')
        colors = cm.resampled(max(1,num_models))


    for i, (model_name, ppl_values) in enumerate(ppl_by_model.items()):
        if not ppl_values:
            continue
        ppl_array = np.array(ppl_values, dtype=float)

        plot_values = ppl_array.copy()
        if len(plot_values) > 0 and np.isfinite(plot_values[0]) and plot_scale_first_bin != 0 and plot_scale_first_bin != 1.0:
            plot_values[0] /= plot_scale_first_bin

        plot_heights = np.nan_to_num(plot_values, nan=0.0, posinf=0.0, neginf=0.0)

        offset = (i - (num_models - 1) / 2.0) * bar_width
        rects = ax.bar(x_indices + offset, plot_heights, bar_width, label=model_name, color=colors(i/num_models if num_models > 0 else 0), alpha=0.9)

        for j, rect in enumerate(rects):
            if j >= len(ppl_array): continue
            original_val = ppl_array[j]
            display_height = rect.get_height()

            if np.isnan(original_val): text_label = "N/A"
            elif np.isinf(original_val): text_label = "Inf"
            else: text_label = f"{original_val:.1f}"

            if j == 0 and np.isfinite(original_val) and plot_scale_first_bin != 0 and plot_scale_first_bin != 1.0:
                text_label += "*"

            text_y_offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            text_y_pos = display_height + text_y_offset
            ax.text(rect.get_x() + rect.get_width() / 2.0, text_y_pos, text_label,
                    ha="center", va="bottom", fontsize=8, fontweight='normal', rotation=0)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=10)

    ylabel_text = "Perplexity (PPL) - Lower is Better"
    if plot_scale_first_bin != 0 and plot_scale_first_bin != 1.0 and \
       any(len(p)>0 and np.isfinite(p[0]) for p in ppl_by_model.values() if p): # Added check for p being non-empty
        ylabel_text += f"\n(* First bin value scaled by 1/{int(plot_scale_first_bin)} for plotting)"

    ax.set_ylabel(ylabel_text, fontsize=12)

    plot_title = title_override if title_override is not None else "Language Modeling PPL by Token Frequency (WikiText-2 Test Set)"
    ax.set_title(plot_title, fontsize=14, pad=20) 

    if num_models > 0 : ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_finite_positive_for_log = all(
        val > 0 and np.isfinite(val)
        for p_list in ppl_by_model.values() if p_list
        for val in p_list if not np.isnan(val)
    )

    if all_finite_positive_for_log:
        plotted_values_for_log = [
            (val / plot_scale_first_bin if idx == 0 and plot_scale_first_bin != 0 and plot_scale_first_bin != 1.0 else val)
            for p_list in ppl_by_model.values() if p_list
            for idx, val in enumerate(p_list)
            if np.isfinite(val) and val > 0
        ]
        plotted_values_for_log = [pv for pv in plotted_values_for_log if np.isfinite(pv) and pv > 0]

        if plotted_values_for_log: 
            min_plot_val = min(plotted_values_for_log) if plotted_values_for_log else 1.0
            original_positive_values = [val for p_list in ppl_by_model.values() if p_list for val in p_list if np.isfinite(val) and val > 0]
            max_plot_val = max(original_positive_values) if original_positive_values else 1000.0

            ax.set_yscale('log')
            ax.set_ylim(bottom=max(0.01, min_plot_val * 0.8), top=max_plot_val * 1.2)
        else:
            print("Plot: No valid positive PPLs for log scale after considering scaling.")
            ax.set_ylim(bottom=0) 
    else:
        print("Plot: Not using log scale due to non-positive, NaN, or Inf PPL values.")
        ax.set_ylim(bottom=0) 


    fig.tight_layout(pad=1.0)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"LM PPL plot saved to {filename}")


# def plot_formal_heatmaps(
#         results_by_model: Dict[str, List[float]],
#         task_names: Tuple[str, str, str] = (
#             "Parity (regular)",
#             "Dyck-1 (context-free)",
#             "aⁿbⁿcⁿ (context-sensitive)"
#         ),
#         filename: str = "formal_benchmark_heatmap.png"):
#     """
#     Args
#     ----
#     results_by_model  {model_name: [acc_parity, acc_dyck, acc_abc]}
#                       – *exactly* three floats per model.
#     """

#     import matplotlib.pyplot as plt
#     import numpy as np

#     models = list(results_by_model.keys())
#     data   = np.array([results_by_model[m] for m in models]).T   # (3, n_models)

#     # one row-heat-map per task
#     n_tasks = len(task_names)
#     fig, axes = plt.subplots(n_tasks, 1,
#                              figsize=(max(8, 0.6*len(models)), 2.5*n_tasks),
#                              sharex=True)

#     vmin, vmax = 0.0, 1.0   # accuracy is in [0,1]

#     for i, ax in enumerate(np.ravel(axes)):
#         im = ax.imshow(data[i:i+1, :], aspect='auto',
#                        cmap='viridis', vmin=vmin, vmax=vmax)

#         # y-tick: the task name
#         ax.set_yticks([0])
#         ax.set_yticklabels([task_names[i]])
#         # x-ticks: the model names
#         ax.set_xticks(np.arange(len(models)))
#         ax.set_xticklabels(models, rotation=15, ha='right')

#         # write the number on the cell
#         for j, val in enumerate(data[i]):
#             txt_colour = "white" if val < 0.5 else "black"
#             ax.text(j, 0, f"{val:.3f}",
#                     ha='center', va='center', color=txt_colour, fontsize=8)

#     fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.01,
#                  label="Accuracy")

#     fig.suptitle("Formal-language validation accuracy (per Chomsky level)",
#                  fontsize=14, y=1.02)
#     fig.tight_layout()
#     fig.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.show()
#     print(f"Formal-language heat-map saved to {filename}")

def plot_formal_heatmaps(
        results_by_model: Dict[str, List[Union[float, List[float]]]],
        filename: str = "formal_heatmap.png",
        rotation: int = 60,
        wrap_labels: bool = True,
        figsize_per_model: float = 1.0,
        value_range: tuple = (50, 100)):
    """
    Draw three one-row heat-maps (Parity, Dyck-1, aⁿbⁿcⁿ).

    Args
    ----
    results_by_model  {model: [acc_parity, acc_dyck, acc_abc]}
                      – values may be scalars or lists (→ μ ± CI).
    value_range       (vmin, vmax) for the colour scale **in percent**.
                      Defaults to (50, 100).
    """
    task_names = ["Parity (regular)",
                  "Dyck-1 (context-free)",
                  "aⁿbⁿcⁿ (context-sens.)"]

    # ── collect data ────────────────────────────────────────────────────────
    models = list(results_by_model.keys())
    if wrap_labels:
        models = [m.replace(" (", "\n(") for m in models]

    data, annot = [], []
    for t in range(3):
        row_vals, row_ann = [], []
        for m_disp in models:
            m = m_disp.replace("\n(", " (")          # undo wrapping
            v = results_by_model[m][t]
            mu, ci = _mean_ci(v) if isinstance(v, (list, tuple)) else (float(v), 0.0)
            # convert to percent ------------------------------------------------
            mu_p, ci_p = mu * 100.0, ci * 100.0
            row_vals.append(mu_p)
            row_ann.append(f"{mu_p:.1f}" if ci_p == 0.0 else f"{mu_p:.1f}±{ci_p:.1f}")
        data.append(row_vals); annot.append(row_ann)

    data = np.asarray(data)                           # shape (3, n_models)
    vmin, vmax = value_range

    # ── plot ────────────────────────────────────────────────────────────────
    fig_w = max(6, figsize_per_model * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    im = ax.imshow(data, aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(models)), models, rotation=rotation, ha="right")
    ax.set_yticks(np.arange(3), task_names)

    # cell annotations
    for i in range(3):
        for j in range(len(models)):
            v = data[i, j]
            ax.text(j, i, annot[i][j],
                    ha="center", va="center",
                    color="white" if v < (vmin + vmax) / 2 else "black",
                    fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=.02)
    cbar.set_label("Accuracy (%)")

    fig.suptitle("Formal-language validation accuracy ", y=1.05)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved →", filename)


def plot_mqar_capacity_panels(results_by_kv: Dict[int, Dict[str, Dict[int, Union[float, List[float]]]]],
                              *, context_len: int = 2048,
                              save_path: str = "mqar_capacity.png") -> None:
    """
    Accepts scalar or list values.  Lists are treated as repetitions;
    the mean is plotted with a 95 % CI error-bar.
    """
    kv_list   = sorted(results_by_kv.keys())
    n_panels  = len(kv_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4), sharey=True)
    axes      = [axes] if n_panels == 1 else axes

    cmap = plt.cm.get_cmap('tab10')
    model2colour = {}

    for ax, kv in zip(axes, kv_list):
        panel = results_by_kv[kv]
        for mi, (model, width2acc) in enumerate(panel.items()):
            colour = model2colour.setdefault(model, cmap(mi % 10))
            widths, means, errs = [], [], []
            for w in sorted(width2acc.keys()):
                v = width2acc[w]
                mu, ci = _mean_ci(v) if isinstance(v, (list, tuple)) else (float(v), 0.0)
                widths.append(w); means.append(mu); errs.append(ci)
            ax.errorbar(widths, means, yerr=errs, color=colour,
                        marker='o', capsize=4, label=model if kv==kv_list[0] else "")
        ax.set_xlabel("Model dimension")
        ax.set_title(f"{kv} KV pairs")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', alpha=.5)

    axes[0].set_ylabel("Validation accuracy")
    fig.suptitle(f"MQAR – memory capacity (context {context_len})", fontsize=14, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    fig.tight_layout(rect=[0,0,0.85,1])
    fig.savefig(save_path, dpi=150)
    plt.show()
    print("Saved →", save_path)

# ──────────────────────────────────────────────────────────────────────────────
def plot_mqar_heatmap(
        results_by_model: Dict[str, float],
        filename: str = "mqar_benchmark_heatmap.png",
        rotation: int = 60,
        wrap_labels: bool = True,
        figsize_per_model: float = 1.0,
        value_range: tuple = (0, 100)):
    """
    One-row heat-map for the MQAR benchmark (accuracy per model).
    """
    import numpy as np, matplotlib.pyplot as plt

    task_names = ["MQAR"]                       # single row
    models     = list(results_by_model.keys())

    if wrap_labels:
        models = [m.replace(" (", "\n(") for m in models]

    data = np.array([[results_by_model[m.replace("\n(", " (")]
                     for m in models]])
    data = _to_percent(data)               # NEW

    vmin, vmax = value_range               # NEW

    fig_w = max(6, figsize_per_model * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 2.4))

    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(models)), models, rotation=rotation, ha="right")
    ax.set_yticks([0], task_names)

    # numbers on the cells
    for j, v in enumerate(data[0]):
        ax.text(j, 0, f"{v:.3f}",
                ha="center", va="center",
                color="white" if v < .5 else "black",
                fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=.02)
    cbar.set_label("Accuracy")

    fig.suptitle("MQAR validation accuracy (row-normalised)", y=1.05)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved →", filename)


def plot_lm_training_dynamics(
    histories: Dict[str, Dict[str, List[float]]],
    title: str,
    filename: str = "lm_training_dynamics.png",
    max_steps_display: Optional[int] = None,
    loss_cap_display: float = 5000,
    ppl_cap_display: float = 100000,
    total_max_steps_configured: Optional[int] = None,
    baseline_model_name: Optional[str] = None, # New argument
    baseline_color: str = 'blue',              # Color for baseline
    variant_color: str = 'orange'              # Color for variants
):
    if not histories:
        print("No training histories to plot.")
        return

    all_original_steps_for_axis_calc = []
    for history in histories.values():
        steps_orig_list = history.get("steps", [])
        if isinstance(steps_orig_list, np.ndarray):
            steps_orig_list = steps_orig_list.tolist()
        if steps_orig_list:
            all_original_steps_for_axis_calc.extend(steps_orig_list)

    actual_max_step_in_original_data = 0
    if all_original_steps_for_axis_calc:
        actual_max_step_in_original_data = np.max(all_original_steps_for_axis_calc)

    final_x_axis_limit = actual_max_step_in_original_data
    xlabel_text = "Training Steps"

    if total_max_steps_configured is not None:
        final_x_axis_limit = total_max_steps_configured
        note = f"Axis to {total_max_steps_configured}"
        max_data_plotted_up_to = actual_max_step_in_original_data
        if max_steps_display is not None:
            max_data_plotted_up_to = min(actual_max_step_in_original_data, max_steps_display)
        
        if max_steps_display is not None and max_data_plotted_up_to < total_max_steps_configured :
             if actual_max_step_in_original_data > max_steps_display : 
                note += f", data to {max_data_plotted_up_to}"
        elif actual_max_step_in_original_data < total_max_steps_configured and max_steps_display is None:
             note += f", data to {actual_max_step_in_original_data}"
        xlabel_text += f" ({note})"
    elif max_steps_display is not None and actual_max_step_in_original_data > max_steps_display:
        final_x_axis_limit = max_steps_display 
        xlabel_text += f" (Data to {max_steps_display})"
    
    name_part, ext_part = "", ".png" 
    if '.' in filename:
        parts = filename.rsplit('.', 1)
        name_part = parts[0]
        ext_part = '.' + parts[1]
    else:
        name_part = filename

    loss_plot_filename = name_part + "_loss" + ext_part
    ppl_plot_filename = name_part + "_ppl" + ext_part

    use_two_color_scheme = baseline_model_name is not None

    # --- Plot 1: Training Loss ---
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6)) 
    
    # Color management
    if not use_two_color_scheme:
        try:
            colors_cmap_loss = plt.cm.get_cmap('viridis', len(histories))
        except AttributeError:
            cm_loss = plt.colormaps.get_cmap('viridis')
            colors_cmap_loss = cm_loss.resampled(len(histories))
    
    color_idx_loss = 0
    min_loss_overall, max_loss_overall = float('inf'), float('-inf')
    loss_legend_handles, loss_legend_labels = [], []
    
    baseline_legend_label_loss = f"{baseline_model_name} Train Loss" if baseline_model_name else "Baseline Train Loss"
    variant_legend_label_loss = "Variant Runs Train Loss" # Generic label for all variants

    for model_name, history in histories.items():
        model_color = None
        current_plot_label_loss = f"{model_name} Train Loss" # Default label

        if use_two_color_scheme:
            is_baseline = (model_name == baseline_model_name)
            model_color = baseline_color if is_baseline else variant_color
            current_plot_label_loss = baseline_legend_label_loss if is_baseline else variant_legend_label_loss
        else:
            model_color = colors_cmap_loss(color_idx_loss)

        steps_orig = np.array(history.get("steps", []))
        train_losses_orig = np.array(history.get("train_loss", []))
        steps_to_plot, train_losses_to_plot = steps_orig.copy(), train_losses_orig.copy()
        
        if max_steps_display is not None:
            if len(steps_orig) > 0:
                mask = steps_orig <= max_steps_display
                steps_to_plot, train_losses_to_plot = steps_orig[mask], train_losses_orig[mask]
            else:
                steps_to_plot, train_losses_to_plot = np.array([]), np.array([])
        
        if len(steps_to_plot) == 0:
            if current_plot_label_loss not in loss_legend_labels:
                dummy_line, = ax_loss.plot([], [], color=model_color, linestyle='-', label=current_plot_label_loss)
                loss_legend_handles.append(dummy_line)
                loss_legend_labels.append(current_plot_label_loss)
            if not use_two_color_scheme: color_idx_loss += 1
            continue
            
        finite_loss_mask = np.isfinite(train_losses_to_plot)
        
        label_for_plot = current_plot_label_loss if current_plot_label_loss not in loss_legend_labels else ""
        line, = ax_loss.plot(steps_to_plot[finite_loss_mask], train_losses_to_plot[finite_loss_mask], 
                               color=model_color, linestyle='-', label=label_for_plot)
        
        if current_plot_label_loss not in loss_legend_labels:
            loss_legend_handles.append(line)
            loss_legend_labels.append(current_plot_label_loss)
            
        if np.any(finite_loss_mask):
            min_loss_overall = min(min_loss_overall, np.min(train_losses_to_plot[finite_loss_mask]))
            max_loss_overall = max(max_loss_overall, np.max(train_losses_to_plot[finite_loss_mask]))
            
        invalid_loss_indices = np.where(np.logical_not(finite_loss_mask))[0]
        first_invalid_loss_idx = invalid_loss_indices[0] if len(invalid_loss_indices) > 0 else None
        exploded_label_loss, explosion_marker_loss_obj = "Loss Exploded", None
        
        if first_invalid_loss_idx is not None and first_invalid_loss_idx > 0:
            last_valid_step, last_valid_loss = steps_to_plot[first_invalid_loss_idx - 1], train_losses_to_plot[first_invalid_loss_idx - 1]
            explosion_step_loss = steps_to_plot[first_invalid_loss_idx]
            ax_loss.scatter(last_valid_step, last_valid_loss, marker='o', color=model_color, s=50, zorder=3)
            ax_loss.plot([last_valid_step, explosion_step_loss], [last_valid_loss, loss_cap_display * 0.95], linestyle=':', color=model_color, alpha=0.7, zorder=2)
            explosion_marker_loss_obj = ax_loss.scatter(explosion_step_loss, loss_cap_display*0.95, marker='x', color='red', s=100, zorder=5, label="Loss Exploded" if "Loss Exploded" not in loss_legend_labels else "") 
        elif first_invalid_loss_idx == 0: 
            explosion_marker_loss_obj = ax_loss.scatter(steps_to_plot[0], loss_cap_display*0.95, marker='x', color='red', s=100, zorder=5, label="Loss Exploded" if "Loss Exploded" not in loss_legend_labels else "")
        
        if explosion_marker_loss_obj and "Loss Exploded" not in loss_legend_labels:
            loss_legend_handles.append(explosion_marker_loss_obj)
            loss_legend_labels.append("Loss Exploded")
            
        if not use_two_color_scheme: color_idx_loss += 1

    ax_loss.set_xlabel(xlabel_text)
    ax_loss.set_ylabel("Training Loss (per token)")
    ax_loss.set_title(f"{title}: Training Loss Dynamics")
    if np.isfinite(min_loss_overall) and np.isfinite(max_loss_overall):
        ax_loss.set_ylim(max(0, min_loss_overall * 0.90), min(loss_cap_display, max_loss_overall * 1.10))
    else: ax_loss.set_ylim(0, loss_cap_display)
    if final_x_axis_limit == 0:
        is_loss_only_step_zero = (actual_max_step_in_original_data == 0 and histories and any(h.get("steps") and np.array_equal(h.get("steps"),[0]) for h in histories.values()))
        ax_loss.set_xlim(-0.5 if is_loss_only_step_zero else -1, 0.5 if is_loss_only_step_zero else 1)
    else: ax_loss.set_xlim(-0.05 * final_x_axis_limit, final_x_axis_limit * 1.05)
    if loss_legend_handles: ax_loss.legend(loss_legend_handles, loss_legend_labels, loc="upper right", fontsize=9)
    ax_loss.grid(True, linestyle=':', alpha=0.7)
    fig_loss.tight_layout()
    if filename: 
        fig_loss.savefig(loss_plot_filename, dpi=150)
        print(f"Training loss plot saved to {loss_plot_filename}")
    plt.show() 
    plt.close(fig_loss) 

    # --- Plot 2: Validation PPL ---
    fig_ppl, ax_ppl = plt.subplots(figsize=(10, 6))
    if not use_two_color_scheme:
        try:
            colors_cmap_ppl = plt.cm.get_cmap('viridis', len(histories)) 
        except AttributeError:
            cm_ppl = plt.colormaps.get_cmap('viridis')
            colors_cmap_ppl = cm_ppl.resampled(len(histories))
            
    color_idx_ppl = 0 # Used only if not two-color scheme
    min_ppl_overall, max_ppl_overall = float('inf'), float('-inf')
    ppl_legend_handles, ppl_legend_labels = [], []

    baseline_legend_label_ppl = f"{baseline_model_name} Val PPL" if baseline_model_name else "Baseline Val PPL"
    variant_legend_label_ppl = "Variant Runs Val PPL"

    for model_name, history in histories.items():
        model_color = None
        current_plot_label_ppl = f"{model_name} Val PPL" # Default label

        if use_two_color_scheme:
            is_baseline = (model_name == baseline_model_name)
            model_color = baseline_color if is_baseline else variant_color
            current_plot_label_ppl = baseline_legend_label_ppl if is_baseline else variant_legend_label_ppl
        else:
            model_color = colors_cmap_ppl(color_idx_ppl)

        steps_orig, val_ppls_orig = np.array(history.get("steps", [])), np.array(history.get("val_ppl", []))
        steps_to_plot, val_ppls_to_plot = steps_orig.copy(), val_ppls_orig.copy()
        
        if max_steps_display is not None:
            if len(steps_orig) > 0:
                mask = steps_orig <= max_steps_display
                steps_to_plot, val_ppls_to_plot = steps_orig[mask], val_ppls_orig[mask]
            else:
                steps_to_plot, val_ppls_to_plot = np.array([]), np.array([])
        
        if len(steps_to_plot) == 0: 
            if current_plot_label_ppl not in ppl_legend_labels:
                dummy_line, = ax_ppl.plot([], [], color=model_color, linestyle='--', label=current_plot_label_ppl)
                ppl_legend_handles.append(dummy_line)
                ppl_legend_labels.append(current_plot_label_ppl)
            if not use_two_color_scheme: color_idx_ppl +=1
            continue
            
        finite_ppl_mask = np.isfinite(val_ppls_to_plot)
        
        label_for_plot = current_plot_label_ppl if current_plot_label_ppl not in ppl_legend_labels else ""
        line, = ax_ppl.plot(steps_to_plot[finite_ppl_mask], val_ppls_to_plot[finite_ppl_mask], 
                              color=model_color, linestyle='--', label=label_for_plot)
        
        if current_plot_label_ppl not in ppl_legend_labels:
            ppl_legend_handles.append(line)
            ppl_legend_labels.append(current_plot_label_ppl)
            
        if np.any(finite_ppl_mask):
            min_ppl_overall = min(min_ppl_overall, np.min(val_ppls_to_plot[finite_ppl_mask]))
            max_ppl_overall = max(max_ppl_overall, np.max(val_ppls_to_plot[finite_ppl_mask]))
            
        # Explosion marker logic
        invalid_ppl_indices = np.where(np.logical_not(finite_ppl_mask))[0]
        first_invalid_ppl_idx = invalid_ppl_indices[0] if len(invalid_ppl_indices)> 0 else None
        exploded_label_ppl, explosion_marker_ppl_obj = "PPL Exploded", None # Corrected: This was exploded_label_loss before
        
        if first_invalid_ppl_idx is not None and first_invalid_ppl_idx > 0:
            last_valid_step_ppl, last_valid_ppl = steps_to_plot[first_invalid_ppl_idx - 1], val_ppls_to_plot[first_invalid_ppl_idx - 1]
            explosion_step_ppl = steps_to_plot[first_invalid_ppl_idx]
            ax_ppl.scatter(last_valid_step_ppl, last_valid_ppl, marker='o', s=50, facecolors='none', edgecolors=model_color, zorder=3) # Use model_color for marker edge
            ax_ppl.plot([last_valid_step_ppl, explosion_step_ppl], [last_valid_ppl, ppl_cap_display * 0.95], linestyle=':', color=model_color, alpha=0.7, zorder=2) # Use model_color for dashed line
            explosion_marker_ppl_obj = ax_ppl.scatter(explosion_step_ppl, ppl_cap_display * 0.95, marker='x', color='blue', s=100, zorder=5, label="PPL Exploded" if "PPL Exploded" not in ppl_legend_labels else "") # Fixed color for explosion 'x'
        elif first_invalid_ppl_idx == 0: # Explosion at the very first step
             explosion_marker_ppl_obj = ax_ppl.scatter(steps_to_plot[0], ppl_cap_display * 0.95, marker='x', color='blue', s=100, zorder=5, label="PPL Exploded" if "PPL Exploded" not in ppl_legend_labels else "")
        
        if explosion_marker_ppl_obj and "PPL Exploded" not in ppl_legend_labels: # Corrected: This was exploded_label_loss before
            ppl_legend_handles.append(explosion_marker_ppl_obj)
            ppl_legend_labels.append("PPL Exploded") # Corrected: This was exploded_label_loss before
            
        if not use_two_color_scheme: color_idx_ppl += 1
        
    ax_ppl.set_xlabel(xlabel_text)
    ax_ppl.set_ylabel("Validation PPL")
    ax_ppl.set_title(f"{title}: Validation PPL Dynamics")
    if np.isfinite(min_ppl_overall) and np.isfinite(max_ppl_overall):
         ax_ppl.set_ylim(max(1, min_ppl_overall * 0.85), min(ppl_cap_display, max_ppl_overall * 1.15))
         if min_ppl_overall > 1 and max_ppl_overall > 1 and (max_ppl_overall / min_ppl_overall > 10): 
             ax_ppl.set_yscale('log')
    else: ax_ppl.set_ylim(1, ppl_cap_display )
    if final_x_axis_limit == 0:
        is_ppl_only_step_zero = (actual_max_step_in_original_data == 0 and histories and any(h.get("steps") and np.array_equal(h.get("steps"),[0]) for h in histories.values()))
        ax_ppl.set_xlim(-0.5 if is_ppl_only_step_zero else -1, 0.5 if is_ppl_only_step_zero else 1)
    else: ax_ppl.set_xlim(-0.05 * final_x_axis_limit, final_x_axis_limit * 1.05)
    if ppl_legend_handles: ax_ppl.legend(ppl_legend_handles, ppl_legend_labels, loc="upper right", fontsize=9)
    ax_ppl.grid(True, linestyle=':', alpha=0.7)
    fig_ppl.tight_layout()
    if filename: 
        fig_ppl.savefig(ppl_plot_filename, dpi=150)
        print(f"Validation PPL plot saved to {ppl_plot_filename}")
    plt.show() 
    plt.close(fig_ppl)