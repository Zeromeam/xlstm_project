import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def plot_nns_results(results: Dict[str, float], title: str, ylabel: str, filename: str = "nns_results.png"):
    plt.figure(figsize=(8, 6))
    models = list(results.keys())
    plot_heights = []
    for m in models:
        val = results.get(m, float('nan'))
        plot_heights.append(val if np.isfinite(val) else 0)
    
    bars = plt.bar(models, plot_heights) 
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, model_name in enumerate(models):
        bar = bars[i]
        yval = bar.get_height() 
        original_val = results[model_name] 
        
        text_label = f"{original_val:.4f}" if np.isfinite(original_val) else "N/A"
        text_y_offset = 0.01 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) 
        text_y_pos = yval + text_y_offset if yval >= 0 else yval - text_y_offset

        plt.text(bar.get_x() + bar.get_width()/2.0, text_y_pos, text_label, 
                 va='bottom' if yval >= 0 else 'top', 
                 ha='center', fontsize=9)

    plt.tight_layout(pad=1.0)
    plt.savefig(filename, dpi=120)
    plt.show()
    print(f"NNS plot saved to {filename}")


def plot_lm_results(ppl_by_model: Dict[str, List[float]], bin_edges: Tuple[int, int], 
                    plot_scale_first_bin: float, filename: str = "lm_results.png"):
    labels = [f"Freq <{bin_edges[0]}", f"Freq {bin_edges[0]}-{bin_edges[1]}", f"Freq >{bin_edges[1]}", "Overall PPL"]
    model_names = list(ppl_by_model.keys())
    num_models = len(model_names)
    bar_width = 0.8 / max(1, num_models) 
    x_indices = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 7)) 
    colors = plt.cm.get_cmap('viridis', max(1, num_models)) 

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
       any(len(p)>0 and np.isfinite(p[0]) for p in ppl_by_model.values()):
        ylabel_text += f"\n(* First bin value scaled by 1/{int(plot_scale_first_bin)} for plotting)"
        
    ax.set_ylabel(ylabel_text, fontsize=12)
    ax.set_title("Language Modeling PPL by Token Frequency (WikiText-2 Test Set)", fontsize=14, pad=20)
    if num_models > 0 : ax.legend(fontsize=10, loc='upper right') # Only show legend if there are models
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
            min_plot_val = min(plotted_values_for_log)
            max_plot_val = max(val for p_list in ppl_by_model.values() if p_list for val in p_list if np.isfinite(val) and val > 0) # Max of original values for upper limit
            ax.set_yscale('log')
            ax.set_ylim(bottom=max(0.01, min_plot_val * 0.8), top=max_plot_val * 1.2) 
        else:
            print("Plot: No valid positive PPLs for log scale after considering scaling.")
    else:
        print("Plot: Not using log scale due to non-positive, NaN, or Inf PPL values.")

    fig.tight_layout(pad=1.0)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"LM PPL plot saved to {filename}")