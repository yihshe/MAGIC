"""
RTM Final Sensitivity Analysis - Matching Reference Style

Generates gradient plots with:
- Square-ish plot layout (6.5x5 figure)
- Colorbar with actual min/max values from data
- Mean gradient and saturation rate in title
- LaTeX formatting
- fc marked in title
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Add MAGIC to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from physics.rtm.rtm import RTM

# Constants
SD = 500.0
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1',
                 'B06_RE2', 'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B10',
                 'B11_SWI1', 'B12_SWI2']
BANDS_INDEX = [i for i in range(len(S2_FULL_BANDS)) if S2_FULL_BANDS[i] not in ['B01', 'B10']]
S2_BANDS_USED = [S2_FULL_BANDS[i] for i in BANDS_INDEX]

BAND_LABELS = {
    'B02_BLUE': 'Blue', 'B03_GREEN': 'Green', 'B04_RED': 'Red',
    'B05_RE1': 'RE1', 'B06_RE2': 'RE2', 'B07_RE3': 'RE3',
    'B08_NIR1': 'NIR1', 'B8A_NIR2': 'NIR2', 'B09_WV': 'WV',
    'B11_SWI1': 'SWI1', 'B12_SWI2': 'SWI2'
}

STD_GRAD_THRESHOLD = 0.001


def fc_to_cd_h(fc, sd=SD):
    """Convert fractional cover to crown diameter and tree height."""
    cd = np.sqrt((fc * 10000) / (np.pi * sd)) * 2
    h = np.exp(2.117 + 0.507 * np.log(cd))
    return cd, h


def get_default_paras(rtm: RTM) -> dict:
    """Return fresh copy of RTM default parameters."""
    rtm.para_init()
    defaults = {k: v.clone().detach().cpu() for k, v in rtm.para_dict.items()}
    return defaults


def run_grid_at_fc(rtm: RTM, defaults: dict, lai_vals: np.ndarray, 
                   laiu_vals: np.ndarray, fc: float) -> np.ndarray:
    """Run RTM grid at specified fractional cover."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cd, h = fc_to_cd_h(fc)
    print(f"  fc={fc:.2f} → cd={cd:.2f}m, h={h:.2f}m, sd={SD:.0f} trees/ha")
    
    H, W = len(lai_vals), len(laiu_vals)
    spectra = np.zeros((H, W, len(BANDS_INDEX)), dtype=np.float32)
    
    for i, lai in enumerate(lai_vals):
        batch = len(laiu_vals)
        paras = {}
        
        for k, v in defaults.items():
            paras[k] = torch.full((batch,), float(v[0].item()), 
                                 dtype=torch.float32, device=device)
        
        paras['LAI'] = torch.full((batch,), float(lai), dtype=torch.float32, device=device)
        paras['LAIu'] = torch.tensor(laiu_vals, dtype=torch.float32, device=device)
        paras['cd'] = torch.full((batch,), float(cd), dtype=torch.float32, device=device)
        paras['h'] = torch.full((batch,), float(h), dtype=torch.float32, device=device)
        paras['sd'] = torch.full((batch,), float(SD), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            out = rtm.run(**paras)
            out = out[:, BANDS_INDEX].detach().cpu().numpy()
        spectra[i, :, :] = out
    
    return spectra


def compute_numeric_gradient(values: np.ndarray, axis: int, grid: np.ndarray) -> np.ndarray:
    """Central-difference gradient along axis."""
    vals = np.moveaxis(values, axis, 0)
    coords = grid.astype(np.float64)
    grad = np.zeros_like(vals, dtype=np.float64)
    
    denom = (coords[2:] - coords[:-2])
    grad[1:-1] = (vals[2:] - vals[:-2]) / denom[:, None, None]
    grad[0] = (vals[1] - vals[0]) / (coords[1] - coords[0])
    grad[-1] = (vals[-1] - vals[-2]) / (coords[-1] - coords[-2])
    
    return np.moveaxis(grad, 0, axis)


def load_standardization(config_path):
    """Load mean and scale from config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    x_mean_path = os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_mean'])
    x_scale_path = os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale'])
    
    x_mean = np.load(x_mean_path)
    x_scale = np.load(x_scale_path)
    
    return x_mean, x_scale


def standardize_gradient(grad, x_scale):
    """Transform gradient: d(x_std)/d(param) = (1/scale) * dx/d(param)"""
    return grad / x_scale


def compute_gradient_stats(grad_data, threshold=None):
    """
    Compute mean and saturation rate for gradient data.
    Similar to ml_metrics_fc5.txt calculation.
    """
    abs_grad = np.abs(grad_data)
    
    if threshold is not None:
        # For standardized gradients: compute saturation and mean on non-saturated
        mask_saturated = abs_grad < threshold
        sat_rate = np.mean(mask_saturated)
        if sat_rate < 1.0:
            mean_grad = np.mean(abs_grad[~mask_saturated])
        else:
            mean_grad = 0.0
    else:
        # For raw gradients: just compute mean
        mean_grad = np.mean(abs_grad)
        sat_rate = 0.0
    
    return mean_grad, sat_rate


def plot_gradient_heatmap_reference_style(grad_data, lai_vals, laiu_vals, 
                                          band_name, param_name, fc_value, 
                                          is_standardized, vmin_global, vmax_global,
                                          output_dir):
    """
    Plot gradient heatmap matching reference style (heatmap_B03_GREEN.png).
    - Figure size ~6.5x5 (roughly square)
    - Colorbar roughly same size as plot
    - Actual min/max values from data
    - Mean gradient and saturation in title
    - Same scale across all fc and parameters for same band
    """
    label = BAND_LABELS[band_name]
    
    # Always use same axis orientation as reflectance plots: x=LAIu, y=LAI
    # For LAI gradients: data is already correct (axis 0 = LAI, axis 1 = LAIu)
    # For LAIu gradients: data is (n_lai, n_laiu), which matches reflectance orientation
    # So we don't need to transpose for LAIu - keep same orientation as reflectance
    if param_name == 'LAI':
        # ∂X/∂LAI: x-axis is LAIu (fixed), y-axis is LAI (gradient variable)
        xlabel = r'$Z_{\mathrm{LAIu}}$'
        ylabel = r'$Z_{\mathrm{LAI}}$'
        x_range = laiu_vals
        y_range = lai_vals
    else:  # LAIu
        # ∂X/∂LAIu: keep same orientation as reflectance (x=LAIu, y=LAI)
        # Data is already in correct orientation (n_lai, n_laiu)
        xlabel = r'$Z_{\mathrm{LAIu}}$'
        ylabel = r'$Z_{\mathrm{LAI}}$'
        x_range = laiu_vals
        y_range = lai_vals
    
    # Compute statistics
    threshold = STD_GRAD_THRESHOLD if is_standardized else None
    mean_grad, sat_rate = compute_gradient_stats(grad_data, threshold)
    
    # Use global min/max for consistent scale
    vmin = vmin_global
    vmax = vmax_global
    
    # Plot using seaborn heatmap for square appearance (like reference)
    plt.figure(figsize=(6.5, 5))
    ax = sns.heatmap(
        grad_data,
        cmap='coolwarm',
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        square=True,  # This makes it square!
        xticklabels=False,  # We'll set custom ticks
        yticklabels=False
    )
    
    # Set custom tick positions and labels
    # Always use same ticks as reflectance plots: x=LAIu (0.5 intervals), y=LAI (1.0 intervals)
    
    # X-axis ticks (always LAIu)
    x_min, x_max = x_range[0], x_range[-1]
    # X is always LAIu -> use 0.5 intervals up to 3.0
    x_nice_vals = np.arange(0.5, 3.0 + 0.01, 0.5)
    
    # Add edge positions (0.01 at start isn't shown, but max value position is at edge)
    x_tick_positions = [(val - x_min) / (x_max - x_min) * (len(x_range) - 1) for val in x_nice_vals]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([f'{val:.1f}' for val in x_nice_vals], fontsize=18)
    
    # Y-axis ticks (always LAI)
    y_min, y_max = y_range[0], y_range[-1]
    # Y is always LAI -> use 1.0 intervals up to 5.0
    y_nice_vals = np.arange(1.0, 5.0 + 0.01, 1.0)
    
    y_tick_positions = [(val - y_min) / (y_max - y_min) * (len(y_range) - 1) for val in y_nice_vals]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([f'{val:.1f}' for val in y_nice_vals], fontsize=18)  # <-- set fontsize
    
    # Invert y-axis so 0.01 is at bottom
    ax.invert_yaxis()
    
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    # add padding to the x and y labels
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5
    
    # Build title NOTE edit
    z_var = f'Z_{{\mathrm{{{param_name}}}}}'
    x_var = f'X_{{\mathrm{{{label}}}}}'

    z_var_fixed = f'Z_{{\mathrm{{LAIu}}}}' if param_name == 'LAI' else f'Z_{{\mathrm{{LAI}}}}'
    
    if is_standardized:
        # title = fr'$\partial {x_var} / \partial {z_var}$ ($Z_{{\mathrm{{fc}}}}$={fc_value:.1f})'
        title = fr'$(\partial {x_var} / \partial {z_var}) \mid {z_var_fixed}, Z_{{\mathrm{{fc}}}}={fc_value:.1f}$'
        # Add mean value to upper right corner
        ax.text(
            0.95, 0.05, 
            f'Mean={mean_grad:.3f}', 
            va='bottom', ha='right', 
            transform=ax.transAxes, 
            fontsize=18, fontweight='normal'
        )
    else:
        title = fr'$\partial {x_var} / \partial {z_var}$ ($Z_{{\mathrm{{fc}}}}$={fc_value:.1f})'
        # Add mean value to upper right corner
        ax.text(
            0.95, 0.05, 
            f'Mean={mean_grad:.6f}', 
            va='bottom', ha='right', 
            transform=ax.transAxes, 
            fontsize=18, fontweight='normal'
        )
    
    # ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=8)
    
    # ===== COLORBAR CONFIGURATION FOR GRADIENT PLOTS =====
    # Get the colorbar object from the heatmap
    cbar = ax.collections[0].colorbar
    
    # Set colorbar label based on gradient type (standardized or raw)
    # if is_standardized:
    #     cbar_label = fr'$Gradient (standardized)$'
    # else:
    #     cbar_label = fr'$Gradient (raw)$'
    # cbar.set_label(cbar_label, fontsize=18)
    
    # Set font size of colorbar tick labels
    cbar.ax.tick_params(labelsize=18)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    grad_type = 'std' if is_standardized else 'raw'
    filename = f'{grad_type}_{param_name}_{band_name}_fc{int(fc_value*10)}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=250, bbox_inches='tight')
    plt.close()


def plot_reflectance_heatmap(spectra, lai_vals, laiu_vals, band_idx, band_name, 
                             fc_value, output_dir, vmin_global=None, vmax_global=None):
    """Plot reflectance heatmap."""
    data = spectra[:, :, band_idx]
    
    plt.figure(figsize=(6.5, 5))
    ax = sns.heatmap(
        data,
        cmap='viridis',
        square=True,  # Square heatmap!
        xticklabels=False,
        yticklabels=False,
        vmin=vmin_global,
        vmax=vmax_global
    )
    
    # Set custom tick positions and labels for LAIu (x-axis)
    # LAIu: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    laiu_min, laiu_max = laiu_vals[0], laiu_vals[-1]
    laiu_nice_vals = np.arange(0.5, 3.0 + 0.01, 0.5)
    laiu_tick_positions = [(val - laiu_min) / (laiu_max - laiu_min) * (len(laiu_vals) - 1) for val in laiu_nice_vals]
    ax.set_xticks(laiu_tick_positions)
    ax.set_xticklabels([f'{val:.1f}' for val in laiu_nice_vals], fontsize=18)
    
    # Set custom tick positions and labels for LAI (y-axis)
    # LAI: 1.0, 2.0, 3.0, 4.0, 5.0
    lai_min, lai_max = lai_vals[0], lai_vals[-1]
    lai_nice_vals = np.arange(1.0, 5.0 + 0.01, 1.0)
    lai_tick_positions = [(val - lai_min) / (lai_max - lai_min) * (len(lai_vals) - 1) for val in lai_nice_vals]
    ax.set_yticks(lai_tick_positions)
    ax.set_yticklabels([f'{val:.1f}' for val in lai_nice_vals], fontsize=18)
    
    # Invert y-axis so 0.01 is at bottom
    ax.invert_yaxis()
    
    ax.set_xlabel(r'$Z_{\mathrm{LAIu}}$', fontsize=20)
    ax.set_ylabel(r'$Z_{\mathrm{LAI}}$', fontsize=20)

    # add padding to the x and y labels
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5
    
    label = BAND_LABELS[band_name]
    ax.set_title(fr'$X_{{\mathrm{{{label}}}}}$ ($Z_{{\mathrm{{fc}}}}$={fc_value:.1f})', 
                fontsize=18, fontweight='bold', pad=8)
    
    # ===== COLORBAR CONFIGURATION FOR REFLECTANCE PLOTS =====
    # Get the colorbar object from the heatmap
    cbar = ax.collections[0].colorbar
    
    # Set colorbar label to show reflectance as percentage
    # cbar.set_label('Reflectance (%)', fontsize=18)
    
    # Set font size of colorbar tick labels
    cbar.ax.tick_params(labelsize=18)
    
    # Format tick labels: multiply by 100 to convert from fraction (0-1) to percentage (0-100)
    # Use a formatter function to preserve colorbar size (same approach as gradient plots)
    # This ensures the colorbar maintains its original aspect ratio and matches gradient plots
    def percentage_formatter(x, p):
        """Format tick labels as percentages (multiply by 100)"""
        # return f'{x*100:.1f}'
        return f'{x*100:.0f}'
    
    # Apply the formatter to the colorbar axis (preserves colorbar size)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'reflectance_{band_name}_fc{int(fc_value*10)}.png'),
               dpi=200, bbox_inches='tight')
    plt.close()


def compute_global_scales_per_band(all_gradients_raw, all_gradients_std):
    """
    Compute global vmin/vmax for each band across all fc values and parameters.
    NOTE to be updated for NIR bands to update the max values
    
    Args:
        all_gradients_raw: dict {fc: {'LAI': array, 'LAIu': array}}
        all_gradients_std: dict {fc: {'LAI': array, 'LAIu': array}}
    
    Returns:
        scales_raw: dict {band_idx: (vmin, vmax)}
        scales_std: dict {band_idx: (vmin, vmax)}
    """
    n_bands = len(S2_BANDS_USED)
    scales_raw = {}
    scales_std = {}
    
    for b in range(n_bands):
        # Raw gradients - combine LAI and LAIu across all fc values
        all_vals_raw = []
        for fc_data in all_gradients_raw.values():
            all_vals_raw.append(fc_data['LAI'][:, :, b].flatten())
            all_vals_raw.append(fc_data['LAIu'][:, :, b].flatten())
        all_vals_raw = np.concatenate(all_vals_raw)
        
        vmax_raw = max(abs(np.nanmin(all_vals_raw)), abs(np.nanmax(all_vals_raw)))
        vmin_raw = -vmax_raw
        scales_raw[b] = (vmin_raw, vmax_raw)
        
        # Standardized gradients
        all_vals_std = []
        for fc_data in all_gradients_std.values():
            all_vals_std.append(fc_data['LAI'][:, :, b].flatten())
            all_vals_std.append(fc_data['LAIu'][:, :, b].flatten())
        all_vals_std = np.concatenate(all_vals_std)
        
        vmax_std = max(abs(np.nanmin(all_vals_std)), abs(np.nanmax(all_vals_std)))
        vmin_std = -vmax_std
        scales_std[b] = (vmin_std, vmax_std)
    
    return scales_raw, scales_std


def compute_global_reflectance_scales_per_band(all_spectra):
    """
    Compute global vmin/vmax for reflectance for each band across all fc values.
    
    Args:
        all_spectra: dict {fc: array of shape (n_lai, n_laiu, n_bands)}
    
    Returns:
        scales_refl: dict {band_idx: (vmin, vmax)}
    """
    n_bands = len(S2_BANDS_USED)
    scales_refl = {}
    
    for b in range(n_bands):
        # Combine reflectance values across all fc values
        all_vals = []
        for spectra in all_spectra.values():
            all_vals.append(spectra[:, :, b].flatten())
        all_vals = np.concatenate(all_vals)
        
        vmin_refl = np.nanmin(all_vals)
        vmax_refl = np.nanmax(all_vals)
        scales_refl[b] = (vmin_refl, vmax_refl)
    
    return scales_refl


def save_metrics_report(output_dir, grad_lai_raw, grad_laiu_raw, 
                       grad_lai_std, grad_laiu_std, fc_value):
    """Save ML metrics report similar to ml_metrics_fc5.txt."""
    metrics = {}
    
    for b, band in enumerate(S2_BANDS_USED):
        mean_lai_raw, _ = compute_gradient_stats(grad_lai_raw[:, :, b], threshold=None)
        mean_laiu_raw, _ = compute_gradient_stats(grad_laiu_raw[:, :, b], threshold=None)
        mean_lai_std, sat_lai = compute_gradient_stats(grad_lai_std[:, :, b], 
                                                        threshold=STD_GRAD_THRESHOLD)
        mean_laiu_std, sat_laiu = compute_gradient_stats(grad_laiu_std[:, :, b], 
                                                          threshold=STD_GRAD_THRESHOLD)
        
        metrics[band] = {
            'grad_lai_mean_abs_raw': mean_lai_raw,
            'grad_laiu_mean_abs_raw': mean_laiu_raw,
            'grad_lai_mean_abs_std': mean_lai_std,
            'grad_laiu_mean_abs_std': mean_laiu_std,
            'grad_lai_saturation': sat_lai,
            'grad_laiu_saturation': sat_laiu,
        }
    
    # Rankings by standardized gradients
    lai_ranking = sorted(metrics.items(), 
                        key=lambda x: x[1]['grad_lai_mean_abs_std'], 
                        reverse=True)
    laiu_ranking = sorted(metrics.items(), 
                         key=lambda x: x[1]['grad_laiu_mean_abs_std'], 
                         reverse=True)
    
    report_path = os.path.join(output_dir, f'ml_metrics_fc{int(fc_value*10)}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ML LEARNING DIFFICULTY ANALYSIS (fc={fc_value:.2f})\n")
        f.write("="*80 + "\n\n")
        
        f.write("NOTE: Rankings based on STANDARDIZED gradients (what ML sees)\n")
        f.write("-"*80 + "\n\n")
        
        f.write("RANKING: LAI Learning Difficulty (Easiest to Hardest)\n")
        f.write("-"*80 + "\n")
        for rank, (band, m) in enumerate(lai_ranking, 1):
            f.write(f"{rank:2d}. {band:12s}  |∂R_std/∂LAI| = {m['grad_lai_mean_abs_std']:.6f}  "
                   f"(raw: {m['grad_lai_mean_abs_raw']:.6f}, sat: {m['grad_lai_saturation']*100:.1f}%)\n")
        
        f.write("\n")
        f.write("RANKING: LAIu Learning Difficulty (Easiest to Hardest)\n")
        f.write("-"*80 + "\n")
        for rank, (band, m) in enumerate(laiu_ranking, 1):
            f.write(f"{rank:2d}. {band:12s}  |∂R_std/∂LAIu| = {m['grad_laiu_mean_abs_std']:.6f}  "
                   f"(raw: {m['grad_laiu_mean_abs_raw']:.6f}, sat: {m['grad_laiu_saturation']*100:.1f}%)\n")
    
    print(f"  ✓ Saved metrics report: {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_values', type=float, nargs='+', default=[0.4, 0.8],
                       help='Fractional coverage values to analyze')
    args = parser.parse_args()
    
    base_dir = '/maps/ys611/MAGIC'
    
    # Load parameter ranges
    rtm_paras_path = os.path.join(base_dir, 'configs/rtm_paras.json')
    with open(rtm_paras_path, 'r') as f:
        rtm_paras = json.load(f)
    
    lai_min, lai_max = rtm_paras['LAI']['min'], rtm_paras['LAI']['max']
    laiu_min, laiu_max = rtm_paras['LAIu']['min'], rtm_paras['LAIu']['max']
    
    # Grid
    n_lai = 61
    n_laiu = 61
    lai_vals = np.linspace(lai_min, lai_max, n_lai)
    laiu_vals = np.linspace(laiu_min, laiu_max, n_laiu)
    
    # Load standardization parameters
    config_path = os.path.join(base_dir, 'configs/phys/AE_RTM_C_wytham.json')
    x_mean, x_scale = load_standardization(config_path)
    print(f"\nLoaded standardization: x_mean shape={x_mean.shape}, x_scale shape={x_scale.shape}")
    
    # Initialize RTM
    rtm = RTM()
    defaults = get_default_paras(rtm)
    
    fc_values = args.fc_values
    print(f"\nAnalyzing fc values: {fc_values}")
    
    base_output_dir = os.path.join(base_dir, 'saved/rtm/sensitivity_final')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # PASS 1: Compute all gradients to determine global scales
    print(f"\n{'='*80}")
    print("PASS 1: Computing all gradients for global scale determination")
    print(f"{'='*80}")
    
    all_spectra = {}
    all_gradients_raw = {}
    all_gradients_std = {}
    
    for fc in fc_values:
        print(f"\nProcessing fc={fc:.2f}...")
        
        # Try to load saved data first
        fc_dir = os.path.join(base_output_dir, f'data_fc{int(fc*10)}')
        spectra_file = os.path.join(fc_dir, 'spectra.npy')
        lai_grad_file = os.path.join(fc_dir, 'dR_dLAI_raw.npy')
        laiu_grad_file = os.path.join(fc_dir, 'dR_dLAIu_raw.npy')
        lai_std_file = os.path.join(fc_dir, 'dR_dLAI_std.npy')
        laiu_std_file = os.path.join(fc_dir, 'dR_dLAIu_std.npy')
        
        if (os.path.exists(spectra_file) and os.path.exists(lai_grad_file) and 
            os.path.exists(laiu_grad_file) and os.path.exists(lai_std_file) and 
            os.path.exists(laiu_std_file)):
            print("  Loading saved data...")
            spectra = np.load(spectra_file)
            dR_dLAI = np.load(lai_grad_file)
            dR_dLAIu = np.load(laiu_grad_file)
            dR_std_dLAI = np.load(lai_std_file)
            dR_std_dLAIu = np.load(laiu_std_file)
        else:
            print("  Computing RTM spectra and gradients...")
            # Run RTM
            spectra = run_grid_at_fc(rtm, defaults, lai_vals, laiu_vals, fc)
            
            # Compute gradients
            dR_dLAI = compute_numeric_gradient(spectra, axis=0, grid=lai_vals)
            dR_dLAIu = compute_numeric_gradient(spectra, axis=1, grid=laiu_vals)
            
            dR_std_dLAI = standardize_gradient(dR_dLAI, x_scale)
            dR_std_dLAIu = standardize_gradient(dR_dLAIu, x_scale)
            
            # Save for future use
            os.makedirs(fc_dir, exist_ok=True)
            np.save(spectra_file, spectra)
            np.save(lai_grad_file, dR_dLAI)
            np.save(laiu_grad_file, dR_dLAIu)
            np.save(lai_std_file, dR_std_dLAI)
            np.save(laiu_std_file, dR_std_dLAIu)
            print(f"  Saved data to: {fc_dir}")
        
        all_spectra[fc] = spectra
        all_gradients_raw[fc] = {'LAI': dR_dLAI, 'LAIu': dR_dLAIu}
        all_gradients_std[fc] = {'LAI': dR_std_dLAI, 'LAIu': dR_std_dLAIu}
    
    # Compute global scales for each band
    print("\nComputing global scales across all bands and fc values...")
    scales_raw, scales_std = compute_global_scales_per_band(all_gradients_raw, all_gradients_std)
    scales_refl = compute_global_reflectance_scales_per_band(all_spectra)
    
    # PASS 2: Generate all plots with consistent scales
    print(f"\n{'='*80}")
    print("PASS 2: Generating all visualizations with consistent scales")
    print(f"{'='*80}")
    
    for fc in fc_values:
        print(f"\nGenerating plots for fc={fc:.2f}...")
        
        spectra = all_spectra[fc]
        dR_dLAI = all_gradients_raw[fc]['LAI']
        dR_dLAIu = all_gradients_raw[fc]['LAIu']
        dR_std_dLAI = all_gradients_std[fc]['LAI']
        dR_std_dLAIu = all_gradients_std[fc]['LAIu']
        
        # Output directories
        refl_dir = os.path.join(base_output_dir, f'reflectance_fc{int(fc*10)}')
        raw_dir = os.path.join(base_output_dir, f'gradients_raw_fc{int(fc*10)}')
        std_dir = os.path.join(base_output_dir, f'gradients_std_fc{int(fc*10)}')
        
        # Generate plots
        for b, band in enumerate(S2_BANDS_USED):
            # Get global scales for this band
            vmin_raw, vmax_raw = scales_raw[b]
            vmin_std, vmax_std = scales_std[b]
            vmin_refl, vmax_refl = scales_refl[b]
            
            # Reflectance
            plot_reflectance_heatmap(spectra, lai_vals, laiu_vals, b, band, fc, refl_dir,
                                   vmin_global=vmin_refl, vmax_global=vmax_refl)
            
            # Raw gradients
            plot_gradient_heatmap_reference_style(
                dR_dLAI[:, :, b], lai_vals, laiu_vals, band, 'LAI', fc,
                is_standardized=False, vmin_global=vmin_raw, vmax_global=vmax_raw,
                output_dir=raw_dir
            )
            
            plot_gradient_heatmap_reference_style(
                dR_dLAIu[:, :, b], lai_vals, laiu_vals, band, 'LAIu', fc,
                is_standardized=False, vmin_global=vmin_raw, vmax_global=vmax_raw,
                output_dir=raw_dir
            )
            
            # Standardized gradients
            plot_gradient_heatmap_reference_style(
                dR_std_dLAI[:, :, b], lai_vals, laiu_vals, band, 'LAI', fc,
                is_standardized=True, vmin_global=vmin_std, vmax_global=vmax_std,
                output_dir=std_dir
            )
            
            plot_gradient_heatmap_reference_style(
                dR_std_dLAIu[:, :, b], lai_vals, laiu_vals, band, 'LAIu', fc,
                is_standardized=True, vmin_global=vmin_std, vmax_global=vmax_std,
                output_dir=std_dir
            )
        
        # Save metrics report
        save_metrics_report(base_output_dir, dR_dLAI, dR_dLAIu, 
                          dR_std_dLAI, dR_std_dLAIu, fc)
        
        print(f"  ✓ Reflectance plots: {refl_dir}")
        print(f"  ✓ Raw gradients: {raw_dir}")
        print(f"  ✓ Standardized gradients: {std_dir}")
    
    print(f"\n{'='*80}")
    print("All visualizations complete!")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()


