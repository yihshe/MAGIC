# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import os
import json

# %%
"""
Configuration: Update these paths based on your model predictions
"""
# Path to model predictions (after running inference on full_region_for_mapping.csv)
# This should be the output CSV from your model with columns: date, sample_id, x, y, latent_N, latent_cab, etc.
BASE_PATH = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1016_194432/models'
# BASE_PATH = '/maps/ys611/MAGIC/saved_archived/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/0923_023534_kl0_edge1_rank4/models'
CSV_PREDICTIONS = os.path.join(BASE_PATH, 'model_best_testset_analyzer_full_region.csv')

# Path to save spatial maps
SAVE_PATH = os.path.join(BASE_PATH, 'spatial_maps')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Load in-situ site locations for marking on maps  
CSV_INSITU_COORDS = '/maps/ys611/MAGIC/data/raw/wytham/csv_preprocessed_data/frm4veg_plot_coordinates.csv'

# RTM parameters for colormaps
rtm_paras = json.load(open('/maps/ys611/MAGIC/configs/rtm_paras.json'))

# %%
"""
Load data
"""
print("Loading predictions...")
df_pred = pd.read_csv(CSV_PREDICTIONS)
print(f"Predictions shape: {df_pred.shape}")
print(f"Columns: {df_pred.columns.tolist()}")
print(f"Dates: {df_pred['date'].unique()}")

# Load in-situ site locations
print("\nLoading in-situ site locations...")
insitu_locations = pd.read_csv(CSV_INSITU_COORDS)
print(f"Number of in-situ sites: {len(insitu_locations)}")
print(f"Coordinate range: X=[{insitu_locations['x'].min():.0f}, {insitu_locations['x'].max():.0f}], Y=[{insitu_locations['y'].min():.0f}, {insitu_locations['y'].max():.0f}]")

# %%
"""
Create individual spatial maps for each variable and date
One figure per variable per date
"""
ATTRS = list(rtm_paras.keys())
ATTRS_LATEX = {
    'N': r'$Z_{\mathrm{N}}$', 
    'cab': r'$Z_{\mathrm{cab}}$', 
    'cw': r'$Z_{\mathrm{cw}}$',
    'cm': r'$Z_{\mathrm{cm}}$', 
    'LAI': r'$Z_{\mathrm{LAI}}$', 
    'LAIu': r'$Z_{\mathrm{LAIu}}$',
    'fc': r'$Z_{\mathrm{fc}}$'
}

# Date formatting for titles
DATE_FORMAT = {
    '2018.06.26': '26 June',
    '2018.06.29': '29 June',
    '2018.07.06': '6 July',
    '2018.07.11': '11 July',
}

dates = df_pred['date'].unique()
total_plots = len(dates) * len(ATTRS)
current_plot = 0

# for date in dates:
for date in ['2018.06.26', '2018.06.29', '2018.07.06', '2018.07.11']:
    print(f"\nProcessing date: {date}")
    df_date = df_pred[df_pred['date'] == date]
    
    # for attr in ATTRS:
    for attr in ['cab', 'fc']:
        current_plot += 1
        print(f"  [{current_plot}/{total_plots}] Creating map for {attr}...")
        
        # Check if latent variable exists
        latent_col = f'latent_{attr}'
        if latent_col not in df_date.columns:
            print(f"    Warning: {latent_col} not found in predictions")
            continue
        
        # Filter out any NaN values
        df_plot = df_date[['x', 'y', latent_col]].dropna()
        
        if len(df_plot) == 0:
            print(f"    Warning: No valid data for {attr}")
            continue
        
        # Create individual figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create spatial map using scatter plot
        # # Override vmax for cab to the max value of the for better visualization
        vmax_value = 50 if attr == 'cab' else rtm_paras[attr]['max']
        vmin_value = rtm_paras[attr]['min']
        
        scatter = ax.scatter(
            df_plot['x'],
            df_plot['y'],
            c=df_plot[latent_col],
            cmap='viridis',
            s=15,  # point size
            vmin=vmin_value,
            vmax=vmax_value,
            alpha=0.9,
            edgecolors='none'
        )
        
        # Mark in-situ site locations with red dots
        ax.scatter(
            insitu_locations['x'],
            insitu_locations['y'],
            c='red',
            marker='o',
            s=35,
            alpha=0.5,  # transparency
            edgecolors='darkred',
            linewidths=1,
            label='In-situ sites',
            zorder=10
        )
        
        # Add colorbar matching plot height (using axes_grid1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(scatter, cax=cax)
        # cbar.set_label(ATTRS_LATEX[attr], fontsize=20, rotation=270, labelpad=30)
        # Set cbar tick labels font size to 30
        cbar.ax.tick_params(labelsize=30)
        
        # Formatting
        ax.set_xlabel('Easting (km)', fontsize=30)
        ax.set_ylabel('Northing (km)', fontsize=30)
        
        # Format title: e.g., "$Z_{fc}$, 26 June"
        date_str = DATE_FORMAT.get(date, date)
        ax.set_title(f'{ATTRS_LATEX[attr]} ({date_str})', fontsize=35, fontweight='bold', pad=20)
        
        # Convert tick labels from meters to kilometers
        def format_km(x, pos):
            return f'{x/1000:.1f}'
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_km))
        ax.yaxis.set_major_formatter(FuncFormatter(format_km))
        ax.tick_params(axis='both', which='major', labelsize=30)
        # Rotate y-axis tick labels to horizontal
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
        
        # Set aspect ratio to equal for proper geographic representation
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid for geo-location reference
        ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.8, color='gray')
        
        # Add minor grid for finer reference
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5, color='gray')
        
        plt.tight_layout()
        
        # Save figure with descriptive filename
        date_filename = date.replace('.', '')
        save_filename = os.path.join(SAVE_PATH, f'spatial_map_{attr}_{date_filename}.png')
        # plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"    Saved: {save_filename}")
        plt.close()

print("\n" + "="*80)
print(f"✅ All {current_plot} spatial maps generated successfully!")
print(f"Saved to: {SAVE_PATH}")
print("="*80)



# %%

