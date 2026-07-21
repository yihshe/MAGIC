# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import json

# %%
# BASE_PATH = '/maps/ys611/MAGIC/saved/'
# BASE_PATH = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1017_033515/models'
BASE_PATH = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1016_194432/models'
# BASE_PATH = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1017_033515/models'

CSV_PATH_INSITU_SITES = os.path.join(
    BASE_PATH, 'model_best_testset_analyzer_frm4veg.csv')

CSV_PATH_WHOLE_REGION = os.path.join(
    BASE_PATH, 'model_best_testset_analyzer.csv')

CSV_PATH_INSITU = '/maps/ys611/MAGIC/data/raw/wytham/csv_preprocessed_data/frm4veg_insitu.csv'

SAVE_PATH = os.path.join(BASE_PATH, 'plots_frm4veg')


S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
S2_names = {
    'B02_BLUE': 'Blue',
    'B03_GREEN': 'Green',
    'B04_RED': 'Red',
    'B05_RE1': 'RE1',
    'B06_RE2': 'RE2',
    'B07_RE3': 'RE3',
    'B08_NIR1': 'NIR1',
    'B8A_NIR2': 'NIR2',
    'B09_WV': 'WV',
    'B11_SWI1': 'SWI1',
    'B12_SWI2': 'SWI2'
}
rtm_paras = json.load(open('/maps/ys611/MAGIC/configs/rtm_paras.json'))# Range of LAIu has been changed from [0.01, 1] to [0.01, 5]
# rtm_paras = json.load(open('/maps/ys611/MAGIC/configs/rtm_paras_exp.json'))# Range of LAIu has been changed from [0.01, 1] to [0.01, 5]

ATTRS = list(rtm_paras.keys())
# for each attr in ATTRS, create a LaTex variable name like $Z_{\mathrm{attr}}$
ATTRS_LATEX = {
    'N': '$Z_{\mathrm{N}}$', 'cab': '$Z_{\mathrm{cab}}$', 'cw': '$Z_{\mathrm{cw}}$',
    'cm': '$Z_{\mathrm{cm}}$', 'LAI': '$Z_{\mathrm{LAI}}$', 'LAIu': '$Z_{\mathrm{LAIu}}$',
    'fc': '$Z_{\mathrm{fc}}$'
    }

ATTRS_INSITU = {
    'cab': 'LCC',
    'fc': 'FCOVER_up',
    'LAI': 'LAI_up',
    'LAIu': 'LAI_down',
}
ATTRS_VANILLA = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

NUM_BINS = 100
# mkdir if the save path does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# read the csv file
# df0 = pd.read_csv(CSV_PATH0)
# df1 = pd.read_csv(CSV_PATH1)
df_insitu_sites = pd.read_csv(CSV_PATH_INSITU_SITES)
df_whole_region = pd.read_csv(CSV_PATH_WHOLE_REGION)
df_insitu = pd.read_csv(CSV_PATH_INSITU)

# retrieve the target and output bands to original scale
# MEAN = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/train_x_mean.npy')
# SCALE = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/train_x_scale.npy')
MEAN = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/insitu_period/train_x_mean.npy')
SCALE = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/insitu_period/train_x_scale.npy')

# Scale both datasets
for df_data in [df_insitu_sites, df_whole_region]:
    for x in ['target', 'output', 'init_output']:
        df_data[[f'{x}_{band}' for band in S2_BANDS]] = df_data[[f'{x}_{band}' for band in S2_BANDS]]*SCALE +MEAN
    df_data[[f'bias_{band}' for band in S2_BANDS]] = df_data[[f'bias_{band}' for band in S2_BANDS]]*SCALE
    
# dates = ['2018.04.20', '2018.05.05', '2018.05.07', '2018.05.15', '2018.05.17', 
#          '2018.06.06', '2018.06.11', '2018.06.26', '2018.06.29', '2018.07.06', 
#          '2018.07.11', '2018.07.24', '2018.08.05', '2018.09.02', '2018.09.27', 
#          '2018.10.09', '2018.10.19', '2018.10.22']
dates = ['2018.06.26', '2018.06.29', '2018.07.06', '2018.07.11']

# TODO metrics to fairly compare the retrieved biophysical variables with the in-situ measurements
def r_square(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def rrmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2)) / np.mean(y)

def nmae(y, y_hat, var_name, rtm_paras):
    """
    Normalized MAE: Scale both variables to [0,1] using physical ranges, then calculate MAE
    
    Args:
        y: true values
        y_hat: predicted values
        var_name: variable name (e.g., 'LAI', 'cab', 'fc', 'LAIu')
        rtm_paras: dictionary with min/max ranges
    """
    y_min = rtm_paras[var_name]['min']
    y_max = rtm_paras[var_name]['max']
    
    # Normalize to [0, 1]
    y_norm = (y - y_min) / (y_max - y_min)
    y_hat_norm = (y_hat - y_min) / (y_max - y_min)
    
    # Calculate MAE on normalized values
    return np.mean(np.abs(y_norm - y_hat_norm))

def nrmse(y, y_hat, var_name, rtm_paras):
    y_min = rtm_paras[var_name]['min']
    y_max = rtm_paras[var_name]['max']
    
    # Normalize to [0, 1]
    y_norm = (y - y_min) / (y_max - y_min)
    y_hat_norm = (y_hat - y_min) / (y_max - y_min)
    
    # Calculate RMSE on normalized values
    return np.sqrt(np.mean((y_norm - y_hat_norm) ** 2))


def pred2insitu(df_insitu: pd.DataFrame, df_pred: pd.DataFrame, attrs: dict) -> pd.DataFrame:
    """
    Convert predicted RTM values to the same units as in-situ measurements and merge into a single DataFrame.
    
    Args:
        df_insitu (pd.DataFrame): In-situ measurement data with 'plot' column.
        df_pred (pd.DataFrame): Predicted RTM values with 'plot' column.
        attrs (dict): Mapping from predicted attribute name to in-situ attribute name.
                      e.g., {'LAIu': 'LAI_up', 'fc': 'FCOVER_up'}
    
    Returns:
        pd.DataFrame: Merged DataFrame with predicted and in-situ values side-by-side.
    """
    # Ensure matching and aligned 'plot'
    df_insitu = df_insitu.sort_values(by='sample_id').reset_index(drop=True)
    df_pred = df_pred.sort_values(by='sample_id').reset_index(drop=True)
    assert df_insitu['sample_id'].equals(df_pred['sample_id']), "The plots in df_insitu and df_pred are not the same"

    # Start with plot and sample_id
    df_merged = df_insitu[['plot']].copy()
    df_merged['sample_id'] = df_insitu['sample_id']

    # For each attribute, compute and add both predicted and in-situ versions
    for pred_attr, insitu_attr in attrs.items():
        if f'latent_{pred_attr}' not in df_pred.columns or insitu_attr not in df_insitu.columns:
            print(f"Skipping {pred_attr} → {insitu_attr}: missing in one of the dataframes")
            continue
        
        pred_values = df_pred[f'latent_{pred_attr}'].values
        insitu_values = df_insitu[insitu_attr].values

        # Apply conversion if needed
        if insitu_attr == ATTRS_INSITU['LAI']:
            # insitu_values = insitu_values / (df_insitu[ATTRS_INSITU['fc']]+1e-6)
            pass
        if insitu_attr == ATTRS_INSITU['cab']:
            # Convert cab from g/m² to μg/cm² 
            insitu_values = insitu_values * 100
        
        df_merged[f'{insitu_attr}_pred'] = pred_values
        df_merged[f'{insitu_attr}_insitu'] = insitu_values

    return df_merged


#%%
"""
Scatter plot for predicted variables and insitu measurements
"""
# for date in ['2018.06.06', '2018.06.11', '2018.06.26', '2018.06.29', '2018.07.06', '2018.07.11', '2018.07.24']:
for date in dates:
# date = '2018.07.11'# '2018.06.29' or '2018.07.06'
    df_pred = df_insitu_sites[df_insitu_sites['date'] == date]
    df_merged = pred2insitu(df_insitu, df_pred, ATTRS_INSITU)
    # Scatter plot for the predicted variables and insitu measurements
    # fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))
    for i, (attr, insitu_attr) in enumerate(ATTRS_INSITU.items()):
        # ax = axes[i // 2, i % 2]
        ax = axes[i]
        # Filter out rows where either predicted or insitu values are NaN
        df_merged_filtered = df_merged[['plot', f'{insitu_attr}_pred', f'{insitu_attr}_insitu']].dropna()
        # Check if the filtered DataFrame is empty
        if df_merged_filtered.empty:
            print(f"No data available for {attr} on {date}")
            continue
        sns.scatterplot(
            x=df_merged_filtered[f'{insitu_attr}_insitu'],
            y=df_merged_filtered[f'{insitu_attr}_pred'],
            ax=ax,
            s=40,
            color='blue',
            alpha=0.6,
            # set the point size
            linewidth=0.5,
            marker='o',
        )

        # r2_val = r_square(df_merged_filtered[f'{insitu_attr}_insitu'], df_merged_filtered[f'{insitu_attr}_pred'])
        nmae_val = nmae(df_merged_filtered[f'{insitu_attr}_insitu'], df_merged_filtered[f'{insitu_attr}_pred'], attr, rtm_paras)

        fontsize = 35

        # ax.set_title(f'{ATTRS_LATEX[attr]} vs {insitu_attr}')
        xlabel = f'{ATTRS_LATEX[attr]} (in-situ)'
        ylabel = f'{ATTRS_LATEX[attr]} (inferred)'
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=25)
        # plot the diagonal line

        # For fc, set the max limit as 1
        if attr == 'fc':
            min_limit = np.min([ax.get_xlim(), ax.get_ylim()])
            max_limit = 1.05

        elif attr in ['LAI', 'LAIu']:
            min_limit = 0.01
            max_limit = np.max([ax.get_xlim(), ax.get_ylim()])
        elif attr == 'cab':
            min_limit = 10.0
            max_limit = np.max([ax.get_xlim(), ax.get_ylim()])
        else:
            min_limit = np.min([ax.get_xlim(), ax.get_ylim()])
            max_limit = np.max([ax.get_xlim(), ax.get_ylim()])

        limits = [min_limit, max_limit]
        ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        # set the distance between y label and y axis
        ax.yaxis.labelpad = 10
        ax.set_aspect('equal')
        # make sure both axes have same ticks to display
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        
        # Custom tick formatting based on variable type
        if attr in ['LAI', 'LAIu']:
            ticks = [1.0, 2.0, 3.0, 4.0]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
        elif attr == 'cab':
            ticks = [20, 30, 40, 50, 60]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        # Integer formatting for cab, LAI, LAIu
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
        elif attr == 'fc':
            # For fc, tick labels as 0.2, 0.4, 0.8, 1.0
            ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
        else:
            # Two decimal places for other variables
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
        # set R-squared as a legend
        # ax.legend([f'$R^2$: {r2_val:.3f}'], fontsize=24)        
        # Add metrics as text box (upper left corner)
        metrics_text = f'MAE: {nmae_val:.3f}'
        ax.text(0.4, 0.125, metrics_text, 
                transform=ax.transAxes,
                fontsize=30,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='white', linewidth=1))
    # set the title for the whole figure
    fig.suptitle(f'{date}', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'linescatter_corr_insitu_v_pred_{date}.png'))
    plt.show()

# %%
"""
Scatter plot of target vs output and target vs init_output bands
Similar to the analysis in analysis_rtm.py
"""
datasets = {'insitu_sites': df_insitu_sites, 'whole_region': df_whole_region}
color = 'blue'

for dataset_name, df in datasets.items():
    for token in ['output', 'init_output']:
        # fig, axs = plt.subplots(3, 4, figsize=(24, 16))
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        ylabel = '$X_{\mathrm{S2, C}}$' if token == 'output' else '$X_{\mathrm{S2, F}}$'
        # for i, band in enumerate(S2_BANDS):
        # Only visualize Blue, Red, NIR1 and SWI1 bands
        bands = ['B02_BLUE', 'B04_RED', 'B08_NIR1', 'B11_SWI1']
        for i, band in enumerate(bands):
            # ax = axs[i//4, i % 4]
            ax = axs[i]
            sns.scatterplot(x='target_'+band, y=f'{token}_'+band, data=df, ax=ax,
                            s=8, alpha=0.5, color=color)
            # calculate R-squared
            r2_val = r_square(df[f'target_{band}'], df[f'{token}_{band}'])
            fontsize = 35
            # add the title
            ax.set_title(S2_names[band], fontsize=fontsize)
            xlabel = '$X_{\mathrm{S2}}$'
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            # set the same ticks for both x and y axes
            ax.tick_params(axis='both', which='major', labelsize=25)
            # plot the diagonal line
            limits = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            # set the distance between y label and y axis
            ax.yaxis.labelpad = 10
            ax.set_aspect('equal')
            # make sure both axes have same ticks to display
            ax.locator_params(axis='x', nbins=4)
            ax.locator_params(axis='y', nbins=4)
            # make sure all ticks are rounded to 2 decimal places
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
            # set R-squared as a legend
            ax.legend([f'$R^2$: {r2_val:.3f}'], fontsize=25)
        # make the last subplot empty
        # axs[-1, -1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, f'linescatter_{dataset_name}_bands_target_v_{token}_blue_red_nir1_swi1.png'))
        plt.show()

# %%
"""
Time series of latent biophysical variables with mean and std as error bars
For insitu sites: also overlay in-situ measurements on the plots
Similar to the analysis in analysis_rtm.py
"""
# Map dates to formatted labels like "08 Apr", "21 Aug", etc.
months = {'04': 'Apr', '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct'}
dates_plot = [f"{date.split('.')[2]} {months[date.split('.')[1]]}" for date in dates]

# Define datasets and their configurations
datasets_config = {
    'insitu_sites': {
        'df': df_insitu_sites,
        'show_insitu': True,
        'filename': 'timeseries_insitu_sites_vars_with_measurements.png'
    },
    'whole_region': {
        'df': df_whole_region,
        'show_insitu': True,
        'filename': 'timeseries_whole_region_vars.png'
    }
}

# Find the index for July 3rd (between June 29 and July 6)
# Dynamically find indices of June 29 and July 6 in the dates list
june_29_date = '2018.06.29'
july_6_date = '2018.07.06'

try:
    june_29_idx = dates.index(june_29_date)
    july_6_idx = dates.index(july_6_date)
    # July 3rd is halfway between June 29 and July 6
    insitu_date = (june_29_idx + july_6_idx) / 2.0
except ValueError:
    # If dates are not in the list, set to None and skip in-situ plotting
    insitu_date = None
    print(f"Warning: In-situ measurement dates ({june_29_date} and {july_6_date}) not found in dates list.")

# Initialize dictionary to store statistics
stats_dict = {}

for dataset_name, config in datasets_config.items():
    # Always plot all 7 variables
    # fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    # fig, axs = plt.subplots(4, 2, figsize=(20, 16))
    # fig, axs = plt.subplots(1, 4, figsize=(24, 4))
    fig, axs = plt.subplots(2, 2, figsize=(20, 7.5))
    df = config['df']
    
    # Initialize statistics for this dataset
    if dataset_name not in stats_dict:
        stats_dict[dataset_name] = {}
    
    # for i, attr in enumerate(ATTRS):
    for i, attr in enumerate(['cab', 'fc', 'LAI', 'LAIu']):
        ax = axs[i//2, i % 2]
        # ax = axs[i]
        
        # Get the time series of mean and std for predicted values
        mean_pred = []
        std_pred = []
        
        # Initialize statistics for this variable
        if attr not in stats_dict[dataset_name]:
            stats_dict[dataset_name][attr] = {
                'predictions': {},
                'insitu': None,
                'n_insitu': 0
            }
        
        # Use latent values directly (no conversion needed)
        for date in dates:
            df_filtered = df[df['date']==date]
            pred_values = df_filtered[f'latent_{attr}'].dropna()
            
            # Compute statistics for predictions
            mean_val = pred_values.mean()
            std_val = pred_values.std()
            mean_pred.append(mean_val)
            std_pred.append(std_val)
            
            # Store statistics for this date
            stats_dict[dataset_name][attr]['predictions'][date] = {
                'min': pred_values.min(),
                'max': pred_values.max(),
                'mean': mean_val,
                'median': pred_values.median(),
                'std': std_val,
                'n': len(pred_values)
            }
        
        # Plot the time series with numerical x-positions for better control
        x_positions = list(range(len(dates_plot)))
        ax.errorbar(x=x_positions, y=mean_pred, yerr=std_pred, fmt='o', color='blue', 
                    label='Predicted' if (config['show_insitu'] and attr in ATTRS_INSITU) else None,
                    markersize=8, linewidth=2)
        
        # Add in-situ measurements if applicable (only for insitu_sites dataset and where measurements exist)
        if config['show_insitu'] and attr in ATTRS_INSITU and insitu_date is not None:
            insitu_attr = ATTRS_INSITU[attr]
            if insitu_attr in df_insitu.columns:
                # Get in-situ values directly from df_insitu
                insitu_values = df_insitu[insitu_attr].dropna()
                
                # Apply conversion for cab: g/m² to μg/cm²
                if attr == 'cab':
                    insitu_values = insitu_values * 100
                
                if len(insitu_values) > 0:
                    mean_insitu = insitu_values.mean()
                    std_insitu = insitu_values.std()
                    
                    # Plot at July 3rd position (between June 29 and July 6)
                    ax.errorbar(x=[insitu_date], y=[mean_insitu], yerr=[std_insitu], 
                               fmt='s', color='red', markersize=12, label='In-situ', 
                               zorder=5, markeredgecolor='black', linewidth=1.5)
                    
                    # Store in-situ statistics
                    stats_dict[dataset_name][attr]['insitu'] = {
                        'min': insitu_values.min(),
                        'max': insitu_values.max(),
                        'mean': mean_insitu,
                        'median': insitu_values.median(),
                        'std': std_insitu
                    }
                    stats_dict[dataset_name][attr]['n_insitu'] = len(insitu_values)
        
        fontsize = 40
        ax.set_ylabel(ATTRS_LATEX[attr], fontsize=fontsize)
        
        # Always set y-axis limits to physical range
        # ax.set_ylim(rtm_paras[attr]['min'], rtm_paras[attr]['max'])
        # custom y-axis limits
        if attr == 'cab':
            ax.set_ylim(10, 80)
        elif attr == 'fc':
            ax.set_ylim(0.1, 1.1)
        elif attr == 'LAI':
            ax.set_ylim(0.01, 5.0)
        elif attr == 'LAIu':
            ax.set_ylim(0.01, 3.1)
        
        # Set appropriate y-axis ticks for each variable
        if attr == 'N':
            ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
        elif attr == 'cab':
            ax.set_yticks([20, 40, 60, 80])
        elif attr == 'cw':
            ax.set_yticks([0.005, 0.010, 0.015, 0.020])
        elif attr == 'cm':
            ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.05])
        elif attr == 'LAI':
            ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])
        elif attr == 'LAIu':
            ax.set_yticks([1.0, 2.0, 3.0])
        elif attr == 'fc':
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            # ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        
        # if config['show_insitu'] and attr in ATTRS_INSITU:
        #     ax.legend(fontsize=25, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        # Set x-tick positions and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(dates_plot, rotation=-30)
    
    # Turn off the last subplot
    # axs[-1, -1].axis('off')
    
    plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_PATH, config['filename']), dpi=300)
    plt.show()

# Compute normalized MAE and RMSE for all dates (compare with in-situ measurements made on July 3-5)
for dataset_name in stats_dict:
    df_dataset = datasets_config[dataset_name]['df']
    for attr in stats_dict[dataset_name]:
        if stats_dict[dataset_name][attr]['insitu'] and attr in ATTRS_INSITU:
            insitu_attr = ATTRS_INSITU[attr]
            if insitu_attr in df_insitu.columns:
                for date in dates:
                    df_filtered = df_dataset[df_dataset['date']==date]
                    if 'sample_id' in df_filtered.columns:
                        df_merged = pd.merge(
                            df_filtered[['sample_id', f'latent_{attr}']],
                            df_insitu[['sample_id', insitu_attr]],
                            on='sample_id', how='inner'
                        )
                        if len(df_merged) > 0:
                            # Drop rows where either value is NaN to keep them aligned
                            df_merged_clean = df_merged[[f'latent_{attr}', insitu_attr]].dropna()
                            if len(df_merged_clean) > 0:
                                pred_clean = df_merged_clean[f'latent_{attr}'].values
                                insitu_clean = df_merged_clean[insitu_attr].values
                                # Apply conversion for cab
                                if attr == 'cab':
                                    insitu_clean = insitu_clean * 100
                                # Compute normalized metrics
                                stats_dict[dataset_name][attr]['predictions'][date]['nmae'] = nmae(insitu_clean, pred_clean, attr, rtm_paras)
                                stats_dict[dataset_name][attr]['predictions'][date]['nrmse'] = nrmse(insitu_clean, pred_clean, attr, rtm_paras)

# Save statistics to CSV
rows = []
for dataset_name in stats_dict:
    for attr in stats_dict[dataset_name]:
        # In-situ statistics
        if stats_dict[dataset_name][attr]['insitu']:
            ins = stats_dict[dataset_name][attr]['insitu']
            rows.append({
                'dataset': dataset_name, 'variable': attr, 'date': 'in_situ_measurement', 'type': 'in_situ',
                'n': stats_dict[dataset_name][attr]['n_insitu'], 'min': ins['min'], 'max': ins['max'],
                'mean': ins['mean'], 'median': ins['median'], 'std': ins['std'], 'nmae': np.nan, 'nrmse': np.nan
            })
        # Prediction statistics
        for date, stats in stats_dict[dataset_name][attr]['predictions'].items():
            rows.append({
                'dataset': dataset_name, 'variable': attr, 'date': date, 'type': 'prediction',
                'n': stats['n'], 'min': stats['min'], 'max': stats['max'], 'mean': stats['mean'],
                'median': stats['median'], 'std': stats['std'], 'nmae': stats.get('nmae', np.nan),
                'nrmse': stats.get('nrmse', np.nan)
            })

# if rows:
#     pd.DataFrame(rows).to_csv(os.path.join(SAVE_PATH, 'timeseries_statistics.csv'), index=False)

print(f"Statistics saved to {os.path.join(SAVE_PATH, 'timeseries_statistics.csv')}")

# %%
"""
Kernel Density Estimation (KDE) plots for normalized residuals (y_hat_norm - y_norm)
for each variable and each date, comparing both models
"""
# Define both model paths
BASE_PATH_SMPL = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1016_194432/models'
BASE_PATH_PVAE = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1017_033515/models'

# Load data from both models
CSV_PATH_INSITU_SITES_SMPL = os.path.join(BASE_PATH_SMPL, 'model_best_testset_analyzer_frm4veg.csv')
CSV_PATH_INSITU_SITES_PVAE = os.path.join(BASE_PATH_PVAE, 'model_best_testset_analyzer_frm4veg.csv')

df_insitu_sites_smpl = pd.read_csv(CSV_PATH_INSITU_SITES_SMPL)
df_insitu_sites_pvae = pd.read_csv(CSV_PATH_INSITU_SITES_PVAE)

# Scale both datasets (same scaling as before)
for df_data in [df_insitu_sites_smpl, df_insitu_sites_pvae]:
    for x in ['target', 'output', 'init_output']:
        df_data[[f'{x}_{band}' for band in S2_BANDS]] = df_data[[f'{x}_{band}' for band in S2_BANDS]]*SCALE +MEAN
    df_data[[f'bias_{band}' for band in S2_BANDS]] = df_data[[f'bias_{band}' for band in S2_BANDS]]*SCALE

# Helper function to calculate residuals for a given dataset
def calculate_residuals(df_dataset, attr, date, insitu_attr, df_insitu, rtm_paras):
    """Calculate normalized residuals for a given model dataset"""
    df_filtered = df_dataset[df_dataset['date'] == date]
    if 'sample_id' not in df_filtered.columns or len(df_filtered) == 0:
        return None
    
    # Merge predictions with in-situ measurements
    df_merged = pd.merge(
        df_filtered[['sample_id', f'latent_{attr}']],
        df_insitu[['sample_id', insitu_attr]],
        on='sample_id', how='inner'
    )
    
    if len(df_merged) == 0:
        return None
    
    # Drop rows where either value is NaN
    df_merged_clean = df_merged[[f'latent_{attr}', insitu_attr]].dropna()
    
    if len(df_merged_clean) == 0:
        return None
    
    pred_clean = df_merged_clean[f'latent_{attr}'].values
    insitu_clean = df_merged_clean[insitu_attr].values
    
    # Apply conversion for cab
    if attr == 'cab':
        insitu_clean = insitu_clean * 100
    
    # Get normalization parameters
    y_min = rtm_paras[attr]['min']
    y_max = rtm_paras[attr]['max']
    
    # Normalize to [0, 1]
    y_norm = (insitu_clean - y_min) / (y_max - y_min)
    y_hat_norm = (pred_clean - y_min) / (y_max - y_min)
    # y_norm = insitu_clean
    # y_hat_norm = pred_clean
    
    # Calculate residuals: y_hat_norm - y_norm (prediction - measurement)
    residuals = y_hat_norm - y_norm
    
    return residuals

# Create a separate figure for each variable
for attr in ['cab', 'fc', 'LAI', 'LAIu']:
# for attr in ['LAI']:
    insitu_attr = ATTRS_INSITU[attr]
    if insitu_attr not in df_insitu.columns:
        continue
    
    # First pass: collect all residuals across all dates to determine common x-axis range
    all_residuals_all_dates = []
    for date in dates:
        residuals_smpl = calculate_residuals(df_insitu_sites_smpl, attr, date, insitu_attr, df_insitu, rtm_paras)
        residuals_pvae = calculate_residuals(df_insitu_sites_pvae, attr, date, insitu_attr, df_insitu, rtm_paras)
        
        if residuals_smpl is not None:
            all_residuals_all_dates.extend(residuals_smpl)
        if residuals_pvae is not None:
            all_residuals_all_dates.extend(residuals_pvae)
    
    # Calculate common x-axis range for all dates
    if len(all_residuals_all_dates) == 0:
        continue
    
    common_x_range = np.max(np.abs(all_residuals_all_dates)) * 1.2
    
    # Create figure with 1 row, 4 columns (one for each date)
    fig, axs = plt.subplots(1, 4, figsize=(26, 5))
    
    # Loop through each date
    for j, date in enumerate(dates):
        ax = axs[j]
        
        # Calculate residuals for both models
        residuals_smpl = calculate_residuals(df_insitu_sites_smpl, attr, date, insitu_attr, df_insitu, rtm_paras)
        residuals_pvae = calculate_residuals(df_insitu_sites_pvae, attr, date, insitu_attr, df_insitu, rtm_paras)
        
        if residuals_smpl is None and residuals_pvae is None:
            ax.axis('off')
            continue
        
        if (residuals_smpl is None or len(residuals_smpl) == 0) and (residuals_pvae is None or len(residuals_pvae) == 0):
            ax.axis('off')
            continue
        
        # Use the common x-axis range for all subplots
        x_range = common_x_range
        
        # Calculate MAE for both models
        mae_smpl = None
        mae_pvae = None
        if residuals_smpl is not None and len(residuals_smpl) > 0:
            mae_smpl = np.mean(np.abs(residuals_smpl))
        if residuals_pvae is not None and len(residuals_pvae) > 0:
            mae_pvae = np.mean(np.abs(residuals_pvae))
        
        # Plot KDE for PVAE model (red, "PVAE") - plot first so "ours" is on top
        if residuals_pvae is not None and len(residuals_pvae) > 1:
            sns.kdeplot(data=residuals_pvae, ax=ax, fill=True, alpha=0.5, color='red', 
                       linewidth=2, zorder=1)
        
        # Plot KDE for SMPL model (blue, "ours") - plot last with higher z-order to bring to front
        if residuals_smpl is not None and len(residuals_smpl) > 1:
            sns.kdeplot(data=residuals_smpl, ax=ax, fill=True, alpha=0.5, color='blue', 
                       linewidth=2, zorder=2)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2.0, alpha=1.0, zorder=0)
        
        # Set x-axis limits to center around 0
        ax.set_xlim(-x_range, x_range)
        
        # Formatting
        fontsize = 30
        # if j == 0:  # Leftmost column
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.set_xlabel('Prediction Error', fontsize=fontsize)
        # ax.set_title(date, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # Format y-axis labels to 1 decimal place
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
        # ax.grid(True, alpha=0.5)
        ax.grid(False)
        ax.yaxis.labelpad = 10
        
        # Add MAE as colored text annotations
        # Position text in upper left corner except for LAI
        if attr == 'LAI':
            text_y_pos = 0.95
            text_x_pos = 0.54
        else:
            text_y_pos = 0.95
            text_x_pos = 0.02
        
        # First row: "MAE" in black
        # ax.text(text_x_pos, text_y_pos, 'MAE', 
        #        transform=ax.transAxes, fontsize=12, color='black', 
        #        verticalalignment='top', weight='bold')
        # text_y_pos -= 0.08
        
        # Second row: "PVAE: X.XXX" in red
        if residuals_pvae is not None and len(residuals_pvae) > 1 and mae_pvae is not None:
            ax.text(text_x_pos, text_y_pos, f'HVAE: {mae_pvae:.3f}', 
                   transform=ax.transAxes, fontsize=23, color='red', 
                   verticalalignment='top', weight='bold')
            text_y_pos -= 0.11
        
        # Third row: "Ours: X.XXX" in blue
        if residuals_smpl is not None and len(residuals_smpl) > 1 and mae_smpl is not None:
            ax.text(text_x_pos, text_y_pos, f'PILA: {mae_smpl:.3f}', 
                   transform=ax.transAxes, fontsize=23, color='blue', 
                   verticalalignment='top', weight='bold')
    
    # Add overall title for the variable
    # fig.suptitle(f'KDE_Normalized_Residuals_{ATTRS_LATEX[attr]}', fontsize=22, y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'kde_absoluate_error_normalized_{attr}.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
"""
Histogram of predicted variables for whole region test set
"""
NUM_BINS = 100
ATTRS = list(rtm_paras.keys())

# Create one figure with 2x4 subplots for all 7 variables
fig, axs = plt.subplots(2, 4, figsize=(26, 10))

for i, attr in enumerate(ATTRS):
    ax = axs[i//4, i % 4]
    
    # Plot histogram for predicted values
    sns.histplot(
        df_whole_region[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        label='Predicted',
        alpha=0.5,
    )
    
    # Change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=25)
    # Set the range of x axis as the physical range of the variable
    ax.set_xlim(rtm_paras[attr]['min'], rtm_paras[attr]['max'])
    fontsize = 30
    ax.set_xlabel(ATTRS_LATEX[attr], fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    # Set the distance between the y label and the y axis
    ax.yaxis.labelpad = 10
    ax.legend(fontsize=fontsize-5)

# Remove the last subplot
axs[-1, -1].axis('off')
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, 'histogram_whole_region_vars.png'), dpi=300)
plt.show()


# %%
# TODO how is the spectra recostructed for these regions?
