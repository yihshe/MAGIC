# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import json

# %%
BASE_PATH = '/maps/ys611/MAGIC/saved/'
CSV_PATH0 = os.path.join(
    BASE_PATH, 'rtm/models/PHYS_VAE_RTM_A_WYTHAM/0323_204514/model_best_testset_analyzer.csv')
CSV_PATH1 = os.path.join(
    BASE_PATH, 'rtm/models/PHYS_VAE_RTM_B_WYTHAM/0329_080231/model_best_testset_analyzer_frm4veg.csv')#NOTE make sure that the value range of LAIu is [0.01, 5]
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_B_WYTHAM/0323_222945/model_best_testset_analyzer_frm4veg.csv')
CSV_PATH2 = os.path.join(
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM/0329_075709/model_best_testset_analyzer_frm4veg.csv')
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM/0323_204415/model_best_testset_analyzer_frm4veg.csv')
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_1/0406_094832/model_best_testset_analyzer_frm4veg.csv')
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3/0406_105747/model_best_testset_analyzer_frm4veg.csv')
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3_prior/0406_114131/model_best_testset_analyzer_frm4veg.csv')
    # BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.1/0407_102159/model_best_testset_analyzer_frm4veg.csv')
    BASE_PATH, 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.2/0407_120629/model_best_testset_analyzer_frm4veg.csv')


CSV_PATH_INSITU = '/maps/ys611/MAGIC/data/raw/wytham/csv_preprocessed_data/frm4veg_insitu.csv'

SAVE_PATH = os.path.join(BASE_PATH, 
                        # 'rtm/models/PHYS_VAE_RTM_C_WYTHAM/0329_075709/plots_frm4veg')
                        # 'rtm/models/PHYS_VAE_RTM_C_WYTHAM/0323_204415/plots_frm4veg')
                        # 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_1/0406_094832/plots_frm4veg')
                        # 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3/0406_105747/plots_frm4veg')
                        # 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3_prior_std0.1/0406_114131/plots_frm4veg')
                        # 'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.1/0407_102159/plots_frm4veg')
                        'rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.2/0407_120629/plots_frm4veg')



S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
S2_names = {
    'B02_BLUE': 'B2', 'B03_GREEN': 'B3', 'B04_RED': 'B4', 'B05_RE1': 'B5',
    'B06_RE2': 'B6', 'B07_RE3': 'B7', 'B08_NIR1': 'B8', 'B8A_NIR2': 'B8a',
    'B09_WV': 'B9', 'B11_SWI1': 'B11', 'B12_SWI2': 'B12'
}
# rtm_paras = json.load(open('/maps/ys611/MAGIC/configs/rtm_paras.json'))# Range of LAIu has been changed from [0.01, 1] to [0.01, 5]
rtm_paras = json.load(open('/maps/ys611/MAGIC/configs/rtm_paras_exp.json'))# Range of LAIu has been changed from [0.01, 1] to [0.01, 5]

ATTRS = list(rtm_paras.keys())
# for each attr in ATTRS, create a LaTex variable name like $Z_{\mathrm{attr}}$
ATTRS_LATEX = {
    'N': '$Z_{\mathrm{N}}$', 'cab': '$Z_{\mathrm{cab}}$', 'cw': '$Z_{\mathrm{cw}}$',
    'cm': '$Z_{\mathrm{cm}}$', 'LAI': '$Z_{\mathrm{LAI}}$', 'LAIu': '$Z_{\mathrm{LAIu}}$',
    'fc': '$Z_{\mathrm{fc}}$'
    }
ATTRS_INSITU = {
    'LAIu': 'LAI_down',
    'LAI': 'LAI_up',
    'fc': 'FCOVER_up',
    'cab': 'LCC'
}
ATTRS_VANILLA = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

NUM_BINS = 100
# mkdir if the save path does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# read the csv file
df0 = pd.read_csv(CSV_PATH0)
df1 = pd.read_csv(CSV_PATH1)
df2 = pd.read_csv(CSV_PATH2)
df_insitu = pd.read_csv(CSV_PATH_INSITU)

# retrieve the target and output bands to original scale
# MEAN = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/train_x_mean.npy')
# SCALE = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/train_x_scale.npy')
MEAN = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/insitu_period/train_x_mean.npy')
SCALE = np.load('/maps/ys611/MAGIC/data/processed/rtm/wytham/insitu_period/train_x_scale.npy')
for x in ['target', 'output']:
    df1[[f'{x}_{band}' for band in S2_BANDS]] = df1[[f'{x}_{band}' for band in S2_BANDS]]*SCALE + MEAN
    df2[[f'{x}_{band}' for band in S2_BANDS]] = df2[[f'{x}_{band}' for band in S2_BANDS]]*SCALE + MEAN
df2[[f'init_output_{band}' for band in S2_BANDS]] = df2[[f'init_output_{band}' for band in S2_BANDS]]*SCALE + MEAN
df2[[f'bias_{band}' for band in S2_BANDS]] = df2[[f'bias_{band}' for band in S2_BANDS]]*SCALE# df2[[f'init_output_{band}' for band in S2_BANDS]] = df2[[f'init_output_{band}' for band in S2_BANDS]]*SCALE + MEAN
    
# dates = ['2018.04.20', '2018.05.05', '2018.05.07', '2018.05.15', '2018.05.17', 
#          '2018.06.06', '2018.06.11', '2018.06.26', '2018.06.29', '2018.07.06', 
#          '2018.07.11', '2018.07.24', '2018.08.05', '2018.09.02', '2018.09.27', 
#          '2018.10.09', '2018.10.19', '2018.10.22']
dates = ['2018.06.26', '2018.06.29', '2018.07.06', '2018.07.11']

def r_square(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

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
        if pred_attr == 'LAI':
            # Multiply LAI by fractional cover to get actual area-based LAI
            if 'latent_fc' not in df_pred.columns:
                raise ValueError("'latent_fc' is required in df_pred to convert LAI")
            pred_values = pred_values * df_pred['latent_fc'].values
        elif pred_attr == 'cab':
            # Convert cab from μg/cm² to g/m²
            pred_values = pred_values / 100
        
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
    df_pred = df2[df2['date'] == date]
    df_merged = pred2insitu(df_insitu, df_pred, ATTRS_INSITU)
    # Scatter plot for the predicted variables and insitu measurements
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (attr, insitu_attr) in enumerate(ATTRS_INSITU.items()):
        ax = axes[i // 2, i % 2]
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
            s=20,
            color='blue',
            alpha=0.5,
            # set the point size
            linewidth=0.5,
            marker='o',
        )
        # Scatter plot, add the plot name as text for each point
        # for j, plot in enumerate(df_merged_filtered['plot']):
        #     ax.text(df_merged_filtered[f'{insitu_attr}_insitu'].iloc[j], 
        #             df_merged_filtered[f'{insitu_attr}_pred'].iloc[j], 
        #             plot, fontsize=10)

        # r2 = r_square(df_merged_filtered[f'{insitu_attr}_insitu'], df_merged_filtered[f'{insitu_attr}_pred'])
        fontsize = 25

        # ax.set_title(f'{ATTRS_LATEX[attr]} vs {insitu_attr}')
        xlabel = f'{ATTRS_LATEX[attr]} (in-situ)'
        ylabel = f'{ATTRS_LATEX[attr]} (predicted)'
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
        # ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
    # set the title for the whole figure
    fig.suptitle(f'Pred vs In-situ on {date}', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'linescatter_corr_insitu_v_pred_{date}.png'))
    plt.show()


# %%
