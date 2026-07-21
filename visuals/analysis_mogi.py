# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pykalman import KalmanFilter
# update plt to allow latex bold font
# plt.rcParams['text.usetex'] = True
# %%
MEAN = np.load('/maps/ys611/MAGIC/data/processed/mogi/train_x_mean.npy')
SCALE = np.load('/maps/ys611/MAGIC/data/processed/mogi/train_x_scale.npy')
mogi_paras = json.load(open(
    '/maps/ys611/MAGIC/configs/mogi_paras.json'))
# update dV to 10^6 m^3 scale for better visualization
mogi_paras['dV'] = {'min': -10, 'max': 10}
# ATTRS = list(mogi_paras.keys())
ATTRS = {
    'xcen': '$Z_{\mathrm{x_m}}$ (km)', 
    'ycen': '$Z_{\mathrm{y_m}}$ (km)', 
    'd': '$Z_{\mathrm{d}}$ (km)', 
    'dV': '$Z_{\mathrm{\Delta V}}$ ($10^6~\mathrm{m^3}$)'
}
assert list(ATTRS.keys()) == list(mogi_paras.keys())
station_info = json.load(open(
    '/maps/ys611/MAGIC/configs/station_info.json'))
STATIONS = list(station_info.keys())
DIRECTIONS = ['ux', 'uy', 'uz']
dirs = {
    'ux': 'East (mm)',
    'uy': 'North (mm)',
    'uz': 'Vertical (mm)'
}
GPS = []
for direction in DIRECTIONS:
    for station in STATIONS:
        GPS.append(f'{direction}_{station}')
# rescale the output to the original scale
def recale_output(df, mean, scale, corr=True):
    for attr in ['xcen', 'ycen', 'd']:
        df[f'latent_{attr}'] = df[f'latent_{attr}'] / 1000 # scale back to km
    for attr in ['dV']:
        df[f'latent_{attr}'] = df[f'latent_{attr}'] / np.power(10, 6) # scale back to 10^6 m^3
    for i, gps in enumerate(GPS):
        df[f'output_{gps}'] = df[f'output_{gps}'] * scale[i] + mean[i]
        df[f'target_{gps}'] = df[f'target_{gps}'] * scale[i] + mean[i]
        if corr:
            df[f'init_output_{gps}'] = df[f'init_output_{gps}'] * scale[i] + mean[i]
            df[f'bias_{gps}'] = df[f'bias_{gps}'] * scale[i]
    df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
    return df

def r_square(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# define an exponential moving average function to fit the 1D time series data
def ema_filter(data, alpha=0.2):
    ema = [data[0]]  # Initialize EMA with first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

# define a kalman filter to fit the 1D time series

def kalman_filter(data, alpha=0.002):
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=alpha)
    state_means, _ = kf.filter(data)
    smooth_state_means, _ = kf.smooth(data)
    return smooth_state_means

# %%
# BASE_PATH = '/maps/ys611/MAGIC/saved/mogi/'
BASE_PATH = '/maps/ys611/MAGIC/saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_191930_rank4/models'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi/0509_103601_smooth'
# TODO add comparison results for wosmooth case in appendix
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0509_103248_wosmooth'

CSV_PATH0 = os.path.join( 
    BASE_PATH, 'model_best_testset_analyzer.csv'
)
# CSV_PATH1 = os.path.join(
#     BASE_PATH, 'models/PHYS_VAE_Mogi_B_seq/1210_093812',
#     'model_best_testset_analyzer.csv'
# )

# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0509_103248_wosmooth'
# CSV_PATH3 = os.path.join(
#     BASE_PATH, 'PHYS_VAE_Mogi_C/1208_181105_seq',
#     'model_best_testset_analyzer_test.csv'
# )

df0 = recale_output(pd.read_csv(CSV_PATH0), MEAN, SCALE)
# df1 = recale_output(pd.read_csv(CSV_PATH1), MEAN, SCALE, corr=False) 
# df2 = recale_output(pd.read_csv(CSV_PATH3), MEAN, SCALE)

SAVE_PATH = os.path.join(BASE_PATH, 'plots')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

INFLATION_START = pd.Timestamp('2008-03-01')
INFLATION_END = pd.Timestamp('2008-07-01')

# %% Plot the line scatter for the output and target of each GPS
# df = df0
# for direction in ['ux', 'uy', 'uz']:
for direction in ['uz']:
    # fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))
    # for i, station in enumerate(station_info.keys()):
        # ax = axs[i//4, i % 4]
    i = 0
    for station in ['AV08', 'AV12']:
        for df in [df0, df1]:
            if i % 2 == 0:
                color = 'blue'
                label = 'w/ $\mathbf{C}$'
            else:
                color = 'red'
                label = 'w/o $\mathbf{C}$'
            ax = axs[i]
            i += 1
            gps = f'{direction}_{station}'
            sns.scatterplot(x='target_'+gps, y='output_'+gps, data=df, ax=ax, s=8,
                            alpha=0.5, color=color, label=label)
            # rmse = np.sqrt(np.mean((df[f'target_{gps}'] - df[f'output_{gps}'])**2))
            # NOTE calculate the R-square to measure the linearity of the data
            r2 = r_square(df[f'target_{gps}'], df[f'output_{gps}'])
            fontsize = 30
            ax.set_title(station, fontsize=fontsize)
            xlabel = '$X_{\mathrm{GPS}}$'
            ylabel = '$X_{\mathrm{GPS, B}}$' if i%2 == 0 else '$X_{\mathrm{GPS, C}}$'
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
            # make sure all ticks are rounded to 1 decimal place
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            # set RMSE as a legend
            # ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
            ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_PATH, f'{direction}_output_target_corr_v_wocorr.png'))
    plt.show()

# %% Plot the line scatter for the output and target of each GPS
# NOTE neurips full plot
# NOTE: NOW BEING USED
df = df0
# color = 'red'
color = 'blue'
# for token in ['output', 'init_output']:
for token in ['init_output', 'output']:
# token = 'init_output'
    ylabel = '$X_{\mathrm{GPS, C}}$' if token == 'output' else '$X_{\mathrm{GPS, F}}$'
    for direction in ['ux', 'uy', 'uz']:
        # fig, axs = plt.subplots(3, 4, figsize=(24, 16))
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        # for i, station in enumerate(station_info.keys()):
        for i, station in enumerate(['AKGG', 'AKMO', 'AV10', 'AV12']):
            # ax = axs[i//4, i % 4]
            ax = axs[i]
            gps = f'{direction}_{station}'
            sns.scatterplot(x='target_'+gps, y=f'{token}_'+gps, data=df, ax=ax, s=8,
                            alpha=0.5, color=color)
            # rmse = np.sqrt(np.mean((df[f'target_{gps}'] - df[f'output_{gps}'])**2))
            r2 = r_square(df[f'target_{gps}'], df[f'{token}_{gps}'])
            fontsize = 30
            ax.set_title(station, fontsize=fontsize)
            xlabel = '$X_{\mathrm{GPS}}$'
            # ylabel = '$X_{\mathrm{GPS, B}}$'
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            # set the same ticks for both x and y axes
            ax.tick_params(axis='both', which='major', labelsize=25)
            # Ensure tick labels have only one decimal place for both axes
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            
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
            # make sure all ticks are rounded to 1 decimal place
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
            # set RMSE as a legend
            # ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
            ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, f'{direction}_{token}_target_wocorr_AKGG_AKMO_AV10_AV12.png'))
        plt.show()

# %% plot the histogram of the four variables
# NOTE plot with w/o correction case
# NOTE: NOW BEING USED
NUM_BINS = 100
# df = df0
fig, axs = plt.subplots(1, 4, figsize=(25, 5))
for i, attr in enumerate(ATTRS.keys()):
    ax = axs[i]
    # sns.histplot(
    #     df1[f'latent_{attr}'].values,
    #     bins=NUM_BINS,
    #     ax=ax,
    #     color='red',
    #     alpha=0.5,
    #     # label='w/o $C$'
    #     # set bold font for the latex label $C$
    #     label='w/o $\mathbf{C}$'
    # )
    sns.histplot(
        df0[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        alpha=0.5,
        # label='w/ $\mathbf{C}$'
    )
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlim(mogi_paras[attr]['min'], mogi_paras[attr]['max'])
    fontsize = 30
    ax.set_xlabel(ATTRS[attr], fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.yaxis.labelpad = 10
    ax.legend(fontsize=fontsize-5)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'histogram_latent_corr.png'))
plt.show()

# %%
# scatter plot of the latent variables given date order
# first sort the dataframe by date in ascending order
# NOTE: NOW BEING USED
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
# df = df.sort_values(by='date')
fig, axs = plt.subplots(2, 2, figsize=(20, 8))
for i, attr in enumerate(ATTRS.keys()):
    ax = axs[i//2, i % 2]
    # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
    ax.axvspan(
        INFLATION_START, INFLATION_END,
        color='pink', alpha=0.4
    )
    # Add benchmark horizontal lines
    # if attr == 'xcen':
    #     ax.axhline(y=-1.5, color='red', linestyle='--', linewidth=2, alpha=1.0)
    # elif attr == 'ycen':
    #     ax.axhline(y=1.6, color='red', linestyle='--', linewidth=2, alpha=1.0)
    # elif attr == 'd':
    #     ax.axhline(y=8, color='red', linestyle='--', linewidth=2, alpha=1.0)
    # elif attr == 'dV':
    #     ax.axhline(y=4, color='red', linestyle='--', linewidth=2, alpha=1.0)
    
    sns.scatterplot(
        x='date', y=f'latent_{attr}', data=df, ax=ax, s=15, alpha=0.5, color='purple'
    )
    # plot the fitted curve for only dV
    smooth_state_means = kalman_filter(
        df[f'latent_{attr}'].values, alpha=0.002)
    ax.plot(df['date'], smooth_state_means, color='red', linewidth=3)

    fontsize = 32
    # ax.set_title(attr, fontsize=fontsize)
    ax.set_xlabel('Date', fontsize=fontsize)
    # show only years for Date
    ax.set_ylabel(ATTRS[attr], fontsize=fontsize)
    ax.set_ylim(mogi_paras[attr]['min'], mogi_paras[attr]['max'])
    ax.tick_params(axis='both', which='major', labelsize=25) #25
    # rotate the x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(-20)
    
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'latent_vars_date.png'))
plt.show()

# %%
# Compare kalman filter lines from two models (PILA and HVAE)
# Load data from both models
SMPL_PATH = '/maps/ys611/MAGIC/saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_191930_rank4/models'
BASELINE_PATH = '/maps/ys611/MAGIC/saved/mogi/PHYS_VAE_MOGI_C/1016_192308/models'

CSV_PATH_SMPL = os.path.join(SMPL_PATH, 'model_best_testset_analyzer.csv')
CSV_PATH_BASELINE = os.path.join(BASELINE_PATH, 'model_best_testset_analyzer.csv')

df_smpl = recale_output(pd.read_csv(CSV_PATH_SMPL), MEAN, SCALE)
df_baseline = recale_output(pd.read_csv(CSV_PATH_BASELINE), MEAN, SCALE)

df_smpl['date'] = pd.to_datetime(df_smpl['date'], format='%Y.%m.%d')
df_baseline['date'] = pd.to_datetime(df_baseline['date'], format='%Y.%m.%d')

fig, axs = plt.subplots(2, 2, figsize=(20, 8))
for i, attr in enumerate(ATTRS.keys()):
    ax = axs[i//2, i % 2]
    # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
    ax.axvspan(
        INFLATION_START, INFLATION_END,
        color='pink', alpha=0.4
    )
    # Add benchmark horizontal lines (dotted)
    if attr == 'xcen':
        ax.axhline(y=-1.5, color='red', linestyle='--', linewidth=2, alpha=1.0)
    elif attr == 'ycen':
        ax.axhline(y=1.6, color='red', linestyle='--', linewidth=2, alpha=1.0)
    elif attr == 'd':
        ax.axhline(y=8, color='red', linestyle='--', linewidth=2, alpha=1.0)
    elif attr == 'dV':
        ax.axhline(y=4, color='red', linestyle='--', linewidth=2, alpha=1.0)
    
    # Plot kalman filter lines for both models
    smooth_state_means_smpl = kalman_filter(
        df_smpl[f'latent_{attr}'].values, alpha=0.002)
    smooth_state_means_baseline = kalman_filter(
        df_baseline[f'latent_{attr}'].values, alpha=0.002)
    
    ax.plot(df_baseline['date'], smooth_state_means_baseline, color='red', linewidth=3)
    ax.plot(df_smpl['date'], smooth_state_means_smpl, color='blue', linewidth=3.5)

    # Set y-axis limits based on min and max of fitted curves
    y_min = min(np.min(smooth_state_means_smpl), np.min(smooth_state_means_baseline))
    y_max = max(np.max(smooth_state_means_smpl), np.max(smooth_state_means_baseline))
    y_range = y_max - y_min
    # Add 30% padding on both sides
    ax.set_ylim(y_min - 0.3 * y_range, y_max + 0.3 * y_range)

    fontsize = 32
    # ax.set_title(attr, fontsize=fontsize)
    ax.set_xlabel('Date', fontsize=fontsize)
    # show only years for Date
    ax.set_ylabel(ATTRS[attr], fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=25) #25
    # rotate the x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(-20)
    
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, 'latent_vars_date_comparison.png'))
plt.show()

# Calculate and save statistics for the inflation period (using fitted Kalman filter curves and raw predictions)
# Create lists to store statistics for both fitted curves and raw predictions
stats_data_fitted = []
stats_data_raw = []

for attr in ATTRS.keys():
    # Apply Kalman filter to full datasets for fitted curve statistics
    smooth_state_means_smpl = kalman_filter(
        df_smpl[f'latent_{attr}'].values, alpha=0.002)
    smooth_state_means_baseline = kalman_filter(
        df_baseline[f'latent_{attr}'].values, alpha=0.002)
    
    # Create temporary dataframes with fitted values
    df_smpl_fitted = df_smpl.copy()
    df_smpl_fitted[f'fitted_{attr}'] = smooth_state_means_smpl
    df_baseline_fitted = df_baseline.copy()
    df_baseline_fitted[f'fitted_{attr}'] = smooth_state_means_baseline
    
    # Filter to only include dates within the inflation period
    df_smpl_inflation = df_smpl_fitted[(df_smpl_fitted['date'] >= INFLATION_START) & (df_smpl_fitted['date'] <= INFLATION_END)]
    df_baseline_inflation = df_baseline_fitted[(df_baseline_fitted['date'] >= INFLATION_START) & (df_baseline_fitted['date'] <= INFLATION_END)]
    
    # Get fitted curve values for both models within inflation period
    smpl_fitted_values = df_smpl_inflation[f'fitted_{attr}'].values
    baseline_fitted_values = df_baseline_inflation[f'fitted_{attr}'].values
    
    # Get raw prediction values for both models within inflation period
    smpl_raw_values = df_smpl_inflation[f'latent_{attr}'].values
    baseline_raw_values = df_baseline_inflation[f'latent_{attr}'].values
    
    # Calculate statistics for PILA model (fitted curve)
    stats_data_fitted.append({
        'Variable': attr,
        'Model': 'PILA',
        'Min': np.min(smpl_fitted_values),
        'Max': np.max(smpl_fitted_values),
        'Mean': np.mean(smpl_fitted_values),
        'Std': np.std(smpl_fitted_values),
        'Period_Start': INFLATION_START.strftime('%Y-%m-%d'),
        'Period_End': INFLATION_END.strftime('%Y-%m-%d')
    })
    
    # Calculate statistics for HVAE model (fitted curve)
    stats_data_fitted.append({
        'Variable': attr,
        'Model': 'HVAE',
        'Min': np.min(baseline_fitted_values),
        'Max': np.max(baseline_fitted_values),
        'Mean': np.mean(baseline_fitted_values),
        'Std': np.std(baseline_fitted_values),
        'Period_Start': INFLATION_START.strftime('%Y-%m-%d'),
        'Period_End': INFLATION_END.strftime('%Y-%m-%d')
    })
    
    # Calculate statistics for PILA model (raw predictions)
    stats_data_raw.append({
        'Variable': attr,
        'Model': 'PILA',
        'Min': np.min(smpl_raw_values),
        'Max': np.max(smpl_raw_values),
        'Mean': np.mean(smpl_raw_values),
        'Std': np.std(smpl_raw_values),
        'Period_Start': INFLATION_START.strftime('%Y-%m-%d'),
        'Period_End': INFLATION_END.strftime('%Y-%m-%d')
    })
    
    # Calculate statistics for HVAE model (raw predictions)
    stats_data_raw.append({
        'Variable': attr,
        'Model': 'HVAE',
        'Min': np.min(baseline_raw_values),
        'Max': np.max(baseline_raw_values),
        'Mean': np.mean(baseline_raw_values),
        'Std': np.std(baseline_raw_values),
        'Period_Start': INFLATION_START.strftime('%Y-%m-%d'),
        'Period_End': INFLATION_END.strftime('%Y-%m-%d')
    })

# Convert to DataFrames and save as CSV files
stats_df_fitted = pd.DataFrame(stats_data_fitted)
stats_df_raw = pd.DataFrame(stats_data_raw)

# Use the SAVE_PATH from the comparison models
SAVE_PATH_COMPARISON = os.path.join(SMPL_PATH, 'plots')
if not os.path.exists(SAVE_PATH_COMPARISON):
    os.makedirs(SAVE_PATH_COMPARISON)

stats_csv_path_fitted = os.path.join(SAVE_PATH_COMPARISON, 'latent_vars_inflation_period_stats_fitted.csv')
stats_csv_path_raw = os.path.join(SAVE_PATH_COMPARISON, 'latent_vars_inflation_period_stats_raw.csv')

stats_df_fitted.to_csv(stats_csv_path_fitted, index=False)
stats_df_raw.to_csv(stats_csv_path_raw, index=False)

print(f"Fitted curve statistics saved to: {stats_csv_path_fitted}")
print(f"Raw prediction statistics saved to: {stats_csv_path_raw}")


# %%
# scatter plot of the gps displacement given date order for both output and target
# NOTE: NOW BEING USED
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
# for direction in ['uy']:
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//3, i % 3]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=10, alpha=0.5, 
            color='orange'
        )
        sns.scatterplot(
            x='date', y=f'output_{gps}', data=df, ax=ax, s=10, alpha=0.4, 
            color='blue'
        )
        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
    plt.tight_layout()
    # plt.savefig(os.path.join(
    #     SAVE_PATH, f'{direction}_target_v_finaloutput_gps_date.png'))
    plt.show()

"""
The following code is for AE_Mogi_corr only
"""
# %%
# scater plot of the corrected bias given date order
# NOTE: NOW BEING USED
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
# for direction in ['uy']:
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//3, i % 3]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'bias_{gps}', data=df, ax=ax, s=10, alpha=0.5
        )
        fontsize = 32
        # smooth_bias = kalman_filter(df[f'bias_{gps}'].values, alpha=0.01)
        # ax.plot(df['date'], smooth_bias, color='red', linewidth=2)
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'{direction}_bias_gps_date.png'))
    plt.show()
# %%
# plot the scatter plot of the init_output from Mogi and target
# NOTE: NOW BEING USED
df = df0
# fig, axs = plt.subplots(2, 3, figsize=(30, 10))
fig, axs = plt.subplots(12, 3, figsize=(30, 60))
for i, direction in enumerate(['ux', 'uy', 'uz']):
    # for j, station in enumerate(['AV08', 'AV12']):
    for j, station in enumerate(station_info.keys()):
        ax = axs[j, i]
        # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
        ax.axvspan(
            INFLATION_START, INFLATION_END,
            color='pink', alpha=0.4
        )
        gps = f'{direction}_{station}'
        # sns.scatterplot(
        #     x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
        #     color='orange'
        # )
        sns.scatterplot(
            x='date', y=f'init_output_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='blue'
        )
        # kalman filter to fit the curve for init_output
        smooth_init_output = kalman_filter(df[f'init_output_{gps}'].values, alpha=0.005)
        ax.plot(df['date'], smooth_init_output, color='red', linewidth=3)

        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # set y limit same so easier for visual
        y_min = -22 if direction == 'uz' else -12
        y_max = 22 if direction == 'uz' else 12
        ax.set_ylim(y_min, y_max)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, f'{direction}_target_v_mogioutput_gps_date_AV08_12.png'))
plt.savefig(os.path.join(
    SAVE_PATH, 'target_v_mogioutput_gps_date.png'), dpi=300)
plt.show()

# %% NOTE comparing the effects of the smoothness term
# plot the scatter plot of the init_output from Mogi and target
# df = df0
station = 'AV08'
fig, axs = plt.subplots(2, 3, figsize=(30, 10))
for i, direction in enumerate(['ux', 'uy', 'uz']):
    # for j, station in enumerate(['AV08', 'AV12']):
    for j, df in enumerate([df2, df0]):
        ax = axs[j, i]
        # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
        ax.axvspan(
            INFLATION_START, INFLATION_END,
            color='pink', alpha=0.4
        )
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='orange'
        )
        sns.scatterplot(
            x='date', y=f'init_output_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='blue'
        )
        # kalman filter to fit the curve for init_output
        smooth_init_output = kalman_filter(df[f'init_output_{gps}'].values, alpha=0.005)
        ax.plot(df['date'], smooth_init_output, color='red', linewidth=2)

        fontsize = 32
        title  = f'{station} (w/o smoothness)' if j == 1 else f'{station} (w/ smoothness)'
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # set y limit same so easier for visual
        y_min = -22 if direction == 'uz' else -12
        y_max = 22 if direction == 'uz' else 12
        ax.set_ylim(y_min, y_max)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, f'target_v_mogioutput_gps_date_AV08_smoothness_ablation.png'))
plt.show()
# %%
# TODO combine csvs from both train and test sets to plot the gps displacements
# NOTE: This timeseries plot combining both train and test sets can be done later
# NOTE: NOW BEING USED
# CSV_PATH = os.path.join(
#     BASE_PATH, 'AE_Mogi_corr/0509_102619_smooth', 
#     'model_best_testset_analyzer_train.csv'
# )
# df0_train = recale_output(pd.read_csv(CSV_PATH), MEAN, SCALE)
# df0_full = pd.concat([df0, df0_train], axis=0)
# # select the data from df0_full up to 2014-01-01
# df0_full = df0_full[df0_full['date'] < pd.Timestamp('2013-06-01')]
# df= df0_full
df = df0
# get the last date in the test set
last_date = df0['date'].max() # 2009-07-29
# for AV08, AV12 (each row), plot the gps displacements of 'uz' in each column
# column 1: corrected_output v. target, column 2: init_output v. target, column 3: bias
# fig, axs = plt.subplots(2, 3, figsize=(30, 10))
fig, axs = plt.subplots(12, 3, figsize=(30, 60))
direction = 'uy'
# for i, station in enumerate(['AV08', 'AV12']):
for i, station in enumerate(station_info.keys()):
    for j, attr in enumerate(['init_output', 'bias', 'output']):
        ax = axs[i, j]
        gps = f'{direction}_{station}'
        # plot the background of the plot as pink for date between 2008-03-01 to 2008-07-01
        ax.axvspan(
            INFLATION_START, INFLATION_END,
            color='pink', alpha=0.4
        )
        if attr in ['output', 'init_output']:
            sns.scatterplot(
                x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
                color='orange'
            )
        sns.scatterplot(
            x='date', y=f'{attr}_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='blue'
        )
        # kalman filter to fit the curve for init_output, 0.005 is the default alpha
        if attr in ['output', 'init_output']:
            smooth_init_output = kalman_filter(df[f'{attr}_{gps}'].values, alpha=0.005)
            ax.plot(df['date'], smooth_init_output, color='red', linewidth=2)
        # plot a blue line to indicate the last date in the test set
        # ax.axvline(x=last_date, color='grey', linestyle='-', linewidth=2)
        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        y_min = -21 if direction == 'uz' else -11
        y_max = 21 if direction == 'uz' else 11
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, f'{direction}_mogi_v_bias_v_corr_gps_date_AV08_12.png'))
plt.savefig(os.path.join(
    SAVE_PATH, f'{direction}_mogi_v_bias_v_corr_gps_date.png'), dpi=300)
plt.show()

# %%
# NOTE: NOW BEING USED
# Simple visualization of latent auxiliary variables
def visualize_latent_aux_simple(
    df,
    save_path=None,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    standardize=True,
):
    """
    Simple latent_aux visualization.
    - 2 variables: direct 2D scatter
    - >2 variables: UMAP to 2D
    Points colored by phases: pre / inflation / post.
    """
    latent_aux_cols = [col for col in df.columns if col.startswith('latent_aux_')]
    n_aux = len(latent_aux_cols)
    if n_aux == 0:
        print("No latent auxiliary variables found in the dataframe.")
        return

    aux_data = df[latent_aux_cols].values.astype(np.float32, copy=False)
    if standardize:
        mean = np.mean(aux_data, axis=0, keepdims=True)
        std = np.std(aux_data, axis=0, keepdims=True)
        std[std == 0] = 1.0
        aux_data = (aux_data - mean) / std

    dates = pd.to_datetime(df['date'], format='%Y.%m.%d')
    phase = np.where(dates < INFLATION_START, 'pre', np.where(dates > INFLATION_END, 'post', 'inflation'))
    phase_to_color = {'pre': 'gray', 'inflation': 'crimson', 'post': 'navy'}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if n_aux == 2:
        for p in ['pre', 'inflation', 'post']:
            mask = (phase == p)
            ax.scatter(aux_data[mask, 0], aux_data[mask, 1], s=14, alpha=0.65, c=phase_to_color[p], label=p)
        ax.set_xlabel('latent_aux_1', fontsize=14)
        ax.set_ylabel('latent_aux_2', fontsize=14)
        ax.set_title('Latent Auxiliary Variables (2D)', fontsize=16)
        ax.legend(title='Phase')
    else:
        import umap
        X = np.ascontiguousarray(aux_data, dtype=np.float32)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
            print(f"UMAP: filtering out {X.shape[0] - np.count_nonzero(finite_mask)} non-finite rows")
        X = X[finite_mask]
        phase_used = phase[finite_mask]

        # Minimal compatibility shim for sklearn check_array (for version mismatches)
        try:
            import inspect
            from sklearn.utils import validation as _skval
            sig = inspect.signature(_skval.check_array)
            if 'ensure_all_finite' not in sig.parameters:
                _orig = _skval.check_array
                def _check_array_compat(A, *args, ensure_all_finite=True, **kwargs):
                    kwargs['force_all_finite'] = ensure_all_finite
                    return _orig(A, *args, **kwargs)
                _skval.check_array = _check_array_compat
                try:
                    import umap.umap_ as _umap_mod
                    _umap_mod.check_array = _check_array_compat
                except Exception:
                    pass
        except Exception:
            pass

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        try:
            emb = reducer.fit_transform(X)
        except TypeError:
            reducer.fit(X)
            emb = reducer.transform(X)

        for p in ['pre', 'inflation', 'post']:
            mask = (phase_used == p)
            ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.7, c=phase_to_color[p], label=p)
        ax.set_xlabel('UMAP-1', fontsize=14)
        ax.set_ylabel('UMAP-2', fontsize=14)
        ax.set_title('Latent Auxiliary Variables (UMAP 2D)', fontsize=16)
        ax.legend(title='Phase')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_latent_aux_pairwise(df, save_path=None):
    """
    Pairwise scatter plots for all latent_aux variables with LaTeX axis labels.
    """
    latent_aux_cols = [col for col in df.columns if col.startswith('latent_aux_')]
    n_aux = len(latent_aux_cols)
    if n_aux == 0:
        print("No latent auxiliary variables found in the dataframe.")
        return

    aux_labels = [f'$Z_{{\\mathrm{{aux,{i+1}}}}}$' for i in range(n_aux)]
    fig, axs = plt.subplots(n_aux, n_aux, figsize=(4*n_aux, 4*n_aux))
    for i, attr1 in enumerate(latent_aux_cols):
        for j, attr2 in enumerate(latent_aux_cols):
            ax = axs[i, j]
            sns.scatterplot(x=attr1, y=attr2, data=df, ax=ax, s=8, alpha=0.5)
            fontsize = 22
            ax.set_xlabel(aux_labels[j], fontsize=fontsize)
            ax.set_ylabel(aux_labels[i], fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_latent_aux_timeseries(df, save_path=None):
    """
    Time series plots for each latent auxiliary variable.
    One subplot per latent_aux variable, arranged vertically.
    """
    latent_aux_cols = [col for col in df.columns if col.startswith('latent_aux_')]
    n_aux = len(latent_aux_cols)
    if n_aux == 0:
        print("No latent auxiliary variables found in the dataframe.")
        return
    
    print(f"Found {n_aux} latent auxiliary variables: {latent_aux_cols}")
    
    # Create subplots - one row per latent_aux variable
    fig, axes = plt.subplots(n_aux, 1, figsize=(15, 3*n_aux))
    if n_aux == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    dates = pd.to_datetime(df['date'], format='%Y.%m.%d')
    
    for i, col in enumerate(latent_aux_cols):
        ax = axes[i]
        
        # Plot the background for inflation period
        ax.axvspan(INFLATION_START, INFLATION_END, color='pink', alpha=0.4)
        
        # Plot the time series
        sns.scatterplot(x='date', y=col, data=df, ax=ax, s=8, alpha=0.8, color='blue')
        
        # Add Kalman filter smooth curve
        # smooth_values = kalman_filter(df[col].values, alpha=0.002)
        # ax.plot(dates, smooth_values, color='red', linewidth=2)
        
        # Formatting
        fontsize = 16
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(f'$Z_{{\\mathrm{{aux,{i+1}}}}}$', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Rotate x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# %%
# Example usage of the latent auxiliary variables visualization
# Since you have 8 latent auxiliary variables, the simple function will use UMAP

# Method 1: Simple visualization (UMAP for 8 variables)
visualize_latent_aux_simple(df0, save_path=os.path.join(SAVE_PATH, 'latent_aux_simple.png'))
# Method 2: Pairwise scatter plots (like RTM analysis)
visualize_latent_aux_pairwise(df0, save_path=os.path.join(SAVE_PATH, 'latent_aux_pairwise.png'))
# Method 3: Time series plots for each latent auxiliary variable
visualize_latent_aux_timeseries(df0, save_path=os.path.join(SAVE_PATH, 'latent_aux_timeseries.png'))

# %%

