import math
import numpy as np
import epanet.toolkit as en
import pandas as pd
import wntr.network
import yaml
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as mcolors


def run_hyd_sim(inp_path):
    ph = en.createproject()
    err = en.open(ph, inp_path, "net.rpt", "net.out")
    en.setqualtype(ph, qualType=en.CHEM, chemName='', chemUnits='', traceNode='')

    en.openH(ph)
    en.initH(ph, en.SAVE_AND_INIT)
    en.openQ(ph)
    en.initQ(ph, saveFlag=0)

    t_step = 1
    report_step = 3600
    res_dem = pd.DataFrame()
    res_qual = pd.DataFrame()

    while t_step > 0:
        # run hydraulic and quality simulations of the current time step
        # in case of real system this step is not needed
        # after applying the control values we wait for the measured values and save them as recorded
        t = en.runH(ph)
        t = en.runQ(ph)
        # record values of the target elements
        if t % report_step == 0:
            for node_idx in range(1, en.getcount(ph, en.NODE) + 1):
                node_id = en.getnodeid(ph, node_idx)
                demand = en.getnodevalue(ph, node_idx, en.DEMAND)
                quality = en.getnodevalue(ph, node_idx, en.QUALITY)
                res_dem.loc[t, node_id] = demand
                res_qual.loc[t, node_id] = quality

            t_step = en.nextH(ph)
            t_step = en.nextQ(ph)

    res_qual.index /= 3600
    res_dem.index /= 3600
    en.closeH(ph)
    en.closeQ(ph)
    en.close(ph)
    return res_dem, res_qual


def get_atd(demand, quality, n, y_ref, n_box=1000):
    """
    average target deviation - https://doi-org.ezlibrary.technion.ac.il/10.1061/(ASCE)WR.1943-5452.0001509
    atd  = sum(d' * abs(c*-c))

    :return:
    """
    positive_cols = demand.loc[:, (demand > 0).all()].columns
    demand = demand[positive_cols]
    quality = quality[positive_cols]
    weighted_d = demand.iloc[-n:, :].values / demand.iloc[-n:, :].values.sum()
    atd = (weighted_d * np.abs(quality.iloc[-n:, :].values - y_ref)).sum()
    mae = np.mean(np.abs(quality.iloc[-n:].values - y_ref))
    print(f"ATD: {atd:.3f}, MAE: {mae:.3f}")


def systemwide_statistics(data, n, max_boxes):
    n_plots = max(math.floor(len(data.columns) / max_boxes), 1)
    fig, axes = plt.subplots(nrows=n_plots)
    axes = np.atleast_2d(axes).ravel()
    for _ in range(n_plots):
        axes[_].boxplot(data.iloc[-n:, _ * max_boxes: (_ + 1) * max_boxes])
        axes[_].set_xticklabels(data.columns[_ * max_boxes: (_ + 1) * max_boxes])
        axes[_].grid()

    fig.text(0.45, 0.05, 'Junction')
    fig.text(0.02, 0.5, 'Chlorine Residuals (mg/L)', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.15, top=0.95, hspace=0.3)


def tempo_spatial(inp_path, times: list):
    wn = wntr.network.WaterNetworkModel(inp_path)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    n = len(times)
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = max(1, int(math.ceil(n / ncols)))

    min_quality = float('inf')
    max_quality = float('-inf')
    for t in times:
        quality = results.node['quality'].loc[t*3600, :] * 1000
        min_quality = min(min_quality, quality.min())
        max_quality = max(max_quality, quality.max())

    cmap = plt.get_cmap('Spectral_r')
    norm = mcolors.Normalize(vmin=min_quality, vmax=max_quality)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 5))
    axes = axes.ravel()
    for i, t in enumerate(times):
        res_qual = results.node['quality'].loc[t*3600, :] * 1000
        ax = wntr.graphics.plot_network(wn, node_attribute=res_qual, ax=axes[i], add_colorbar=False)
        ax.set_title(f"Time: {t%24} Hr", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.03, pad=0.01)
    cbar.set_label('Chlorine Residuals (mg/L)')
    plt.subplots_adjust(left=0.03, right=0.84, wspace=0.02, hspace=0.25, bottom=0.05)

    for i in range(len(times), len(axes)):
        fig.delaxes(axes[i])  # Delete each unused axis


def grid_search_stats(results_file, independent_cols, target_cols):
    df = pd.read_csv(results_file, index_col=0)

    n = max(len(target_cols), len(independent_cols))
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = max(1, int(math.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4))
    axes = axes.ravel()
    k = 0
    for indep_col, indep_label in independent_cols.items():
        for target_col, target_label in target_cols.items():
            grouped_data = df.groupby(indep_col)[target_col].apply(list)
            axes[k].boxplot(grouped_data, labels=grouped_data.index)
            axes[k].set_xlabel(indep_label)
            axes[k].set_ylabel(target_label)
            axes[k].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.0%}'.format((x-1)/100)))
            k += 1

    plt.subplots_adjust(top=0.96, wspace=0.32, left=0.085, right=0.96)

def visualize_hyperparam(lambdas_path, n_train_path=''):
    lambdas_res = pd.read_csv(lambdas_path, usecols=['lg', 'ly', 'lu', 'mae', 'v_rate'])
    lambdas_res = lambdas_res.loc[(0 < lambdas_res['lg']) & (0 < lambdas_res['ly']) & (0 < lambdas_res['lu'])]
    nrows = 3
    ncols = 1

    if n_train_path:
        n_train_res = pd.read_csv(n_train_path, usecols=['n_train', 'mae', 'v_rate'])
        nrows = 2
        ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 5))
    axes = axes.ravel()

    opt = lambdas_res[lambdas_res['mae'] == lambdas_res['mae'].min()].iloc[0]
    def plot_ax(ax, x, mae, v_rate, x_label):
        ax.plot(x, mae)
        ax.set_xscale('log')
        ax.set_xlabel(x_label)
        ax.tick_params('y', colors='C0')
        ax.grid()
        ax.set_ylabel('MAE', color='C0')
        axes_sec = ax.twinx()
        axes_sec.plot(x, v_rate, 'C1')
        axes_sec.tick_params('y', colors='C1')
        axes_sec.set_ylabel('Constraints Violation Rate', color='C1')
        return ax

    temp = lambdas_res.loc[(lambdas_res['ly'] == opt['ly']) & (lambdas_res['lu'] == opt['lu'])]
    axes[0] = plot_ax(axes[0], x=temp['lg'], mae=temp['mae'], v_rate=temp['v_rate'], x_label='$\lambda_g$')
    axes[0].annotate(f'$\lambda_y$={opt["ly"]}\n$\lambda_u={opt["lu"]}$', (0.07, 0.73), xycoords='axes fraction',
                     bbox=dict(facecolor='w', alpha=0.6, edgecolor='none'))

    temp = lambdas_res.loc[(lambdas_res['lg'] == opt['lg']) & (lambdas_res['lu'] == opt['lu'])]
    axes[1] = plot_ax(axes[1], x=temp['ly'], mae=temp['mae'], v_rate=temp['v_rate'], x_label='$\lambda_y$')
    axes[1].annotate(f'$\lambda_g$={opt["lg"]}\n$\lambda_u={opt["lu"]}$', (0.07, 0.73), xycoords='axes fraction')

    temp = lambdas_res.loc[(lambdas_res['lg'] == opt['lg']) & (lambdas_res['ly'] == opt['ly'])]
    axes[2] = plot_ax(axes[2], x=temp['lu'], mae=temp['mae'], v_rate=temp['v_rate'], x_label='$\lambda_u$')
    axes[2].annotate(f'$\lambda_g$={opt["lg"]}\n$\lambda_y={opt["ly"]}$', (0.07, 0.73), xycoords='axes fraction',
                     bbox=dict(facecolor='w', alpha=0.6, edgecolor='none'))

    if n_train_path:
        axes[3] = plot_ax(axes[3], x=n_train_res['n_train'], mae=n_train_res['mae'], v_rate=n_train_res['v_rate'],
                          x_label='T')
        axes[3].annotate(f'$\lambda_g$={opt["lg"]}\n$\lambda_y={opt["ly"]}$\n$\lambda_u={opt["lu"]}$',
                         (0.6, 0.63), xycoords='axes fraction',
                         bbox=dict(facecolor='w', alpha=0.6, edgecolor='none'))
        axes[3].set_xscale('linear')

    plt.subplots_adjust(top=0.96, hspace=0.35, wspace=0.6, left=0.09, right=0.9)






