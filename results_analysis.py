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
from matplotlib.ticker import MultipleLocator


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


def visualize_hyperparam(lambdas_path, n_train_path='', target_col='mae', target_y_label='MAE'):
    """
    Text box locations, fig size and subplots_adjust are manually adjusted with trial and error for each case study
    """
    min_lambda = 0
    lambdas_res = pd.read_csv(lambdas_path, usecols=['lg', 'ly', 'lu', target_col, 'v_rate'])
    lambdas_res = lambdas_res.loc[(min_lambda <= lambdas_res['lg']) & (min_lambda <= lambdas_res['ly']) & (min_lambda <= lambdas_res['lu'])]
    nrows = 3
    ncols = 1
    alpha= 0.7
    if n_train_path:
        n_train_res = pd.read_csv(n_train_path, usecols=['n_train', target_col, 'v_rate'])
        nrows = 2
        ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 6.5))
    axes = axes.ravel()
    opt = lambdas_res[lambdas_res[target_col] == lambdas_res[target_col].min()].iloc[0]

    def plot_ax(ax, x, mae, v_rate, x_label):
        ax.plot(x, mae, markersize=5, marker='o', markerfacecolor='w')
        ax.set_xscale('log')
        ax.set_xlabel(x_label)
        ax.tick_params('y', colors='C0')
        ax.grid(True, linestyle='--', c='grey', lw=0.5)
        ax.grid(True, which='minor', linestyle='--', c='grey', lw=0.5, alpha=0.3)
        ax.set_ylabel(target_y_label, color='C0')
        axes_sec = ax.twinx()
        axes_sec.plot(x, v_rate, markersize=5, c='C1', marker='o', markerfacecolor='w')
        axes_sec.tick_params('y', colors='C1')
        axes_sec.set_ylabel('Constraints Violation Rate', color='C1')
        return ax

    temp = lambdas_res.loc[(lambdas_res['ly'] == opt['ly']) & (lambdas_res['lu'] == opt['lu'])].sort_values("lg")
    axes[0] = plot_ax(axes[0], x=temp['lg'], mae=temp[target_col], v_rate=temp['v_rate'], x_label='$\lambda_g$')
    axes[0].annotate(f'$\lambda_y$={opt["ly"]}\n$\lambda_u={opt["lu"]}$', (0.06, 0.78), xycoords='axes fraction',
                     bbox=dict(facecolor='w', alpha=alpha, edgecolor='none'))
    #case 1 - 0.78. 0.08 # case 2 - 0.06, 0.75

    temp = lambdas_res.loc[(lambdas_res['lg'] == opt['lg']) & (lambdas_res['lu'] == opt['lu'])].sort_values("ly")
    axes[1] = plot_ax(axes[1], x=temp['ly'], mae=temp[target_col], v_rate=temp['v_rate'], x_label='$\lambda_y$')
    axes[1].annotate(f'$\lambda_g$={opt["lg"]}\n$\lambda_u={opt["lu"]}$', (0.06, 0.75), xycoords='axes fraction',
                     bbox=dict(facecolor='w', alpha=alpha, edgecolor='none'))

    temp = lambdas_res.loc[(lambdas_res['lg'] == opt['lg']) & (lambdas_res['ly'] == opt['ly'])].sort_values("lu")
    axes[2] = plot_ax(axes[2], x=temp['lu'], mae=temp[target_col], v_rate=temp['v_rate'], x_label='$\lambda_u$')
    axes[2].annotate(f'$\lambda_g$={opt["lg"]}\n$\lambda_y={opt["ly"]}$', (0.06, 0.75), xycoords='axes fraction',
                     bbox=dict(facecolor='w', alpha=alpha, edgecolor='none'))

    if n_train_path:
        axes[3] = plot_ax(axes[3], x=n_train_res['n_train'], mae=n_train_res[target_col], v_rate=n_train_res['v_rate'],
                          x_label='T')
        axes[3].annotate(f'$\lambda_g$={opt["lg"]}\n$\lambda_y={opt["ly"]}$\n$\lambda_u={opt["lu"]}$',
                         (0.75, 0.67), xycoords='axes fraction',
                         bbox=dict(facecolor='w', alpha=alpha, edgecolor='none'))
        axes[3].set_xscale('linear')
        axes[3].xaxis.set_minor_locator(MultipleLocator(25))

    plt.subplots_adjust(top=0.96, hspace=0.35, wspace=0.3, left=0.07, right=0.93)


def get_total_chlorine_mass(dem, qual, n):
    mass = -dem[['15', '43', '65']] * qual[['15', '43', '65']] * 3600 / 1000000
    print(f"Avg outflow (L/s):\n{-dem[['15', '43', '65']].iloc[-n:].mean(axis=0)}")
    print(f"Avg source concentration (mg/L):\n{qual[['15', '43', '65']].iloc[-n:].mean(axis=0)}")
    print(f"Total chlorine mass (kg):\n{mass.iloc[-n:].sum(axis=0)}")


def compare_to_pecci_et_al(n, y_ref, max_boxes):
    dem_ref, qual_ref = run_hyd_sim("Data/pecci_et_al/pescara_optimal_nv2_nb3_extended.inp")
    get_atd(dem_ref, qual_ref, n=n, y_ref=y_ref)
    dem, qual = run_hyd_sim("Output/pescara_output.inp")
    get_atd(dem, qual, n=n, y_ref=y_ref)

    positive_cols = dem.loc[:, (dem > 0).all()].columns
    qual = qual[positive_cols]
    qual_ref = qual_ref[positive_cols]

    colors = ['C0', 'C1']
    positions = np.arange(len(qual_ref.columns))
    width = 0.3

    n_plots = max(math.floor(len(positive_cols) / max_boxes), 1)
    fig, axes = plt.subplots(nrows=n_plots, figsize=(12, 6))
    axes = np.atleast_2d(axes).ravel()
    leg_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='k', alpha=0.6) for c in colors]

    for _ in range(n_plots):
        idx_min, idx_max = _ * max_boxes, (_ + 1) * max_boxes
        bp1 = axes[_].boxplot([qual[col].iloc[-n:] for col in qual.columns[idx_min: idx_max]],
                              positions=positions[:max_boxes] - width / 2, widths=width, patch_artist=True,
                              flierprops={'markersize': 4})
        bp2 = axes[_].boxplot([qual_ref[col].iloc[-n:] for col in qual_ref.columns[idx_min: idx_max]],
                              positions=positions[:max_boxes] + width / 2, widths=width, patch_artist=True,
                              flierprops={'markersize': 4})
        for bplot, color in zip([bp1, bp2], colors):
            for patch in bplot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for element in ['whiskers', 'caps', 'medians']:
                plt.setp(bplot[element], color='k')
            plt.setp(bplot['fliers'], markeredgecolor='grey')

        axes[_].set_xticks([_ for _ in range(len(positions[idx_min: idx_max]))])
        axes[_].set_xticklabels(qual.columns[idx_min: idx_max])
        axes[_].grid()

        if _ == 0:
            axes[_].legend(leg_elements, ['DeePC', 'Pecci et al.'], loc='upper left', ncol=2)

    fig.text(0.01, 0.5, 'Chlorine Residuals (mg\L)', va='center', rotation='vertical')
    fig.text(0.5, 0.02, 'Junction')
    fig.subplots_adjust(top=0.96, left=0.07, right=0.98, hspace=0.3)

    fig, axes = plt.subplots(nrows=3, sharex=True)
    for _, j in enumerate(["1", "39", "45"]):
        axes[_].plot(range(n), qual.loc[:, j].iloc[-n:], label='DeePC')
        axes[_].plot(range(n), qual_ref.loc[:, j].iloc[-n:], label='Pecci et al.')
        axes[_].grid()
        axes[_].set_title(f"Junction {int(j)}", fontsize=9)
        axes[_].legend(ncols=2)
        axes[_].set_ylim(0.8, 1.16)
    axes[-1].set_xlabel("Time (hr)")
    fig.text(0.02, 0.5, 'Chlorine Residuals (mg\L)', va='center', rotation='vertical')
    fig.subplots_adjust(top=0.96, hspace=0.3)


if __name__ == "__main__":
    # Case study I - Fossolo
    visualize_hyperparam("Output/fossolo_lambdas_search_const_seed.csv",
                         n_train_path="Output/fossolo_n_train_search.csv",
                         target_col='mae')
    grid_search_stats("Output/fossolo_noise.csv",
                      independent_cols={'noise': 'Noise Factor'},
                      target_cols={'mae': 'MAE', 'violations_rate': 'Constraints Violations Rate'})

    # Case study II - Pescara
    grid_search_stats("Output/old/pescara_noise.csv",
                        independent_cols={'noise': 'Noise Factor'},
                        target_cols={'mae': 'MAE', 'violations_rate': 'Violations Rate'})

    visualize_hyperparam("Output/pescara_lambdas_search_const_seed.csv",
                         n_train_path="Output/pescara_n_train_search.csv")

    dem, qual = run_hyd_sim("Output/pescara_output.inp")
    systemwide_statistics(qual, n=168, max_boxes=30)
    compare_to_pecci_et_al(n=168, y_ref=1, max_boxes=20)
    T = 600 + 24 * 5
    tempo_spatial("Output/pescara_output.inp", times=[T+0, T+3, T+6, T+9, T+12, T+15, T+18, T+21])
    plt.show()



