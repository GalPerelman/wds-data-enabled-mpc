import copy
import os
import yaml
import random
import epanet.toolkit as en
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

import ddpc
import system
import utils

COLORS = ["#0077B8", "#DF5353", "#fdc85e", "#b7b3aa"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)


class Experiment:
    def __init__(self, inp_path, control_links, control_nodes, target_nodes, target_param,
                 input_signal, n_train, input_loss,
                 y_ref, y_lb, y_ub, u_lb, u_ub, wait, t_ini,
                 horizon, lg, ly, lu, experiment_horizon, noise_std,
                 compare_signals, input_y_label, output_y_label, x_label,
                 plot_demand_pattern, plot_constraints, plot_legend_cols, plot_y_legend_label, plot_u_legend_label,
                 plot_y_labels_max_len
                 ):
        self.inp_path = inp_path
        self.control_links = control_links
        self.control_nodes = control_nodes
        self.target_nodes = target_nodes
        self.target_param = target_param
        self.input_signal = input_signal
        self.n_train = n_train
        self.input_loss = input_loss
        self.y_ref, self.y_lb, self.y_ub = y_ref, y_lb, y_ub
        self.u_lb, self.u_ub = u_lb, u_ub
        self.wait = wait
        self.t_ini = t_ini
        self.horizon = horizon
        self.lg, self.ly, self.lu = lg, ly, lu
        self.experiment_horizon = experiment_horizon
        self.noise_std = noise_std
        self.compare_signals = compare_signals
        self.input_y_label = input_y_label
        self.output_y_label = output_y_label
        self.x_label = x_label
        self.plot_demand_pattern = plot_demand_pattern
        self.plot_constraints = plot_constraints
        self.plot_legend_cols = plot_legend_cols
        self.plot_y_legend_label = plot_y_legend_label
        self.plot_u_legend_label = plot_u_legend_label
        self.plot_y_labels_max_len = plot_y_labels_max_len

        self.mae, self.me = None, None

        self.wds = system.WDSControl(inp_path=self.inp_path,
                                     control_links=self.control_links,
                                     control_nodes=self.control_nodes,
                                     target_nodes=self.target_nodes,
                                     target_param=self.target_param
                                     )

        self.init_input = utils.signals_generator(mu=self.input_signal["mu"],
                                                  sigma=self.input_signal["sigma"],
                                                  freq=self.input_signal["freq"],
                                                  noise=self.input_signal["noise"],
                                                  n=self.n_train,
                                                  times_values=self.input_signal["times_values"],
                                                  m=len(self.control_links) + len(self.control_nodes),
                                                  signal_type=self.input_signal["signal_type"]
                                                  )

    def run_experiment(self):
        # init_input = self.wds.get_init_input_random(mu=self.init_mu, sigma=self.init_sigma, n=self.n_train)
        # mask = (np.arange(len(init_input)) % 12 >= 6) & (np.arange(len(init_input)) % 12 <= 11)
        # init_input[mask] = 0

        model = ddpc.DDPC(sys=self.wds, input_loss=self.input_loss, y_ref=self.y_ref, y_lb=self.y_lb, y_ub=self.y_ub,
                          u_lb=self.u_lb, u_ub=self.u_ub, wait=self.wait, t_ini=self.t_ini, horizon=self.horizon,
                          lambda_g=self.lg, lambda_y=self.ly, lambda_u=self.lu,
                          experiment_horizon=self.experiment_horizon, noise_std=self.noise_std)

        # only for example 2
        # to do: move this to the input config
        # self.init_input[:, 0] = self.init_input[:, 0] * 0.9
        # self.init_input[:, 1] = self.init_input[:, 0] * 1.1
        # self.init_input[:, 2] = self.init_input[:, 0] * 0.9
        model.init_deepc(self.init_input)
        model.run()
        self.mae, self.me = self.get_error(y=self.wds.target_values[-self.experiment_horizon:, :])
        print(f"MAE: {self.mae:.3f} | ME: {self.me:.3f}")
        return self.wds

    def get_error(self, y):
        if not isinstance(self.y_ref, np.ndarray):
            y_ref = np.array([self.y_ref])
        else:
            y_ref = self.y_ref

        y_ref = np.tile(utils.extend_array_to_n_values(y_ref, self.experiment_horizon), (y.shape[1], 1)).T
        mae = np.mean(np.abs(y - y_ref))
        me = np.mean(y - y_ref)
        return mae, me

    def plot(self, sys, label, fig=None, moving_avg_size=0):
        n_inputs = sys.implemented.shape[1]

    def plot(self, sys, fig=None, moving_avg_size=0):
        n_inputs = 1 #sys.implemented.shape[1]
        n_outputs = sys.target_values.shape[1]
        if self.plot_demand_pattern:
            k = 1
        else:
            k = 0

        if fig is None:
            inputs_heights = [1 for _ in range(n_inputs + k)]
            fig, axes = plt.subplots(nrows=n_inputs + 1 + k, sharex=True, height_ratios=[1] + inputs_heights)
        else:
            axes = fig.axes

        t = len(sys.target_values[1:, 0])
        for i, element in enumerate(sys.target_nodes):
            axes[0].plot(sys.target_values[1:, i], label=self.plot_y_legend_label[i], zorder=4)
            axes[0].axvspan(0, self.n_train, facecolor='grey', alpha=0.2, zorder=0)
            axes[0].grid(True)
            axes[0].set_ylabel(utils.split_label(self.output_y_label, self.plot_y_labels_max_len))
        axes[0].hlines(y=self.y_ref, xmin=0, xmax=t, color='k', linestyles="--", zorder=5, label="$y_{ref}$")
        if self.plot_constraints:
            axes[0].hlines(y=self.y_lb, xmin=0, xmax=t, color='k', zorder=5, label='Constraints')
            axes[0].hlines(y=self.y_ub, xmin=0, xmax=t, color='k', zorder=5)
        if self.plot_legend_cols > 0:
            axes[0].legend(ncols=self.plot_legend_cols, fontsize=10)

        for i, element in enumerate(sys.control_nodes + sys.control_links):
            axes[1].step(range(t), sys.implemented[1:, i], label=self.plot_u_legend_label[i], zorder=5, where='post')
            if moving_avg_size > 0:
                window = np.ones(moving_avg_size)
                moving_avg = np.convolve(sys.implemented[1:, i], window, mode='valid') / 12
                pad_width = len(sys.implemented[1:, i]) - len(moving_avg)
                moving_avg = np.pad(moving_avg, (pad_width, 0), 'constant', constant_values=(np.nan,))
                axes[i + 1].plot(moving_avg, 'k')

            axes[1].axvspan(0, self.n_train, facecolor='grey', alpha=0.2, zorder=0)
            axes[1].grid(True)
            axes[1].set_ylabel(utils.split_label(self.input_y_label, self.plot_y_labels_max_len))
            if self.plot_legend_cols > 0:
                axes[1].legend(ncols=self.plot_legend_cols, fontsize=10)

        if self.plot_demand_pattern:
            pat = self.wds.get_demand_pattern(pat_idx=1)
            pat = utils.extend_array_to_n_values(pat, self.wds.implemented[1:, 0].shape[0])
            axes[-1].bar(range(len(pat)), pat, zorder=5, width=1.0, edgecolor='k', alpha=0.8, linewidth=0.1)
            axes[-1].set_ylabel(utils.split_label("Demand Pattern", self.plot_y_labels_max_len))

        axes[-1].set_xlabel(self.x_label)
        secax = axes[0].secondary_xaxis('top')
        secax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        secax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x) % 24}'))
        secax.set_xlabel('Hour of the Day')  # Label for clarity

        ymin, ymax = axes[0].get_ylim()
        if ymax >= 10000:
            if sys.target_param == en.PRESSURE:
                axes[0].set_ylim(-1, 100)
            else:
                axes[0].set_ylim(0, 2)

        fig.align_ylabels()
        return fig

    def plot_comparison_signals(self, fig, signals: list):
        axes = fig.axes
        for s in signals:
            sys = copy.deepcopy(self.wds)
            init_input = utils.signals_generator(mu=s["mu"],
                                                 sigma=s["sigma"],
                                                 freq=s["freq"],
                                                 noise=s["noise"],
                                                 n=self.experiment_horizon + self.n_train,
                                                 m=len(self.control_links) + len(self.control_nodes),
                                                 times_values=s["times_values"],
                                                 signal_type=s["signal_type"]
                                                 )

            # init_input = np.tile(init_input, (1, len(self.control_links) + len(self.control_nodes)))
            sys, u, y = run_comparable_signal(sys, init_input, noise_std=self.noise_std)
            mae, me = self.get_error(y=sys.target_values[-self.experiment_horizon:, :])
            print(f"{s['name']} --> MAE: {mae:.6f} | ME: {me:.3f}")

            t = len(sys.target_values[1:, 0])
            if "plot" in s and s["plot"]:
                for i, element in enumerate(sys.target_nodes):
                    axes[0].plot(sys.target_values[1:, i], label=s["name"], zorder=2)
                    if self.plot_legend_cols > 0:
                        axes[0].legend(ncols=self.plot_legend_cols, fontsize=10)

                for i, element in enumerate(sys.control_nodes + sys.control_links):
                    axes[1].step(range(t), sys.implemented[1:, i], label=s["name"], zorder=2, where='post')
                    if self.plot_legend_cols > 0:
                        axes[1].legend(ncols=self.plot_legend_cols, fontsize=10)

        return fig


def run_comparable_signal(sys, ref_input_signal, noise_std):
    ref_sys = system.WDSControl(inp_path=sys.inp_path,
                                control_links=sys.control_links,
                                control_nodes=sys.control_nodes,
                                target_nodes=sys.target_nodes,
                                target_param=sys.target_param
                                )

    ref_sys.apply_input(ref_input_signal, noise_std=noise_std)
    u, y = sys.implemented, sys.target_values
    return ref_sys, u, y


def grid_search(export_path, cfg):
    df = pd.DataFrame()
    for lg in [0, 0.01, 0.1, 1, 10, 100, 1000]:
        for ly in [0, 0.01, 0.1, 1, 10, 100, 1000]:
            for lu in [0, 0.01, 0.1, 1, 10, 100, 1000]:
                cfg['lg'], cfg['ly'], cfg['lu'] = lg, ly, lu
                mae = run(cfg, plot=False)

                # mae = func(lg=lg, ly=ly, lu=lu, plot=False)
                df = pd.concat([df, pd.DataFrame({"lg": lg, "ly": ly, "lu": lu, "mae": mae}, index=[len(df)])])
                print(df)
                df.to_csv(export_path)


def run(cfg, plot=True, export_inp=''):
    e = Experiment(inp_path=cfg["inp_path"],
                   control_links=cfg["control_links"],
                   control_nodes=cfg["control_nodes"],
                   target_nodes=cfg["target_nodes"],
                   target_param=cfg["target_param"],
                   n_train=cfg["n_train"],
                   input_signal=cfg["input_signal"],
                   input_loss=cfg["input_loss"],
                   y_ref=cfg["y_ref"],
                   y_lb=cfg["y_lb"], y_ub=cfg["y_ub"],
                   u_lb=cfg["u_lb"], u_ub=cfg["u_ub"],
                   wait=cfg["wait"],
                   t_ini=cfg["t_ini"],
                   horizon=cfg["horizon"],
                   lg=cfg["lg"],
                   ly=cfg["ly"],
                   lu=cfg["lu"],
                   experiment_horizon=cfg["experiment_horizon"],
                   noise_std=cfg["noise_std"],
                   compare_signals=cfg["compare_signals"],
                   input_y_label=cfg["input_y_label"],
                   output_y_label=cfg["output_y_label"],
                   x_label=cfg["x_label"],
                   plot_demand_pattern=cfg["plot_demand_pattern"]
                   )

    e.run_experiment()
    if export_inp:
        e.wds.export_inp_with_patterns(export_path=export_inp)

    if plot:
        fig = e.plot(e.wds, label="Node ")
        e.plot_comparison_signals(fig, signals=e.compare_signals)
    return e.mae


if __name__ == "__main__":
    global_seed = 0
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)

    # Example 1
    with open("example_fossolo.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        run(cfg)

    # Example 2
    with open("example_pescara.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        run(cfg, export_inp="pescara_output.inp")
    grid_search(export_path="pescara.csv", cfg=cfg)

    plt.show()
