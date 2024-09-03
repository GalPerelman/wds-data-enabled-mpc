import sys
import numpy as np
import epanet.toolkit as en
import math

from matplotlib import pyplot as plt, ticker
from pydeepc import Data

import utils


def make_array(values):
    dbl_arr = en.doubleArray(len(values))
    for i in range(len(values)):
        dbl_arr[i] = values[i]
    return dbl_arr


class WDSControl:
    """
    x - state variable, water age and tanks levels
    u - system input, valves settings
    y - system output, water age or water pressure

    current code limitations:
    - control links can only be assigned with epanet en.SETTING parameter which govern valves opening
    - control nodes can only be assigned with epanet en.SOURCEQUAL parameter which govern the quality of source
      (mainly for optimizing chlorine injection)
    - target parameter can be nodes pressure (en.PRESSURE), nodes water age (en.AGE) or nodes chemical (en.CHEM)

    """

    def __init__(self, inp_path: str, control_links: list, control_nodes: list, target_nodes: list, target_param: int):

        self.inp_path = inp_path
        self.control_links = control_links
        self.control_nodes = control_nodes
        self.target_nodes = target_nodes
        self.target_param = target_param

        self.control_elements = self.control_links + self.control_nodes

        self.implemented = np.empty(shape=(1, len(self.control_links) + len(self.control_nodes)))
        self.target_values = np.empty(shape=(1, len(self.target_nodes)))
        self.source_supply = np.empty(shape=(1, len(self.control_nodes)))

        self.report_step = 3600

    def apply_input(self, u: np.ndarray, noise_std: float):
        """
        Applies an input signal to the system - implement operational policy
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and M is the number of features
        :return: tuple that contains the (input, output) of the system
        """
        # in every step simulate all the implemented steps to restore current state + horizon steps
        simulation_duration = self.implemented.shape[0] + u.shape[0]
        input_from_zero = np.vstack([self.implemented, u])

        ph = en.createproject()
        err = en.open(ph, self.inp_path, "net.rpt", "net.out")
        err = en.settimeparam(ph, en.REPORTSTEP, self.report_step)
        control_links_idx = [en.getlinkindex(ph, _) for _ in self.control_links]
        control_nodes_idx = [en.getnodeindex(ph, _) for _ in self.control_nodes]
        target_nodes_idx = [en.getnodeindex(ph, _) for _ in self.target_nodes]

        en.settimeparam(ph, en.DURATION, 3600 * simulation_duration)
        en.setqualtype(ph, qualType=en.CHEM, chemName='', chemUnits='', traceNode='')

        en.openH(ph)
        en.initH(ph, en.SAVE_AND_INIT)
        en.openQ(ph)
        en.initQ(ph, saveFlag=0)
        t_step = 1
        t = 3600
        y = np.empty(shape=(simulation_duration+1, len(self.target_nodes)))
        s = np.empty(shape=(simulation_duration+1, len(self.control_nodes)))

        while t_step > 0:
            # assign control values for the current time step
            for _, link_idx in enumerate(control_links_idx):
                en.setlinkvalue(ph, index=link_idx, property=en.SETTING, value=input_from_zero[int(t / 3600), _])

            for _, node_idx in enumerate(control_nodes_idx):
                uu = max(0, input_from_zero[int(t / 3600), len(self.control_links) + _])
                en.setnodevalue(ph, index=node_idx, property=en.SOURCEQUAL, value=uu)

            # run hydraulic and quality simulations of the current time step
            # in case of real system this step is not needed
            # after applying the control values we wait for the measured values and save them as recorded
            t = en.runH(ph)
            t = en.runQ(ph)

            # record values of the target elements
            if t % self.report_step == 0:
                for _, target_idx in enumerate(target_nodes_idx):
                    target_value = en.getnodevalue(ph, target_idx, self.target_param)
                    y[math.floor(t/3600), _] = target_value * (1 + noise_std * np.random.rand())

                for _, node_idx in enumerate(control_nodes_idx):
                    supply = en.getnodevalue(ph, node_idx, en.DEMAND)
                    s[math.floor(t / 3600), _] = supply

            # move forward
            t_step = en.nextH(ph)
            t_step = en.nextQ(ph)

        y = y[-len(u):]
        s = s[-len(u):]
        self.implemented = np.vstack([self.implemented, u])
        self.target_values = np.vstack([self.target_values, y])
        self.source_supply = np.vstack([self.source_supply, s])
        en.closeH(ph)
        en.closeQ(ph)
        en.close(ph)

        return Data(u, y)

    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        return Data(self.implemented[-n:, :], self.target_values[-n:, :])

    def get_init_input_random(self, mu, sigma, n):
        u = np.random.normal(loc=mu, scale=sigma, size=(n, len(self.control_links) + len(self.control_nodes)))
        return u

    def get_demand_pattern(self, pat_idx):
        ph = en.createproject()
        err = en.open(ph, self.inp_path, "net.rpt", "net.out")

        n = en.getpatternlen(ph, pat_idx)
        return [en.getpatternvalue(ph, pat_idx, period) for period in range(1, n + 1)]

    def get_chlorine_injection_all_sources(self):
        dot_products = np.sum(self.implemented * self.source_supply, axis=1)
        row_sums = np.sum(self.source_supply, axis=1)

        # Calculate the weighted average by dividing the dot products by the row sums of arr1
        # Using np.where to avoid division by zero, replacing zero sums with 1 to avoid NaNs.
        weighted_averages = dot_products / np.where(row_sums != 0, row_sums, 1)

        return weighted_averages

    def plot_io(self, x_label, output_label, input_label, shade_train=None, plot_avg_chlorine=False, fig=None,
                secondary=None, y_ref=None, y_lb=None, y_ub=None):

        ls = [":", "--", (0, (5, 5)), (0, (3, 5, 1, 5))]

        if fig is None:
            fig, axes = plt.subplots(nrows=3, sharex=True, height_ratios=[2, 2, 1])
        else:
            axes = fig.axes

        def to_array(signal):
            if isinstance(signal, (int, float)):
                return np.tile(signal, len(self.target_values[1:, 0]))
            else:
                return signal

        y_ref, y_ub, y_lb = to_array(y_ref), to_array(y_ub), to_array(y_lb)

        for i, element in enumerate(self.target_nodes):
            # axes[0].plot(y_ref, label="Ref", zorder=1, c="k")
            # axes[0].plot(y_ub, label="Constraint", zorder=1, c="k", linestyle="--")
            # axes[0].plot(y_lb, label="Constraint", zorder=1, c="k", linestyle="--")
            axes[0].plot(self.target_values[1:, i], label="DeePC", zorder=5)
            if secondary is not None:
                for r, _ in enumerate(secondary):
                    axes[0].plot(_["y"][1:, i], c=_["color"], label=_["label"], zorder=3, linewidth=1.2)

        for i, element in enumerate(self.control_nodes + self.control_links):
            axes[1].plot(self.implemented[1:, i], label="DeePC", zorder=5)
            if secondary is not None:
                for r, _ in enumerate(secondary):
                    axes[1].plot(_["u"][1:, i], c=_["color"], label=_["label"], zorder=3, linewidth=1.2)

        pat = self.get_demand_pattern(pat_idx=1)
        pat = utils.extend_array_to_n_values(pat, self.implemented[1:, 0].shape[0])
        axes[2].bar(range(len(pat)), pat, zorder=5, width=1.0)

        if plot_avg_chlorine:
            axes[1].plot(self.get_chlorine_injection_all_sources()[1:], 'k', zorder=5)
            axes[2].plot(self.get_chlorine_injection_all_sources()[1:], 'k', zorder=5)

        secax = axes[0].secondary_xaxis('top')
        secax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        secax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x) % 24}'))
        secax.set_xlabel('Hour of the Day')  # Label for clarity

        for _ in [0, 1]:
            axes[_].grid(zorder=0)

        axes[0].legend(fontsize=9)
        axes[1].legend(fontsize=9)

        axes[0].set_ylabel(output_label)
        axes[1].set_ylabel(input_label)
        axes[2].set_ylabel("Demand Pattern")
        axes[-1].set_xlabel(x_label)

        if shade_train is not None:
            axes[0].axvspan(0, shade_train, facecolor='grey', alpha=0.2, zorder=0)
            axes[1].axvspan(0, shade_train, facecolor='grey', alpha=0.2, zorder=0)
            axes[2].axvspan(0, shade_train, facecolor='grey', alpha=0.2, zorder=0)

        return fig

    def export_inp_with_patterns(self, export_path):
        ph = en.createproject()
        err = en.open(ph, self.inp_path, "net.rpt", "net.out")
        n = en.getcount(ph, en.PATCOUNT)

        for i, element in enumerate(self.control_links):
            en.addpattern(ph, f"_{element}")
            pat_idx = en.getpatternindex(ph, f"_{element}")
            pat = make_array(self.implemented[:, i])
            en.setpattern(ph, pat_idx, pat, len(self.implemented[:, i]))

        for i, element in enumerate(self.control_nodes):
            en.addpattern(ph, f"_{element}")
            pat_idx = en.getpatternindex(ph, f"_{element}")
            pat = make_array(self.implemented[:, i])
            en.setpattern(ph, pat_idx, pat, len(self.implemented[:, len(self.control_links) + i]))
            node_idx = en.getnodeindex(ph, element)
            en.setnodevalue(ph, node_idx, en.SOURCEPAT, pat_idx)

        en.settimeparam(ph, param=en.DURATION, value=len(self.implemented[:, 0]) * 3600)
        en.saveinpfile(ph, export_path)


if __name__ == "__main__":
    system = WDSControl(inp_path="Data/Fossolo.inp", control_links=['59'], control_nodes=[], target_nodes=['21'],
                        target_param=en.PRESSURE)
    data = system.apply_input(u=np.random.normal(size=48).reshape((48, 1)))
