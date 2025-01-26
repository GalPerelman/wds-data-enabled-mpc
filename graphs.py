import matplotlib.pyplot as plt
import numpy as np


def customize_plot_fossolo(experiment):
    experiment.plot_constraints = False
    experiment.plot_y_legend_label = ["DeePC"]
    experiment.plot_u_legend_label = ["DeePC"]
    for s in experiment.compare_signals:
        s["plot"] = True
    experiment.plot_legend_cols = 5
    fig = experiment.plot(experiment.wds)
    fig, u, y = experiment.plot_comparison_signals(fig, signals=experiment.compare_signals)

    u, y = experiment.run_pid(kp=0.01, kd=0.1, ki=1, T=experiment.experiment_horizon+experiment.n_train)

    _u = u[-experiment.experiment_horizon:, :]
    _y = y[-experiment.experiment_horizon:, :]

    mae, me = experiment.get_error(y=_y)
    ref = np.ones(_y.shape) * experiment.y_ref
    cost = np.linalg.norm(_y - ref, 'fro') ** 2 + experiment.input_loss * np.linalg.norm(_u, 'fro') ** 2
    v_count, v_rate = experiment.get_violations(y=_y)
    print(f"PID --> Cost: {cost:.3f} | MAE: {mae:.3f} | ME: {me:.3f}")
    print(f"PID --> {experiment.y_lb}-{experiment.y_ub} Violations: {v_count:.0f}"
          f" | Violations Rate: {v_rate:.3f}")
    v_count, v_rate = experiment.get_violations(y=_y, lb=20, ub=40)
    print(f"PID --> {20}-{40} Violations: {v_count:.0f} | Violations Rate: {v_rate:.3f}")

    axes = fig.axes
    axes[0].fill_between(range(experiment.n_train, experiment.experiment_horizon+experiment.n_train+2),
                         experiment.y_lb, experiment.y_ub,
                         alpha=0.2, color='grey', label="Constraints")
    axes[0].plot(y, '#35AC78', label="PID", zorder=3, linewidth=1.2)
    axes[1].step(u, '#35AC78', label="PID", zorder=3, where='post', linewidth=1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 4, 0, 2, 3, 5]  # reorder the legend items - after adding PID
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], ncol=6, columnspacing=0.75, handletextpad=0.2)

    # handles, labels = axes[1].get_legend_handles_labels()
    # order = [0, 2, 3, 5]  # reorder the legend items - after adding PID
    # axes[1].legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, columnspacing=0.75, handletextpad=0.15)
    axes[1].legend(ncol=4, columnspacing=0.75, handletextpad=0.15)

    axes[0].set_ylim(8, 42)
    axes[1].set_ylim(30, 66)
    axes[2].set_ylim(0.5, 1.6)
    axes[-1].set_xlim(experiment.n_train, experiment.n_train + experiment.experiment_horizon + 4)
    fig.set_size_inches(8, 6)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.1)


def customize_plot_pescara(experiment):
    experiment.plot_legend_cols = 5
    fig = experiment.plot(experiment.wds)
    fig = experiment.plot_comparison_signals(fig, signals=experiment.compare_signals)
    axes = fig.axes
    axes[0].set_ylim(0.7, 1.4)
    axes[1].set_ylim(0.95, 1.2)
    axes[-1].set_xlim(experiment.n_train, experiment.n_train + experiment.experiment_horizon + 4)
    # fig.set_size_inches(8, 5)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9)
