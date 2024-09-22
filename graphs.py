import matplotlib.pyplot as plt
import numpy as np


def customize_plot_fossolo(experiment):
    experiment.plot_constraints = True
    experiment.plot_y_legend_label = ["DeePC"]
    experiment.plot_u_legend_label = ["DeePC"]
    for s in experiment.compare_signals:
        s["plot"] = True
    experiment.plot_legend_cols = 5
    fig = experiment.plot(experiment.wds)
    fig, u, y = experiment.plot_comparison_signals(fig, signals=experiment.compare_signals)
    axes = fig.axes
    axes[0].set_ylim(6, 44)
    axes[1].set_ylim(26, 65)
    axes[2].set_ylim(0.5, 1.6)
    axes[-1].set_xlim(experiment.n_train, experiment.n_train + experiment.experiment_horizon + 4)
    fig.set_size_inches(8, 6)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9)

    mae, me = experiment.get_error(y=y)
    ref = np.ones(y.shape) * experiment.y_ref
    cost = np.linalg.norm(y - ref, 'fro') ** 2 + experiment.input_loss * np.linalg.norm(u, 'fro') ** 2
    v_count, v_rate = experiment.get_violations(y=y)
    print(f"{s['name']} --> Cost: {cost:.3f} | MAE: {mae:.3f} | ME: {me:.3f}")
    print(f"{s['name']} --> {experiment.y_lb}-{experiment.y_ub} Violations: {v_count:.0f}"
          f" | Violations Rate: {v_rate:.3f}")
    v_count, v_rate = experiment.get_violations(y=y, lb=20, ub=40)
    print(f"{s['name']} --> {20}-{40} Violations: {v_count:.0f} | Violations Rate: {v_rate:.3f}")


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
