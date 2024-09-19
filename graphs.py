import matplotlib.pyplot as plt


def customize_plot_fossolo(experiment):
    experiment.plot_constraints = True
    experiment.plot_y_legend_label = ["DeePC"]
    experiment.plot_u_legend_label = ["DeePC"]
    for s in experiment.compare_signals:
        s["plot"] = True
    experiment.plot_legend_cols = 5
    fig = experiment.plot(experiment.wds)
    fig = experiment.plot_comparison_signals(fig, signals=experiment.compare_signals)
    axes = fig.axes
    axes[0].set_ylim(6, 44)
    axes[1].set_ylim(26, 65)
    axes[2].set_ylim(0.5, 1.6)
    axes[-1].set_xlim(experiment.n_train, experiment.n_train + experiment.experiment_horizon + 4)
    fig.set_size_inches(8, 6)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9)


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
