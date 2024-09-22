import copy
import yaml
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import epanet.toolkit as en

import main
import system
import utils

LAMBDAS_PARAM_GRID = {
        'lg': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 10000],
        'lu': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 10000],
        'ly': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 10000],
    }

N_TRAIN_PARAM_GRID = {
        'n_train': [_ * 50 for _ in range(0, 21)]
    }


def grid_search(export_path, cfg, param_grid):
    np.random.seed(42)
    df = pd.DataFrame()

    keys = param_grid.keys()
    values = param_grid.values()
    for i, combination in enumerate(itertools.product(*values)):
        iter_params = dict(zip(keys, combination))

        for param_name, param_value in iter_params.items():
            cfg[param_name] = param_value

        try:
            experiment = main.run(cfg, plot=False)
            iter_params['cost'] = experiment.cost
            iter_params['mae'] = experiment.mae
            iter_params['v_count'] = experiment.v_count
            iter_params['v_rate'] = experiment.v_rate
            temp = pd.DataFrame(iter_params, index=[len(df)])
            df = pd.concat([df, temp])
            print(df)
            df.to_csv(export_path)
        except Exception as e:
            print(e)


def random_search(export_path, cfg, n):
    df = pd.DataFrame()
    lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
    t_inis = [6, 12, 24, 36, 48]
    waits = [6, 8, 10]
    horizons = [12, 18, 24]

    for i in range(n):
        lg = lambdas[np.random.randint(low=0, high=len(lambdas))]
        ly = lambdas[np.random.randint(low=0, high=len(lambdas))]
        lu = lambdas[np.random.randint(low=0, high=len(lambdas))]
        tini = t_inis[np.random.randint(low=0, high=len(t_inis))]
        wait = t_inis[np.random.randint(low=0, high=len(waits))]
        horizon = t_inis[np.random.randint(low=0, high=len(horizons))]

        cfg['lg'], cfg['ly'], cfg['lu'], cfg['t_ini'], cfg['wait'], cfg['horizon'] = lg, ly, lu, tini, wait, horizon
        try:
            mae = main.run(cfg, plot=False)
            temp = pd.DataFrame({"lg": lg, "ly": ly, "lu": lu, "t_ini": tini, "wait": wait, "horizon": horizon,
                                 "mae": mae}, index=[len(df)])
            df = pd.concat([df, temp])
            print(df)
            df.to_csv(export_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    with open("Experiments/example_fossolo.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        grid_search(export_path="Output/fossolo-lambdas-search.csv", cfg=cfg, param_grid=LAMBDAS_PARAM_GRID)
        grid_search(export_path="Output/fossolo-n_train-search.csv", cfg=cfg, param_grid=N_TRAIN_PARAM_GRID)

    with open("Experiments/example_pescara.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        grid_search(export_path="Output/pescara-lambdas-search.csv", cfg=cfg, param_grid=LAMBDAS_PARAM_GRID)
        grid_search(export_path="Output/pescara-n_train-search.csv", cfg=cfg, param_grid=N_TRAIN_PARAM_GRID)

    plt.show()