import copy
import itertools

import epanet.toolkit as en

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from pymoo.core.callback import Callback
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
# from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination

import main
import system
import utils


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))


class OptiCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())


class OptiChlorine(ElementwiseProblem):
    def __init__(self, sys, duration, n_gen: int, pop_size: int, x_lb, x_ub, noise_std):
        self.sys = sys
        self.duration = duration
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.noise_std = noise_std

        self.inp_path = sys.inp_path
        self.control_links = sys.control_links
        self.control_nodes = sys.control_nodes
        self.target_nodes = sys.target_nodes
        self.target_param = sys.target_param

        self.n_vars = (len(self.control_nodes) + len(self.control_links)) * duration

        self.algorithm = self.set_opti_algorithm()
        n_obj = 1
        n_constr = 0

        self.x_lb = x_lb
        self.x_ub = x_ub

        super().__init__(n_var=self.n_vars, n_obj=n_obj, n_constr=n_constr, xl=x_lb, xu=x_ub)

    def set_opti_algorithm(self):
        return GA(pop_size=self.pop_size)

    def run(self):
        termination = get_termination("n_gen", self.n_gen)
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-4,
            cvtol=1e-4,
            ftol=0.0025,
            period=30,
            n_max_gen=self.n_gen,
        )
        res = minimize(self, self.algorithm, termination, seed=1, save_history=True, verbose=True,
                       callback=OptiCallback())
        return res

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.reshape(-1, 1)

        sys = copy.deepcopy(self.sys)
        x = utils.extend_array_to_n_values(x, self.duration * 2)
        sys.apply_input(x, noise_std=self.noise_std)
        u, y = sys.implemented, sys.target_values

        constraint_violations = np.count_nonzero(y[-self.duration:] < 0.2)
        out["F"] = u[-self.duration:].sum() + constraint_violations * 10


def grid_search(export_path, cfg, param_grid):
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
    # with open("Experiments/example_fossolo.yaml") as f:
    #     cfg = yaml.load(f, Loader=yaml.SafeLoader)
    #     grid_search(export_path="Output/Fossolo-grid-search.csv", cfg=cfg)

    pescara_param_grid = {
        'lg': [0.001, 0.01, 0.1, 1, 10, 100],
        'lu': [0.001, 0.01, 0.1, 1, 10, 100],
        'ly': [0.001, 0.01, 0.1, 1, 10, 100],
        't_ini': [6, 12, 24, 36, 48],
        'wait': [6, 8, 10, 12],
    }

    pescara_lambdas_param_grid = {
        'lg': [0, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
        'lu': [0, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
        'ly': [0, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
    }

    with open("Experiments/example_pescara.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        grid_search(export_path="Output/pescara-lambdas-search.csv", cfg=cfg, param_grid=pescara_lambdas_param_grid)

    plt.show()