import numpy as np
import cvxpy as cp
from typing import List, Union

from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc import DeePC
from pydeepc.utils import Data

import system


class DDPC:
    def __init__(self,
                 sys,  # a system class with apply_input and get_last_n_samples methods
                 input_loss: bool,
                 y_ref: Union[int, float, np.ndarray, list],
                 u_lb=-np.inf,
                 u_ub=np.inf,
                 y_lb=-np.inf,
                 y_ub=np.inf,
                 wait: int = 1,  # Steps between runs
                 t_ini: int = 24,  # Size of the initial set of data
                 horizon: int = 24,  # MPC horizon
                 lambda_g: float = 0.1,  # g regularization param
                 lambda_y: float = 0.01,  # y regularization param
                 lambda_u: float = 0,  # u regularization param
                 experiment_horizon: int = 168,  # Total number of steps
                 solver_verbose: bool = False,
                 noise_std=0
                 ):

        self.sys = sys
        self.input_loss = input_loss
        self.y_ref = y_ref
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.wait = wait
        self.t_ini = t_ini
        self.horizon = horizon
        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.lambda_u = lambda_u
        self.experiment_horizon = experiment_horizon
        self.solver_verbose = solver_verbose
        self.noise_std = noise_std

        self.deepc = None

    def loss_callback(self, u: cp.Variable, y: cp.Variable) -> Expression:
        horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
        if isinstance(self.y_ref, int):
            ref = np.ones(y.shape) * self.y_ref
        else:
            ref = self.y_ref

        return cp.norm(y - ref, 'fro') ** 2 + self.input_loss * cp.norm(u, 'fro') ** 2

    def constraints_callback(self, u: cp.Variable, y: cp.Variable):
        horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
        return [u >= self.u_lb, u <= self.u_ub, y >= self.y_lb, y <= self.y_ub]

    def init_deepc(self, init_input):
        data = self.sys.apply_input(u=init_input, noise_std=0)
        self.deepc = DeePC(data, Tini=self.t_ini, horizon=self.horizon)

        self.deepc.build_problem(
            build_loss=self.loss_callback,
            build_constraints=self.constraints_callback,
            lambda_g=self.lambda_g,
            lambda_y=self.lambda_y,
            lambda_u=self.lambda_u
        )

    def run(self):
        data_ini = Data(
            u=np.zeros((self.t_ini, len(self.sys.control_elements))),
            y=np.zeros((self.t_ini, len(self.sys.target_nodes)))
                        )

        for _ in range(self.experiment_horizon // self.wait):
            # Solve DeePC
            u_optimal, info = self.deepc.solve(data_ini=data_ini, warm_start=True, verbose=self.solver_verbose,
                                               solver=cp.GUROBI)

            # Apply optimal control input
            self.sys.apply_input(u=u_optimal[:self.wait, :], noise_std=self.noise_std)

            # Fetch last T_INI samples
            data_ini = self.sys.get_last_n_samples(self.t_ini)