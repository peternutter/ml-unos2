import numpy as np
from pysurvival.models.semi_parametric import CoxPHModel, NonLinearCoxPHModel
from pysurvival.models.survival_forest import (
    RandomSurvivalForestModel,
    ConditionalSurvivalForestModel,
)
from pysurvival.models.multi_task import NeuralMultiTaskModel
from pysurvival.models.parametric import (
    WeibullModel,
    ExponentialModel,
    GompertzModel,
)

class ParamFactory:
    """
    ParamFactory is a class to generate the parameters and model for different types of survival models.
    The class takes the model type as a string and a boolean indicating whether it's a grid search or not.

    Usage:
    ```
    factory = ParamFactory(model="coxPH", is_grid=True)
    model, params = factory.get_params()
    ```

    Available models are:
    'coxPH', 'non_linear_cox', 'MTLR', 'rsf', 'exponential', 'weibull', 'gompertz'
    """
    def __init__(self, model: str, is_grid: bool = False):
        self.model = model
        self.is_grid = is_grid

    def get_params(self):
        if self.model == 'coxPH':
            return self.get_coxPH_params()
        elif self.model == 'non_linear_cox':
            return self.get_non_linear_cox_params()
        elif self.model == 'MTLR':
            return self.get_MTLR_params()
        elif self.model == 'rsf':
            return self.get_rsf_params()
        elif self.model == 'exponential':
            return self.get_exponential_params()
        elif self.model == 'weibull':
            return self.get_weibull_params()
        elif self.model == 'gompertz':
            return self.get_gompertz_params()
        else:
            raise ValueError('Invalid model type')

    def get_coxPH_params(self):
        if self.is_grid:
            params = {
                "lr": [1.1, 1.0, 1.3],
                "l2_reg": np.linspace(1e-4, 1e-1, num=10),
                "max_iter": [30, 40],
                "init_method": ["zeros"],
            }
        else:
            params = {
                "lr": [0.1],
                "l2_reg": [0.1],
                "max_iter": [30],
                "init_method": ["zeros"],
            }
        return CoxPHModel, params

    def get_non_linear_cox_params(self):
        if self.is_grid:
            params = {
                "structure": [
                    [{"activation": "ReLU", "num_units": 64}],
                    [
                        {"activation": "ReLU", "num_units": 64},
                        {"activation": "Tanh", "num_units": 64},
                    ],
                    [
                        {"activation": "ReLU", "num_units": 64},
                        {"activation": "ReLU", "num_units": 64},
                        {"activation": "ReLU", "num_units": 64},
                    ],
                ],
                "init_method": ["zero", "glorot_uniform"],
                "num_epochs": [100, 200],
                "lr": [1e-4, 1e-3, 1e-2],
                "l2_reg": [1e-3],
                "verbose": [True],
            }
        else:
            params = {
                "structure": [
                    [{"activation": 'ReLU', 'num_units': 64}, {'activation': 'Tanh', 'num_units': 64}],
                ],
                "init_method": ["glorot_uniform"],
                "lr": [0.01],
                "l2_reg": [0.001],
                "verbose": [True],
                "num_epochs": [100],
            }
        return NonLinearCoxPHModel, params

    def get_MTLR_params(self):
        if self.is_grid:
            params = {
                "structure": [
                    [{"activation": "ReLU", "num_units": 256}],
                ],
                "bins": [100],
                "init_method": [
                    "zeros",
                    "glorot_uniform",
                    "glorot_normal",
                    "he_uniform",
                    "he_normal",
                    "ones",
                ],
                "optimizer": ["adam", "sgd", "rmsprop", "adagrad", "adadelta"],
                "lr": [1e-3, 1e-2, 1e-4, 1e-5, 1e-6],
                "num_epochs": [500],
                "l2_reg": [1e-2, 1e-3, 1e-4],
            }
        else:
            params = {
                "bins": [100],
                "structure": [
                    [
                        {"activation": "ReLU", "num_units": 256},
                        {"activation": "ReLU", "num_units": 128},
                        {"activation": "ReLU", "num_units": 64},
                    ]
                ],
                "lr": [1e-4],
                "init_method": ["zeros"],
            }
        return NeuralMultiTaskModel, params

    def get_rsf_params(self):
        if self.is_grid:
            params = {
                "num_trees": [100],
                "max_features": ["sqrt", "log2"],
                "max_depth": [10, 15],
                "min_node_size": [10, 20],
                "sample_size_pct": [0.8],
                "num_threads": [16],
                "seed": [42],
            }
        else:
            params = {
                "num_trees": [100],
                "max_features": ["sqrt"],
                "max_depth": [15],
                "min_node_size": [40],
                "sample_size_pct": [0.8],
            }
        return RandomSurvivalForestModel, params
    

    def get_exponential_params(self):
        if self.is_grid:
            params = {
                "bins": [100],
                "init_method": ["zeros", "glorot_uniform"],
                "lr": np.linspace(0.0001, 0.001, 10),
                "num_epochs": [2100, 2200],
                "l2_reg": np.linspace(0.06, 0.9, 3),
            }
        else:
            params = {
                "bins": [100],
                "init_method": ["glorot_uniform"],
                "lr": [0.0005],
                "num_epochs": [2150],
                "l2_reg": [0.48],
            }
        return ExponentialModel, params

    def get_weibull_params(self):
        if self.is_grid:
            params = {
                "bins": [100],
                "init_method": ["zeros", "glorot_uniform"],
                "lr": np.linspace(0.0001, 0.001, 10),
                "num_epochs": np.linspace(2000, 3000, 10, dtype=int),
                "l2_reg": np.logspace(-3, 2, 5),
            }
        else:
            params = {
                "bins": [100],
                "init_method": ["glorot_uniform"],
                "lr": [0.0005],
                "num_epochs": [2500],
                "l2_reg": [0.1],
            }
        return WeibullModel, params

    def get_gompertz_params(self):
        if self.is_grid:
            params = {
                "bins": [100],
                "init_method": ["zeros", "glorot_uniform"],
                "lr": np.linspace(0.0001, 0.001, 10),
                "num_epochs": np.linspace(2000, 5000, 10, dtype=int),
                "l2_reg": np.logspace(-3, 2, 5),
            }
        else:
            params = {
                "bins": [100],
                "init_method": ["glorot_uniform"],
                "lr": [0.0005],
                "num_epochs": [3500],
                "l2_reg": [0.1],
            }
        return GompertzModel, params
