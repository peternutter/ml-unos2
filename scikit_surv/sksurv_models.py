import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis, IPCRidge
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
)
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.metrics import as_cumulative_dynamic_auc_scorer, as_concordance_index_ipcw_scorer, as_integrated_brier_score_scorer
from utils.utils import calculate_tau

class SurvivalModelFactory:

    @staticmethod
    def get_gradient_boosting_model():
        return GradientBoostingSurvivalAnalysis(
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            verbose=3,
            max_depth=10,
            max_features="sqrt",
            dropout_rate=0.2,
            subsample=0.8,
            min_samples_leaf=10,
        )

    @staticmethod
    def get_random_forest_model(n_jobs=8):
        return RandomSurvivalForest(
            n_estimators=400,
            max_depth=12,
            min_samples_split=15,
            max_features="sqrt",
            random_state=42,
            verbose=10,
            n_jobs=n_jobs,
        )

    @staticmethod
    def get_extra_trees_model(n_jobs=8):
        return ExtraSurvivalTrees(
            n_estimators=800,
            max_depth=15,
            min_samples_split=15,
            min_samples_leaf=20,
            max_features="sqrt",
            max_leaf_nodes=200,
            random_state=42,
            verbose=10,
            n_jobs=n_jobs,
        )

    @staticmethod
    def get_componentwise_gradient_boosting_model():
        return ComponentwiseGradientBoostingSurvivalAnalysis(
            learning_rate=0.5,
            n_estimators=150,
            random_state=42,
            verbose=100,
            subsample=0.8,
            dropout_rate=0.2,
            loss="coxph",
        )

    @staticmethod
    def get_fast_survival_svm_model():
        return FastKernelSurvivalSVM(
            alpha=2, tol=1e-4, max_iter=1000, random_state=42, verbose=True
        )

    @staticmethod
    def get_coxnet_model():
        return CoxnetSurvivalAnalysis(l1_ratio=0.12, verbose=True, fit_baseline_model=True)

    @staticmethod
    def get_coxph_model():
        return CoxPHSurvivalAnalysis(
            verbose=True,
        )

    @staticmethod
    def get_ipc_ridge_model():
        return IPCRidge(solver="sag", alpha=10)

class ModelParameterGridFactory:

    @staticmethod
    def get_gbsa_param_grid():
        return {
            "loss": ["coxph"],
            "learning_rate": np.linspace(0.1, 0.5, num=5),
            "n_estimators": [50, 100, 150, 200],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [10],
            "max_depth": [10, 15],
            "max_features": ["log2"],
            "subsample": [0.8],
            "dropout_rate": [0.2],
            "random_state": [42],
            "verbose": [3],
        }

    @staticmethod
    def get_rsf_param_grid():
        return {
            "n_estimators": [200],
            "max_depth": [13, 20],
            "min_samples_split": [15, 4],
            "min_samples_leaf": [20, 6],
            "max_leaf_nodes": [170, 250],
            "max_features": ["log2", "auto"],
            "min_weight_fraction_leaf": [0.0],
            "random_state": [42],
        }

    @staticmethod
    def get_rsf_extra_param_grid():
        return {
            "n_estimators": [200],
            "max_depth": [15, 30],
            "min_samples_split": [100],
            "min_samples_leaf": [20, 10],
            "max_leaf_nodes": [200, 150],
            "max_features": ["log2", "auto"],
            "min_weight_fraction_leaf": [0.0],
            "random_state": [42],
        }

    @staticmethod
    def get_cgb_param_grid():
        return {
            "loss": ["coxph"],
            "learning_rate": np.linspace(0.1, 0.5, num=5),
            "n_estimators": [50, 100, 150],
            "dropout_rate": [0.2],
            "subsample": [0.8],
            "verbose": [100],
        }

    @staticmethod
    def get_fs_svm_param_grid():
        return {
            "alpha": np.linspace(1, 3, num=3),
            "rank_ratio": [0.5, 0.8, 1.0],
            "kernel": ["rbf"],
            "max_iter": [20],
            "random_state": [42],
            "verbose": [True],
            "timeit": [True],
        }

    @staticmethod
    def get_coxnet_param_grid():
        alphas = np.logspace(-3, 1, num=10)
        alphas = [[a] for a in alphas]
        return {
            "n_alphas": [150, 100],
            "l1_ratio": np.linspace(0.01, 0.99, num=10),
            "alpha_min_ratio": np.logspace(-6, -3, num=10),
            "fit_baseline_model": [True],
            "tol": [1e-5, 1e-4, 1e-6],
            "max_iter": [1000, 2000],
        }

class ScorerFactory:

    @staticmethod
    def get_as_integrated_brier_scorer(model, times):
        return as_integrated_brier_score_scorer(model, times=times)
    @staticmethod
    def as_concordance_index_ipcw_scorer(model, times):
        return as_concordance_index_ipcw_scorer(model, calculate_tau(times))
    @staticmethod
    def as_cumulative_dynamic_auc_scorer(model, times):
        return as_cumulative_dynamic_auc_scorer(model, times=times)



