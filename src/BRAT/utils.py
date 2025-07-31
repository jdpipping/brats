import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
import optuna
import matplotlib.ticker as mticker
import lightgbm as lgb
import xgboost as xgb
import re

from optuna.pruners import MedianPruner
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, LogFormatter
from matplotlib.ticker import MaxNLocator, LogLocator
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from pygam import LinearGAM
from ucimlrepo import fetch_ucirepo

from BRAT.algorithms import BRATD, BRATP


# data generation
def generate_data(function_type='friedman1', n_train=5000, n_test=1000, 
                  n_calibration = None, noise_std=1, seed=2):
    np.random.seed(seed)
    
    if function_type == 'friedman1':
        f = lambda X: 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 0 * X[:, 3] + 5 * X[:, 4]
    elif function_type == 'friedman2':
        f = lambda X: np.sqrt(X[:, 0]**2 + (X[:, 1]*X[:, 2] - 1/(X[:, 1]*X[:, 3] + 1e-6))**2)
    elif function_type == 'radial':
        f = lambda X: np.exp(-np.sum((X - 0.5)**2, axis=1))
    elif function_type == 'smooth_linear':
        f = lambda x: 3 * x[:,0] + 2 * x[:, 1] - x[:, 2] + 0.5 * np.sin(2 * np.pi * x[:, 3]) + 0.3 * x[:, 4]**2
    elif function_type == 'linear':
        f = lambda X: 2 * X[:, 0] - 3 * X[:, 1] + 1.5 * X[:, 2]
    elif function_type == 'constant':
        f = lambda X: np.full(X.shape[0], 5)
    elif function_type == 'stepwise':
        f = lambda X: np.where(X[:, 0] > 0.5, 10, -10) 
    elif function_type == 'sigmoid':
        f = lambda X: 1 / (1 + np.exp(-X[:, 0]))
    elif function_type == 'mild_sine':
        f = lambda X: np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 0]**2
    elif function_type == 'sigmoid_friedman':
        f = lambda X: 1 / (1 + np.exp(-(10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 0 * X[:, 3] + 5 * X[:, 4] - 10) / 3))
    else:
        raise ValueError("Unknown function type. Choose 'friedman1', 'friedman2', 'radial', 'smooth_linear', 'linear', 'constant', 'stepwise', 'sigmoid', or 'mild_sine'.")
    
    X_train = np.random.normal(0, 0.5, size=(n_train, 7))
    X_test = np.random.normal(0, 0.5, (n_test, 7))
    y_train = f(X_train) + np.random.normal(0, noise_std, size=(n_train,))
    y_test_true = f(X_test)
    y_test = f(X_test) + np.random.normal(0, noise_std, size=(n_test,))
    if n_calibration is not None:
        X_cal = np.random.normal(0, 0.5, (n_calibration, 7))
        y_cal = f(X_cal) + np.random.normal(0, noise_std, size=(n_calibration,))
        return X_train, y_train, X_test, y_test, y_test_true, X_cal, y_cal
    else:
        return X_train, y_train, X_test, y_test, y_test_true

# mse calculation
def calculate_mse(estimator, X, y, X_test, y_test):
    estimator.fit(X, y)
    y_pred = estimator.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2).item()
    return mse


############## EMPIRICAL MODULES ##################################

def load_and_clean_uci_data(dataset_id, target_column=None, test_size=0.2, random_state=42,
                             normalize=True, sanity_check=False):
    """
    Fetch, clean, and split UCI dataset with robust target handling.

    Parameters:
    - dataset_id: UCI repository dataset ID.
    - target_column: Name of the target column (if multiple targets exist or none are provided).
    - test_size: Proportion of test set.
    - random_state: Seed for reproducibility.
    - normalize: Whether to normalize features.
    - sanity_check: If True, print dataset shapes and checks.

    Returns:
    - X_train, X_test, y_train_arr, y_test_arr
    """
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    targets = dataset.data.targets

    # Handle missing targets
    if targets is None or (isinstance(targets, pd.DataFrame) and targets.empty):
        if target_column is None:
            raise ValueError(f"No prespecified targets found! Please choose a target from features. "
                             f"Available features: {list(X.columns)}")
        if target_column not in X.columns:
            raise ValueError(f"Specified target_column '{target_column}' not found in features. "
                             f"Available features: {list(X.columns)}")
        y = X[target_column]
        X = X.drop(columns=[target_column])
    else:
        # Handle provided targets
        if isinstance(targets, pd.DataFrame) and target_column is None:
            if targets.shape[1] > 1:
                raise ValueError(
                    f"Multiple targets found. Please specify one using `target_column`. "
                    f"Available targets: {list(targets.columns)}"
                )
            else:
                y = targets.iloc[:, 0]  # Single target column
        elif target_column is not None:
            if target_column in targets.columns:
                y = targets[target_column]
            else:
                raise ValueError(f"Specified target_column '{target_column}' not found. "
                                 f"Available targets: {list(targets.columns)}")
        else:
            y = targets if isinstance(targets, pd.Series) else targets.iloc[:, 0]

    # Drop missing values and align
    X_clean = X.dropna()
    X_clean = pd.get_dummies(X_clean)
    y_clean = y.loc[X_clean.index]

    if sanity_check:
        print(f"Initial data points: {X.shape[0]}")
        print(f"Remaining after cleaning: {X_clean.shape[0]}")
        print(f"Target type: {type(targets)}")
        print(f"Target selected: {y.name if hasattr(y, 'name') else 'N/A'}")
        assert len(y_clean) == X_clean.shape[0], "Mismatch in number of targets and cleaned features!"

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values.astype(np.float32)) if normalize else X_train.values.astype(np.float32)
    X_test = scaler.transform(X_test.values.astype(np.float32)) if normalize else X_test.values.astype(np.float32)

    # Encode target if it's categorical
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    y_train_arr = y_train.astype(np.float32).ravel()
    y_test_arr = y_test.astype(np.float32).ravel()

    if sanity_check:
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train_arr.shape)
        print("y_test shape:", y_test_arr.shape)

    return X_train, X_test, y_train_arr, y_test_arr

def get_objectives(model_name, default_epoch, X_train, y_train_arr):
    subsample_size = int(0.1 * len(X_train))
    idx = np.random.choice(len(X_train), size=subsample_size, replace=False)
    X_train = X_train[idx]
    y_train_arr = y_train_arr[idx]
    if model_name == 'GBT':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'n_estimators': default_epoch,
            }
            model = GradientBoostingRegressor(**params, random_state=42)
            mse = mean_squared_error(y_train_arr, model.fit(X_train, y_train_arr).predict(X_train))
            return mse
        return objective

    elif model_name == 'XGBoost':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'n_estimators': default_epoch,
            }
            model = xgb.XGBRegressor(**params, random_state=42)
            mse = mean_squared_error(y_train_arr, model.fit(X_train, y_train_arr).predict(X_train))
            return mse
        return objective

    elif model_name == 'LightGBM':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'n_estimators': default_epoch,
            }
            lgb_model = lgb.LGBMRegressor(**params, random_state=42)
            X_train_df = pd.DataFrame(X_train)  # Fix feature names warning
            preds = lgb_model.fit(X_train_df, y_train_arr).predict(X_train_df)
            mse = mean_squared_error(y_train_arr, preds)
            return mse
        return objective

    elif model_name == 'RF':
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'n_estimators': default_epoch,
            }
            model = RandomForestRegressor(**params, random_state=42)
            mse = mean_squared_error(y_train_arr, model.fit(X_train, y_train_arr).predict(X_train))
            return mse
        return objective

    elif model_name == 'ElasticNet':
        def objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            }
            model = ElasticNet(**params)
            mse = mean_squared_error(y_train_arr, model.fit(X_train, y_train_arr).predict(X_train))
            return mse
        return objective

    elif model_name == 'GAM':
        def objective(trial):
            params = {
                'lam': trial.suggest_float('lam', 0.01, 10.0, log=True),
                'n_splines': trial.suggest_int('n_splines', 5, 25),
            }
            model = LinearGAM(lam=params['lam'], n_splines=params['n_splines'])
            mse = mean_squared_error(y_train_arr, model.fit(X_train, y_train_arr).predict(X_train))
            return mse
        return objective

    elif model_name == 'BRATD':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'subsample_rate': trial.suggest_float('subsample_rate', 0.5, 1.0),
                'n_estimators': default_epoch,
            }
            model = BRATD(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                        max_depth=params['max_depth'], dropout_rate=params['dropout_rate'],
                        min_samples_split=2, subsample_rate=params['subsample_rate'])
            model.fit(X_train, y_train_arr, X_train, y_train_arr)
            mse = mean_squared_error(y_train_arr, model.predict(X_train))
            return mse
        return objective

    elif model_name == 'Boulevard':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample_rate': trial.suggest_float('subsample_rate', 0.5, 1.0),
                'n_estimators': default_epoch,
            }
            model = BRATD(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                        max_depth=params['max_depth'], dropout_rate=0.0,
                        min_samples_split=2, subsample_rate=params['subsample_rate'])
            model.fit(X_train, y_train_arr, X_train, y_train_arr)
            mse = mean_squared_error(y_train_arr, model.predict(X_train))
            return mse
        return objective

    elif model_name == 'BRATP':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'n_trees_per_group': trial.suggest_int('n_trees_per_group', 5, 30),
                'subsample_rate': trial.suggest_float('subsample_rate', 0.5, 1.0),
                'n_estimators': default_epoch,
            }
            model = BRATP(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                         max_depth=params['max_depth'], n_trees_per_group=params['n_trees_per_group'],
                         min_samples_split=2, subsample_rate=params['subsample_rate'])
            model.fit(X_train, y_train_arr, X_train, y_train_arr)
            mse = mean_squared_error(y_train_arr, model.predict(X_train))
            return mse
        return objective

    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

def tune_all_models(models_to_tune, X_train, y_train_arr, epoch, n_trials=20, manual_configs=None):
    """
    Tune models using Optuna unless manual configurations are provided.
    """
    best_params_dict = {}

    if manual_configs:
        print("Using manual configurations...")
        estimators_set = set()
        for config in manual_configs.values():
            if 'n_estimators' in config:
                estimators_set.add(config['n_estimators'])

        if len(estimators_set) > 1:
            raise ValueError(f"All models must have the same 'n_estimators'. Found: {estimators_set}")

        for model_name in tqdm(models_to_tune, desc="Tuning Models"):
            if model_name in manual_configs:
                print(f"Using manual hyperparameters for {model_name}.")
                best_params_dict[model_name] = manual_configs[model_name]
            else:
                print(f"Tuning {model_name} automatically...")
                objective = get_objectives(model_name, X_train, y_train_arr)
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

                best_params = study.best_params
                if model_name in ['GBT', 'XGBoost', 'LightGBM', 'RF', 'BRATD', 'Boulevard', 'BRATP']:
                    best_params['n_estimators'] = next(iter(estimators_set))
                best_params_dict[model_name] = best_params
    else:
        print(f"No manual configs provided. Setting n_estimators={epoch} for all models.")
        for model_name in tqdm(models_to_tune, desc="Tuning Models"):
            print(f"Tuning {model_name} automatically...")
            objective = get_objectives(model_name, epoch, X_train, y_train_arr)
            study = optuna.create_study(direction='minimize', pruner=MedianPruner())
            study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
            
            best_params = study.best_params
            if model_name in ['GBT', 'XGBoost', 'LightGBM', 'RF', 'BRATD', 'Boulevard', 'BRATP']:
                best_params['n_estimators'] = epoch
            best_params_dict[model_name] = best_params

    return best_params_dict


def train_all_models(X_train, y_train_arr, X_test, y_test_arr, epoch, 
                     tune=True, models=None, manual_configs=None, 
                     n_trials=20, verbose_tqdm=True, run_idx=None):
    """
    Train models with optional hyperparameter tuning.

    Parameters:
    - X_train, y_train_arr, X_test, y_test_arr: Datasets.
    - tune: Whether to auto-tune models.
    - models: List of models to tune/train.
    - manual_configs: Dict of manual hyperparameters {model_name: params}.
    - n_trials: Number of trials for tuning (Optuna).

    Returns:
    - mse_dict: Model name → staged MSEs or single MSE.
    - best_params_dict: Model name → hyperparameters used.
    """

    # Combine models_to_tune and manual_configs to ensure all models are included
    if manual_configs is None:
        manual_configs = {}
    
    if manual_configs:
        models = list(set(models + list(manual_configs.keys())))
    best_params_dict = {}

    # Hyperparameter Tuning Phase for auto-tuned models
    if tune:
        best_params_dict = tune_all_models(models, X_train, y_train_arr, 
                                           epoch, n_trials, manual_configs)
    else:
        best_params_dict = manual_configs or {}

    mse_dict = {}

    if verbose_tqdm:
        iterator = tqdm(models, desc=f"Run {run_idx or 1}: Training Models")
    else:
        iterator = models
    
    base_seed = 13
    seed = base_seed + 1000 * (run_idx if run_idx is not None else 0)
    
    # Training Phase
    for model_name in iterator:
        params = best_params_dict.get(model_name)

        # Check if the model is in manual_configs (i.e., it has pre-set parameters)
        if model_name in manual_configs:
            print(f"Training {model_name} with manual configuration...")
            params = manual_configs[model_name]  # Use manual configuration if provided

        if model_name == 'GBT':
            print(f"Training {model_name}...")
            params = {**params}
            params['random_state'] = seed
            gbt = GradientBoostingRegressor(**params, min_samples_split=2)
            gbt.fit(X_train, y_train_arr)
            mse_dict[model_name] = [
                mean_squared_error(y_test_arr, y_pred)
                for y_pred in tqdm(gbt.staged_predict(X_test), desc=f"{model_name} staged_predict", total=params['n_estimators'])
            ]

        elif model_name == 'XGBoost':
            print(f"Training {model_name}...")
            params = {**params}
            params.setdefault('random_state', seed)
            params.setdefault('seed', seed)
            params.setdefault('colsample_bytree', 0.8)
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(X_train, y_train_arr)
            mse_dict[model_name] = [
                mean_squared_error(y_test_arr, xgb_model.predict(X_test, iteration_range=(0, i+1)))
                for i in tqdm(range(params['n_estimators']), desc=f"{model_name} staged_predict")
            ]

        elif model_name == 'LightGBM':
            print(f"Training {model_name}...")
            params = {**params}
            params['random_state'] = seed
            params.setdefault('bagging_fraction', 0.8)   # <--- key
            params.setdefault('bagging_freq', 1)         # <--- key (activates bagging)
            params.setdefault('feature_fraction', 0.8)   # <--- adds variation
            params.setdefault('bagging_seed', seed)
            params.setdefault('feature_fraction_seed', seed)
            lgb_model = lgb.LGBMRegressor(**params, verbose=-1)
            X_train_lgb = pd.DataFrame(X_train)
            X_test_lgb = pd.DataFrame(X_test)
            lgb_model.fit(X_train_lgb, y_train_arr)
            mse_dict[model_name] = [
                mean_squared_error(y_test_arr, lgb_model.predict(X_test, num_iteration=i+1))
                for i in tqdm(range(params['n_estimators']), desc=f"{model_name} staged_predict")
            ]

        elif model_name == 'RF':
            print(f"Training {model_name}...")
            params = {**params}
            params['random_state'] = seed
            params.setdefault('bootstrap', True)
            params.setdefault('max_features', 'sqrt')
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train_arr)
            preds_rf = np.vstack([tree.predict(X_test) for tree in rf.estimators_])
            cumsum_rf = np.cumsum(preds_rf, axis=0)
            mse_dict[model_name] = [
                mean_squared_error(y_test_arr, cumsum_rf[i] / (i+1))
                for i in tqdm(range(preds_rf.shape[0]), desc=f"{model_name} incremental MSE")
            ]

        elif model_name == 'ElasticNet':
            print(f"Training {model_name}...")
            elastic = ElasticNet(**params)
            elastic.fit(X_train, y_train_arr)
            y_pred_enet = elastic.predict(X_test)
            mse_dict[model_name] = [mean_squared_error(y_test_arr, y_pred_enet)]

        elif model_name == 'GAM':
            print(f"Training {model_name}...")
            gam = LinearGAM(lam=params['lam'], n_splines=params['n_splines'])
            gam.fit(X_train, y_train_arr)
            y_pred_gam = gam.predict(X_test)
            mse_dict[model_name] = [mean_squared_error(y_test_arr, y_pred_gam)]

        elif model_name == 'BRATD':
            print(f"Training {model_name}...")
            bratd = BRATD(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                      max_depth=params['max_depth'], dropout_rate=params['dropout_rate'],
                      min_samples_split=2, subsample_rate=params['subsample_rate'])
            mse_dict[model_name] = bratd.fit(X_train, y_train_arr, X_test, y_test_arr)

        elif model_name == 'Boulevard':
            print(f"Training {model_name}...")
            boulevard = BRATD(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                            max_depth=params['max_depth'], dropout_rate=0.0,
                            min_samples_split=2, subsample_rate=params['subsample_rate'])
            mse_dict[model_name] = boulevard.fit(X_train, y_train_arr, X_test, y_test_arr)

        elif model_name == 'BRATP':
            print(f"Training {model_name}...")
            bratp = BRATP(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                        max_depth=params['max_depth'], n_trees_per_group=params['n_trees_per_group'],
                        min_samples_split=2, subsample_rate=params['subsample_rate'])
            mse_dict[model_name] = bratp.fit(X_train, y_train_arr, X_test, y_test_arr)

        else:
            print(f"Model {model_name} not supported for training.")

    return mse_dict, best_params_dict

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
from cycler import cycler

# Colorblind‑safe Okabe–Ito palette
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#999999",
             "#000000", "#CC3311"]

def plot_mean_std_trajectories(
    mse_runs,
    epoch,
    dataset_id,
    title='Model Performance Comparison',
    plot_dir=None,
    ax=None,
    save_png=True,
    fontsize=6,              # ↓ smaller default
    figsize=(2.2, 2.6),      # ↓ not square; a bit taller for legend & right labels
    dpi=400,                 # ↓ good balance of crispness & size
    show_single_run_band=False  # skip bands if only 1 run
):
    """
    Draw mean trajectories with 95% (±2*SEM) bands across runs.
    If 'ax' is provided, draw onto it and return; otherwise create and save a PNG.
    """

    # --- derived font sizes (smaller overall) ---
    TINY  = max(fontsize - 2, 4)
    SMALL = fontsize
    TITLE = fontsize + 1

    LABEL_MAP = {'BRATD': 'BRAT-D', 'BRATP': 'BRAT-P'}

    created_fig = False
    if ax is None:
        # Use a local rc_context so outside styles don’t leak in
        plt.rcParams.update({
            "font.size": SMALL,
            "axes.titlesize": TITLE,
            "axes.labelsize": SMALL,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "legend.fontsize": TINY,
            "axes.prop_cycle": cycler(color=OKABE_ITO)  # colorblind‑safe
        })
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # leave right margin for the final MSE labels
        plt.subplots_adjust(right=0.80)
        created_fig = True

    # Distinguishable line styles (helps when colors overlap)
    LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    STYLE_MAP = {}  # filled on the fly to keep model→style stable

    # --- aggregate into {model: list_of_runs} ---
    aggregated = {}
    for run in mse_runs:
        for model_name, mses in run.items():
            aggregated.setdefault(model_name, []).append(mses)

    final_mse_legend = []
    line_handles     = []
    final_points = []  # will hold (x_last, y_last, color, text)

    # Ensures deterministic drawing order
    model_names = sorted(aggregated.keys())

    for k_idx, model_name in enumerate(model_names):
        runs = aggregated[model_name]
        # normalize ragged lists -> 2D array
        runs_array = np.array(runs, dtype=float)
        pretty_label = LABEL_MAP.get(model_name, model_name)

        # Assign a stable linestyle per model
        STYLE_MAP.setdefault(model_name, LINESTYLES[k_idx % len(LINESTYLES)])
        ls = STYLE_MAP[model_name]

        # Single-epoch => dashed baseline line
        if runs_array.ndim == 2 and runs_array.shape[1] == 1:
            mean_mse = runs_array.mean(axis=0)[0]
            x_vals   = np.arange(1, epoch + 1)
            y_vals   = np.full_like(x_vals, mean_mse, dtype=float)

            line, = ax.plot(
                x_vals, y_vals,
                label=pretty_label,
                linestyle="--",
                linewidth=0.8,
                zorder=2
            )
            color = line.get_color()
            ax.scatter(x_vals[-1], mean_mse, s=6, zorder=3, color=color)

            final_mse_legend.append((mean_mse, color))
            line_handles.append(line)

            label_text = f"{mean_mse[-1]:.3f}" if runs_array.shape[1] != 1 else f"{mean_mse:.3f}"
            x_last = x_vals[-1]
            y_last = mean_mse[-1] if runs_array.shape[1] != 1 else mean_mse
            final_points.append((x_last, y_last, color, label_text))

            continue

        # ---- multi-epoch branch ----
        # runs: list[list[float]] of per-epoch MSE, may be equal or ragged length
        n_runs = len(runs)
        max_T  = max(len(r) for r in runs)

        # NaN-pad so we can do epoch-wise nanmean/nanstd robustly
        arr = np.full((n_runs, max_T), np.nan, dtype=float)
        for i, r in enumerate(runs):
            arr[i, :len(r)] = r

        mean_mse = np.nanmean(arr, axis=0)
        counts   = np.sum(~np.isnan(arr), axis=0)

        # sample std, then SEM; guard divisions when counts<=1
        std_mse  = np.nanstd(arr, axis=0, ddof=1)
        sem_mse  = np.divide(std_mse, np.sqrt(np.maximum(counts, 1)),
                            out=np.zeros_like(std_mse),
                            where=counts > 1)

        x_vals = np.arange(1, max_T + 1)

        line, = ax.plot(x_vals, mean_mse, label=pretty_label, linewidth=0.9, linestype = ls, zorder=3)
        color = line.get_color()
        ax.scatter(x_vals[-1], mean_mse[-1], s=6, color=color, zorder=4)

        label_text = f"{mean_mse[-1]:.3f}"
        x_last = x_vals[-1]
        y_last = mean_mse[-1]
        final_points.append((x_last, y_last, color, label_text))

        # draw 95% band where we actually have ≥2 runs
        if np.any(counts > 1):
            half = 2.0 * sem_mse  # ~95% CI
            ax.fill_between(
                x_vals,
                mean_mse - half,
                mean_mse + half,
                where=counts > 1,
                interpolate=True,
                color=color,
                alpha=0.15,
                linewidth=0.0,
                zorder=2
            )

        final_mse_legend.append((mean_mse[-1], color))
        line_handles.append(line)
        # ---- end multi-epoch branch ----

    # --- axes, scales, ticks ---
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))

    ax.set_xlabel('Ensemble Size', fontsize=SMALL, labelpad=1)
    ax.set_ylabel('MSE', fontsize=SMALL, labelpad=1)
    ax.set_title(title, fontsize=TITLE, pad=2)

    # fine grid, very light
    ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)

    # --- compact legend ---
    leg = ax.legend(
        handles=line_handles,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        fontsize=TINY,
        frameon=True,
        framealpha=0.75,
        fancybox=False,
        borderpad=0.2,
        labelspacing=0.15,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.6
    )
    leg.get_frame().set_linewidth(0.3)

    # make a bit of room to the right for labels
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.08)

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    # sort by final y, add a tiny jitter on log-scale to avoid perfect overlaps
    final_points_sorted = sorted(final_points, key=lambda t: t[1])
    n = len(final_points_sorted)
    if n > 1:
        logy = np.log10([p[1] for p in final_points_sorted])
        span = max(logy) - min(logy) if max(logy) > min(logy) else 1.0
        jitter = 0.03 * span
    else:
        jitter = 0.0

    for i, (x_last, y_last, color, text) in enumerate(final_points_sorted):
        y_adj = 10 ** (np.log10(y_last) + (i - (n - 1) / 2) * jitter) if n > 1 else y_last
        ax.text(1.02, y_adj, text, transform=trans,
                fontsize=max(fontsize - 2, 4), color=color,
                va='center', ha='left')


    if created_fig and save_png:
        if plot_dir is None:
            plot_dir = "."
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"mse_trajectories_{dataset_id}.png")
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close()

def empirical_coverage(c, y_true, y_pred, widths):
    """
    Compute the fraction of points for which y_true lies in
    [y_pred - c*widths, y_pred + c*widths].
    """
    lower = y_pred - c * widths
    upper = y_pred + c * widths
    return np.mean((y_true >= lower) & (y_true <= upper))

def find_min_scale(y_true, y_pred, widths,
                   target=0.95, tol=1e-3, c_max=50.0, max_iters=50):
    """
    Binary-search for the minimal c ∈ [0, c_max] such that
    empirical_coverage(c) ≥ target, to within tolerance tol.
    """
    lo, hi = 0.0, c_max
    # ensure hi is valid
    if empirical_coverage(hi, y_true, y_pred, widths) < target:
        raise ValueError(f"c_max={c_max} too small; increase it.")
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        if empirical_coverage(mid, y_true, y_pred, widths) >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi

