"""
Sets up a Flower server that:
1. Loads hyperparameters and settings from config.py.
2. Builds a global model from model.py for evaluation on a global test set.
3. Uses a custom SaveModelStrategy to aggregate weights and evaluate each round.
4. Starts the federated learning server.

Example usage: python server.py
"""

import os
import timeit
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl
import keras

from typing import Dict, Optional, Tuple, List, Union
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from custom_aggregators import aggregate_WgtAvg, aggregate_FedAvg
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import iqr

from config import config
from model import create_model 

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()


soil_properties_columns = config["soil_properties"]
path_VNIR = config["path_X_VNIR"]
path_MIR = config["path_X_MIR"]
path_y = config["path_y"]
metrics_dir = config["metrics_dir"]

# --------------------------------------------------------------------------
# Utility Metric Functions
# --------------------------------------------------------------------------
def RMSE(y_test, y_pred):
    """Computes Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))

def RPIQ(y_test, y_pred):
    """Computes Relative Percent Interquartile Range (RPIQ)"""
    return iqr(y_test, axis=0) / RMSE(y_test, y_pred)

def eval_learning(y_test, y_pred):
    """
    Evaluates the model by calculating R2, RMSE, and RPIQ for each soil property.
    
    Args:
        y_true (numpy.array): Actual values of soil properties.
        y_pred (numpy.array): Predicted values of soil properties.
        
    Returns:
        tuple: Arrays of R2, RMSE, and RPIQ values for each target column.
    """
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")
    rmse = RMSE(y_test, y_pred)
    rpiq = RPIQ(y_test, y_pred)
    return r2, rmse, rpiq

# --------------------------------------------------------------------------
# Metrics Aggregation: Weighted & Simple Average
# --------------------------------------------------------------------------
def wgt_average_metrics(metrics: List[Tuple[int, Dict[str, float]]]):
    """
    Computes a weighted average of metrics across all clients, weighted by the number of examples.
    
    Args:
        metrics (list): List of tuples containing number of examples and the metrics for each client.
        
    Returns:
        dict: Aggregated weighted metrics across clients.
    """
    if not metrics:
        raise ValueError("The metrics list is empty.")
    
    print(f"Calculating weighted average metrics from {len(metrics)} clients.")
    wgt_aggregated_metrics = {}
    for metric in ["RMSE","R2", "RPIQ"]:
        for sp in soil_properties_columns:
            key = f"{metric}-{sp}"
            # Weighted average
            numerator = sum(m[0] * m[1][key] for m in metrics)
            denominator = sum(m[0] for m in metrics)
            wgt_aggregated_metrics[key] = numerator / denominator

    return wgt_aggregated_metrics

def simple_mean_metrics(metrics: List[Tuple[int, Dict[str, float]]]):
    """Computes a simple unweighted mean of metrics across clients."""
    if not metrics:
        raise ValueError("The metrics list is empty.")

    print(f"Calculating simple average metrics from {len(metrics)} clients.")
    wgt_aggregated_metrics = {}
    for metric in ["RMSE","R2", "RPIQ"]:
        for sp in soil_properties_columns:
            key = f"{metric}-{sp}"
            # Unweighted average
            values = [m[1][key] for m in metrics]
            wgt_aggregated_metrics[key] = np.mean(values)

    return wgt_aggregated_metrics

# --------------------------------------------------------------------------
# Load Global Test Data (for evaluation)
# --------------------------------------------------------------------------
def load_global_test_data():
    """
    Loads the global test dataset (VNIR or MIR spectra) and corresponding labels.
    Adjust for your use case or unify with dataset.py if needed.
    
    Returns:
        Tuple: Features (X_test) and labels (y_test).
    """
    if config["spectra_type"] == "MIR":
        X_test = pd.read_csv(f"{path_MIR}")
    else:
        X_test = pd.read_csv(f"{path_VNIR}")
    
    y_test = pd.read_csv(f"{path_y}")
    print(X_test.shape)
    print(y_test.shape)
    return X_test, y_test


# --------------------------------------------------------------------------
# Custom Strategy for Saving Model Weights (FedAvg)
# --------------------------------------------------------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    A custom FedAvg strategy that saves aggregated model weights and logs metrics after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates the model weights from all participating clients.
        
        Args:
            server_round (int): Current round number.
            results (list): Results from client fit methods.
            failures (list): List of any failed client results.
        
        Returns:
            Tuple: Aggregated model parameters and the aggregated metrics.
        """

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['client_id'] )
            for _client, fit_res in results
        ]
        if config["strategy"].lower() == "wgtavg":
            aggregated_weights = aggregate_WgtAvg(weights_results)
        elif config["strategy"].lower() == "fedavg":
            aggregated_weights = aggregate_FedAvg(weights_results)
        else:
            raise ValueError(f"Unknown strategy_type: {config['strategy_type']}")
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Save aggregated weights
        save_dir = config["save_weights_dir"]
        os.makedirs(save_dir, exist_ok=True)
        np.savez(f"{save_dir}/round-{server_round}-weights.npz", *aggregated_weights)
        
        metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        weighted_metrics = wgt_average_metrics(metrics_list) 
        simple_avg_metrics = simple_mean_metrics(metrics_list)
        
        print(f"Round {server_round} Weighted Metrics: {weighted_metrics}")
        print(f"Round {server_round} Simple Avg Metrics: {simple_avg_metrics}")
        
        return parameters_aggregated, weighted_metrics


# --------------------------------------------------------------------------
# Global Evaluation Function for the Server
# --------------------------------------------------------------------------
metrics_list = []
best_val_loss = float("inf")
best_round = -1
def get_evaluate_fn(server_model: keras.Model):
    """
    Returns a function that Flower calls to evaluate the global model after each round.
    
    Args:
        server_model (keras.Model): The server's global model.
        
    Returns:
        evaluate (function): Function to evaluate the model on the global test set.
    """
    global config
    X_test, y_test = load_global_test_data()

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        conf: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        global best_val_loss, best_round

        server_model.set_weights(parameters)

        y_pred = server_model.predict(X_test)
        
        total_loss = 0.0
        metrics_dict = {}
        for i, col in enumerate(soil_properties_columns):
            if col not in y_test.columns:
                print(f"Warning: {col} is not in y_test columns.")
                continue
            y_true = y_test[col].values.reshape(-1, 1)
            y_p = y_pred[:, i].reshape(-1, 1)
            loss = mean_squared_error(y_true, y_p)
            total_loss += loss
            r2 = r2_score(y_true, y_p)
            rmse = np.sqrt(loss)
            rpiq_val = iqr(y_true) / rmse if rmse != 0 else 0

            metrics_dict[f"{col}_loss"] = float(loss)
            metrics_dict[f"{col}_R2"] = float(r2)
            metrics_dict[f"{col}_RMSE"] = float(rmse)
            metrics_dict[f"{col}_RPIQ"] = float(rpiq_val)

        avg_loss = total_loss / len(soil_properties_columns)
        metrics_dict["round"] = server_round
        metrics_dict["average_loss"] = float(avg_loss)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_round = server_round

            df_pred = pd.DataFrame(y_pred, columns=soil_properties_columns)
            df_true = y_test[soil_properties_columns].reset_index(drop=True)
            df_out = pd.concat([df_true.add_prefix("True_"), df_pred.add_prefix("Pred_")], axis=1)
            df_out["best_round"] = best_round
            os.makedirs(metrics_dir, exist_ok=True)
            df_out.to_csv(os.path.join(config["metrics_dir"], "obs_pred_bestround.csv"), index=False)

        print(f"Best validation loss is now at round {best_round}, average_loss={best_val_loss:.4f}")
        metrics_list.append(metrics_dict)

        return avg_loss, metrics_dict

    return evaluate


# --------------------------------------------------------------------------
# Main Function to Start the Server
# --------------------------------------------------------------------------
def main():
    
    server_model = create_model()  
    strategy = SaveModelStrategy(
        min_available_clients=config["num_clients"],
        min_evaluate_clients=config["num_clients"],
        min_fit_clients=config["num_clients"],
        on_fit_config_fn=lambda rnd: {
            "batch_size": config["model_params"]["batch_size"],
            "local_epochs": config["model_params"]["epochs"],
            "momentum": config["model_params"]["momentum"],
            "local_epochs": config["model_params"]["epochs"]
        },
        evaluate_fn=get_evaluate_fn(server_model),
        evaluate_metrics_aggregation_fn= simple_mean_metrics if config['strategy'].lower() == "fedavg" else wgt_average_metrics,
        initial_parameters=fl.common.ndarrays_to_parameters(server_model.get_weights()),
    )

    num_rounds = config["num_rounds"]
    fl.server.start_server(
        server_address=config["server_address"],
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
        
    )

    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs(config["metrics_dir"], exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_dir, "server_evaluation_metrics.csv"), index=False)
    print("Server finished training. Metrics saved.")


if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"Execution time in seconds: {execution_time:.1f}")
