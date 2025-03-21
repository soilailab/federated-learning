"""
A Flower client script that:
1. Reads config and loads its assigned client data partition.
2. Builds the CNN model (from model.py) using config hyperparams.
3. Sets up a FlowerClient class that handles local training/evaluation.
4. Connects to the Flower server.

Example usage: python client.py -cid 0
"""
import keras
import argparse
import random
import numpy as np
import tensorflow as tf
import flwr as fl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error as mse
from kennard_stone import train_test_split
from scipy.stats import iqr

from config import config
from dataset import create_partitioned_datasets
from model import create_model

seed = config["seed"]
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
keras.utils.set_random_seed(seed)

soil_properties_columns = config["soil_properties"]
model_params = config["model_params"]
batch_size = model_params["batch_size"]
local_epochs = model_params["epochs"]
early_stop_patience = model_params["early_stop_patience"]
val_ratio = config["val_ratio"] 


# --------------------------------------------------------------------------
# Utility Metric Functions
# --------------------------------------------------------------------------
def RMSE(y_true, y_pred):
    """Computes Root Mean Squared Error (RMSE)"""
    return np.sqrt(mse(y_true, y_pred, multioutput="raw_values"))

def RPIQ(y_true, y_pred):
    """Computes Relative Percent Interquartile Range (RPIQ)"""
    return iqr(y_true, axis=0) / RMSE(y_true, y_pred)

def eval_learning(y_true, y_pred):
    """
    Evaluates the model by calculating R2, RMSE, and RPIQ for each soil property.
    
    Args:
        y_true (numpy.array): Actual values of soil properties.
        y_pred (numpy.array): Predicted values of soil properties.
        
    Returns:
        tuple: Arrays of R2, RMSE, and RPIQ values for each target column.
    """
    r2 = []
    rmse_vals = []
    rpiq_vals = []
    y_true = y_true.to_numpy()
    for col_idx in range(y_true.shape[1]):
        y_col_true = y_true[:, col_idx]
        y_col_pred = y_pred[:, col_idx]
        
        r2.append(r2_score(y_col_true, y_col_pred))
        rmse_vals.append(np.sqrt(mse(y_col_true, y_col_pred)))
        rpiq_vals.append(iqr(y_col_true) / np.sqrt(mse(y_col_true, y_col_pred)))
    
    return np.array(r2), np.array(rmse_vals), np.array(rpiq_vals)


# --------------------------------------------------------------------------
# Flower Client Class Definition
# --------------------------------------------------------------------------
class FlowerClient(fl.client.NumPyClient):
    """
    Flower Client class for federated learning.

    This class handles local model training and evaluation on the client's data.
    It communicates with the Flower server to share model parameters and metrics.

    Args:
        model (keras.Model): The local model to train and evaluate.
        X_train (numpy.array): Training data features.
        y_train (numpy.array): Training data labels.
        X_val (numpy.array): Validation data features.
        y_val (numpy.array): Validation data labels.
        cid (int): Client ID.
    """
    def __init__(self, model, X_train, y_train, X_val, y_val, cid):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cid = cid
        self.local_round = 0

    def get_parameters(self, config):
        """Return local model weights to the server"""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Local model training on the client's data.

        Args:
            parameters (numpy.array): Model weights to initialize the local model.
            config (dict): Configuration parameters passed from the server.

        Returns:
            tuple: Updated model weights, number of training examples, and metrics.
        """
        self.model.set_weights(parameters)
        print(f"Client {self.cid} - Starting local training")
        epochs = config.get("local_epochs", local_epochs)
        b_size = config.get("batch_size", batch_size)


        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            restore_best_weights=True
        )

        # checkpoint_dir = f"client_weights/client_{self.cid}/"
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # model_checkpoint = ModelCheckpoint(
        #     filepath=f"client_weights/client_{self.cid}/round{self.local_round}_best_model.h5",
        #     monitor="val_loss",
        #     save_best_only=True,
        #     save_weights_only=False,
        #     verbose=0
        # )
        # Train the model on the client's data
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=b_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            shuffle=False,
        )

        y_pred = self.model.predict(self.X_val)
        r2_vals, rmse_vals, rpiq_vals = eval_learning(self.y_val, y_pred)

        metrics_dict = {}
        for i, prop in enumerate(soil_properties_columns):
            metrics_dict[f"R2-{prop}"] = r2_vals[i]
            metrics_dict[f"RMSE-{prop}"] = rmse_vals[i]
            metrics_dict[f"RPIQ-{prop}"] = rpiq_vals[i]
            metrics_dict["client_id"] = self.cid


        print(f"Client {self.cid} - Local training metrics: {metrics_dict}")
        self.local_round += 1
        return self.model.get_weights(), len(self.X_train), metrics_dict

    def evaluate(self, parameters, config):
        """
        Evaluate the local model on the validation set after aggregation.
        
        Args:
            parameters (numpy.array): Model weights to evaluate.
            config (dict): Configuration parameters passed from the server.

        Returns:
            tuple: Loss, number of validation examples, and evaluation metrics.
        """
        self.model.set_weights(parameters)
        print(f"Client {self.cid} - Local evaluation after server aggregation")

        loss = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        y_pred = self.model.predict(self.X_val)
        r2_vals, rmse_vals, rpiq_vals = eval_learning(self.y_val, y_pred)

        metrics_dict = {}
        for i, prop in enumerate(soil_properties_columns):
            metrics_dict[f"R2-{prop}"] = r2_vals[i]
            metrics_dict[f"RMSE-{prop}"] = rmse_vals[i]
            metrics_dict[f"RPIQ-{prop}"] = rpiq_vals[i]
        print(f"Client {self.cid} - Local eval metrics: {metrics_dict}")
        return loss, len(self.X_val), metrics_dict



# --------------------------------------------------------------------------
# Main Function to Run Client
# --------------------------------------------------------------------------
def main():
    """
    Main function to start a Flower client instance for federated learning.
    
    It configures the client, partitions data, creates the model, and connects
    to the Flower server to begin the federated learning process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cid",
        "-cid",
        type=int,
        default=0,
        help="Client ID (e.g., 0, 1, 2, ...)"
    )

    args = parser.parse_args()
    cid = args.cid
    X_client, y_client = create_partitioned_datasets(cid=cid)
    print(f"[Client {cid}] Received data with shape X={X_client.shape}, y={y_client.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_client, 
        y_client, 
        test_size=val_ratio, 
        random_state=config["seed"], 
        shuffle=True
    )
    print(f"[Client {cid}] Train size: {X_train.shape}, Val size: {X_val.shape}")
    model = create_model()

    client = FlowerClient(model, X_train, y_train, X_val, y_val, cid=cid)
    fl.client.start_numpy_client(
        server_address=config["server_address"],
        client=client
    )

if __name__ == "__main__":
    main()