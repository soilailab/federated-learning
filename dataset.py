"""
Auxuliary functions for: 
- Loading the full dataset (VNIR or MIR).
- Splitting out a global test set for the server.
- Partitioning the remaining training data among clients according to different criteria (IID, BCL, Regions).
"""

import pandas as pd
import numpy as np
from config import config
from kennard_stone import train_test_split

def load_data():
    """
    Loads the dataset from a CSV file, separates the target (soil properties) and feature 
    (spectral + zone_expl/state_class) matrices. Filters spectral columns based on the specified 
    spectra type (VNIR or MIR).
    
    Returns:
        X (DataFrame): Feature matrix (spectral data and any partitioning columns).
        y (DataFrame): Target matrix (soil properties).
    """
    df = pd.read_csv(config["client_data"])

    # Extract target columns (soil properties)
    target_cols = config["soil_properties"]  
    y = df[target_cols].copy()

    if config["spectra_type"].lower() == "vnir":
        spectral_cols = [
            col for col in df.columns 
            if col.isdigit() and 351 <= int(col) <= 2491
        ]
    elif config["spectra_type"].lower() == "mir":
        spectral_cols = [
            col for col in df.columns 
            if col.isdigit() and 599 <= int(col) <= 4010
        ]
    else:
        raise ValueError("Invalid spectra_type in config. Must be 'VNIR' or 'MIR'.")

    # Keep partitioning columns (zone_expl, state_class) if present
    partition_cols = []
    for col_name in ["zone_expl", "state_class"]:
        if col_name in df.columns:
            partition_cols.append(col_name)

    # Combine spectral + partition columns for X
    X_cols = spectral_cols + partition_cols
    X = df[X_cols].copy()
    return X, y


def partition_IID(X, y, num_clients):
    """
    Partitions the entire dataset into `num_clients` clients in an IID fashion by shuffling and splitting the data randomly.
    
    Args:
        X (DataFrame): Features.
        y (DataFrame): Target (soil properties).
        num_clients (int): Number of clients for partitioning.

    Returns:
        dict: A dictionary mapping client IDs to their corresponding data (X_client, y_client).
    """    
    data_indices = np.arange(len(X))
    np.random.shuffle(data_indices)
    partitions = np.array_split(data_indices, num_clients)

    clients_data_dict = {}
    for client_id, indices in enumerate(partitions):
        X_client = X.iloc[indices].reset_index(drop=True)
        y_client = y.iloc[indices].reset_index(drop=True)
        X_client = X_client.drop(columns=["state_class", "zone_expl"], errors="ignore")
        clients_data_dict[client_id] = (X_client, y_client)

    return clients_data_dict


def partition_BCL(X, y):
    """
    Partitions the data by the `zone_expl` column. Each client corresponds to a unique `zone_expl` value.
    
    Args:
        X (DataFrame): Features (must contain `zone_expl` column).
        y (DataFrame): Target (soil properties).
    
    Returns:
        dict: A dictionary mapping client IDs (corresponding to unique zones) to their respective data.
    """    
    if "zone_expl" not in X.columns:
        raise ValueError("X must contain 'zone_expl' column for BCL partitioning.")

    clients_data_dict = {}
    unique_zones = X["zone_expl"].unique()
    for client_id, zone_val in enumerate(unique_zones):
        idx = (X["zone_expl"] == zone_val)
        X_client = X.loc[idx].reset_index(drop=True)
        y_client = y.loc[idx].reset_index(drop=True)
        X_client = X_client.drop(columns=["state_class", "zone_expl"], errors="ignore")
        clients_data_dict[client_id] = (X_client, y_client)

    return clients_data_dict


def partition_Regions(X, y):
    """
    Partitions the data by the `state_class` column. Each client corresponds to a unique `state_class` value.
    
    Args:
        X (DataFrame): Features (must contain `state_class` column).
        y (DataFrame): Target (soil properties).
    
    Returns:
        dict: A dictionary mapping client IDs (corresponding to unique regions) to their respective data.
    """    
    if "state_class" not in X.columns:
        raise ValueError("X must contain 'state_class' column for Regions partitioning.")

    clients_data_dict = {}
    unique_regions = X["state_class"].unique()
    for client_id, region_val in enumerate(unique_regions):
        idx = (X["state_class"] == region_val)
        X_client = X.loc[idx].reset_index(drop=True)
        y_client = y.loc[idx].reset_index(drop=True)
        X_client = X_client.drop(columns=["state_class", "zone_expl"], errors="ignore")
        clients_data_dict[client_id] = (X_client, y_client)
    return clients_data_dict


def create_partitioned_datasets(cid=None, random_state=47):
    """
    Loads the dataset and partitions it among clients based on the specified criteria (IID, BCL, Regions).
    If `cid` is provided, returns data only for that client. Otherwise, returns data for all clients.
    
    Args:
        cid (int, optional): Client ID for which to return data. If None, returns data for all clients.
        random_state (int, optional): Seed for random operations to ensure reproducibility.
        
    Returns:
        dict or tuple: If `cid` is None, returns a dictionary mapping client IDs to their data.
                       If `cid` is an integer, returns the (X_client, y_client) for that client.
    """
    if random_state is None:
        random_state = config["seed"]

    X, y = load_data()
    criteria = config["criteria"]
    num_clients = config["num_clients"]

    if criteria.lower() in ["iid", "random"]:
        clients_data_dict = partition_IID(X, y, num_clients)
    elif criteria.lower() in ["bcl", "zone_expl"]:
        clients_data_dict = partition_BCL(X, y)
    elif criteria.lower() in ["regions", "state_class"]:
        clients_data_dict = partition_Regions(X, y)
    else:
        raise ValueError(f"Unknown criteria in config: {criteria}")

    if cid is not None:
        return clients_data_dict[cid]
    
    return clients_data_dict
    




