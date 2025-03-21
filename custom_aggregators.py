"""
Cuustom aggregation functions for strategy implementations.
"""

import numpy as np

from functools import reduce
from typing import List, Tuple
from flwr.common import NDArrays

def aggregate_WgtAvg(results_with_id: List[Tuple[NDArrays, int, str]]) -> NDArrays:
    """
    Computes the weighted average of aggregated model parameters.
    
    Each entry in `results_with_id` is a tuple containing the following:
    - weights: model parameters for the client.
    - num_examples: the number of examples the client used during training.
    - client_id: the unique identifier of the client.
    
    This function calculates a weighted average based on the number of examples each client used 
    during training. Larger client datasets have more influence on the aggregated model parameters.
    
    Args:
        results_with_id (List[Tuple[NDArrays, int, str]]): A list of tuples containing the model parameters,
            number of examples, and client ID for each client.

    Returns:
        NDArrays: The weighted average of the model parameters across all clients.
    """
    # Sort by client_id so the summation order is consistent each time
    results_sorted = sorted(results_with_id, key=lambda x: x[2])

    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples, _) in results_sorted)

    # Multiply each layer by the number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for (weights, num_examples, _) in results_sorted
    ]

    # Compute the average weights for each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def aggregate_FedAvg(results_with_id: List[Tuple[NDArrays, int, str]]) -> NDArrays:
    """
    Computes the unweighted average of aggregated model parameters.
    
    Each entry in `results_with_id` is a tuple containing the following:
    - weights: model parameters for the client.
    - num_examples: the number of examples the client used during training (but not used in FedAvg).
    - client_id: the unique identifier of the client.
    
    This function calculates a simple unweighted average of the model parameters across all clients.
    
    Args:
        results_with_id (List[Tuple[NDArrays, int, str]]): A list of tuples containing the model parameters,
            number of examples, and client ID for each client.

    Returns:
        NDArrays: The unweighted average of the model parameters across all clients.
    """
    # Sort by client_id so the summation order is consistent each time
    results_sorted = sorted(results_with_id, key=lambda x: x[2])

    # Extract just the weights from each tuple
    all_weights = [weights for (weights, _num_examples, _client_id) in results_sorted]

    # For each layer, sum up across all clients, then divide by the number of clients
    num_clients = len(all_weights)
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_clients
        for layer_updates in zip(*all_weights)
    ]
    return weights_prime