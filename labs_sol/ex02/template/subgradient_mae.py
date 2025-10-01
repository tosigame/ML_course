import numpy as np

def compute_mae_loss(y,tx,w):
    return np.mean(np.abs(y-tx@w))


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    # ***************************************************
    e = y -tx@w
    subgrad = -(tx.T @ np.sign(e))/len(y)
    return subgrad
