import numpy as np

def mean_squared_loss(y, tx, w):
    """Compute mean squared error loss.
    
    Args:
        y (ndarray): shape (N,)
        tx (ndarray): shape (N, D)
        w (ndarray): shape (D,)
    Returns:
        float: (1 / (2N)) * ||y - txw||²
    """
    e = y - tx @ w 
    return float((e.T @ e) / (2*(len(y))))


def compute_gradient(y, tx, w):
    """Compute gradient of MSE loss.
    
    Args:
        y (ndarray): shape (N,)
        tx (ndarray): shape (N, D)
        w (ndarray): shape (D,)
    Returns:
        ndarray: gradient vector, shape (D,)
    """
    e = y - tx @ w
    return -(tx.T @ e)/len(y)


###################################### Function 1 #####################################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression via gradient descent.
    
    Args:
        y (ndarray): targets, shape (N,)
        tx (ndarray): features, shape (N, D)
        initial_w (ndarray): initial weights, shape (D,)
        max_iters (int): number of iterations
        gamma (float): learning rate
    Returns:
        (ndarray, float): final weights and loss
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma * grad
    loss = mean_squared_loss(y, tx, w)
    return w, loss

#######################################################################################

def compute_stoch_gradient(y, tx, w):
    """Compute stochastic gradient of MSE for one sample."""
    e = y - tx @ w
    return -(tx.T @ e) / y.shape[0]

###################################### Function 2 #####################################
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    
    Args:
        y (ndarray): targets, shape (N,)
        tx (ndarray): features, shape (N, D)
        initial_w (ndarray): initial weights, shape (D,)
        max_iters (int): number of iterations
        gamma (float): learning rate
    Returns:
        (ndarray, float): final weights and loss
    """

    w = initial_w.copy()
    N = len(y)
    rng = np.random.default_rng()

    for _ in range(max_iters):

        i = rng.integers(N)
        y_i = y[i]
        tx_i = tx[i,:]

        grad = compute_stoch_gradient(y_i[None],tx_i[None,:],w)
        w -= gamma * grad

    loss = mean_squared_loss(y,tx,w)
    return w, loss
#######################################################################################

###################################### Function 3 #####################################
def least_squares(y, tx):
    """Least squares regression using normal equations.
    
    Args:
        y (ndarray): targets, shape (N,)
        tx (ndarray): features, shape (N, D)
    Returns:
        (ndarray, float): optimal weights and MSE loss
    """

    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = mean_squared_loss(y, tx, w)
    return w, loss
#######################################################################################

###################################### Function 4 #####################################
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    
    Args:
        y (ndarray): targets, shape (N,)
        tx (ndarray): features, shape (N, D)
        lambda_ (float): regularization parameter
    Returns:
        (ndarray, float): optimal weights and MSE loss (without penalty)
    """
    N, D = tx.shape
    I = np.eye(D)
    A = tx.T @ tx + 2 * N * lambda_ * I
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = mean_squared_loss(y, tx, w)
    return w, loss
#######################################################################################

def sigmoid(t):
    """Apply sigmoid function elementwise."""
    return 1/(1+np.exp(-t))


def log_likelihood_loss(y, tx, w):
    """Compute negative log-likelihood loss for logistic regression."""
    y = y.ravel()
    e = tx @ w
    eps = 1e-15
    sig = np.clip(sigmoid(e), eps, 1 - eps)
    return float(-np.mean(y * np.log(sig) + (1 - y) * np.log(1 - sig)))

def calculate_logistic_gradient(y, tx, w):
    """Compute gradient of logistic loss."""
    y = y.ravel()
    sig = sigmoid(tx @ w)
    return tx.T @ (sig - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.
    
    Args:
        y (ndarray): targets, shape (N,)
        tx (ndarray): features, shape (N, D)
        initial_w (ndarray): initial weights, shape (D,)
        max_iters (int): number of iterations
        gamma (float): learning rate
    Returns:
        (ndarray, float): final weights and loss
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        grad= calculate_logistic_gradient(y,tx,w)
        w -= gamma*grad
    
    loss = log_likelihood_loss(y,tx,w)
    return w, loss


def reg_logistic_regression(y, tx,lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.
    
    Args:
        y (ndarray): targets, shape (N,)
        tx (ndarray): features, shape (N, D)
        lambda_ (float): regularization strength
        initial_w (ndarray): initial weights, shape (D,)
        max_iters (int): number of iterations
        gamma (float): learning rate

    Returns:
        (ndarray, float): final weights and loss (excluding regularization term)
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_ * w
        w -=  gamma*grad
    
    loss = log_likelihood_loss(y,tx,w)
    return w, loss

