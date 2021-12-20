import numpy as np
import matplotlib.pyplot as plt

def splines(knots, x_true=None, y_true=None):
    # Get x-y values from true function
    if x_true is None:
        x_min, x_max = 0, 6
        x_true = np.linspace(x_min, x_max, 200)
    else:
        x_min, x_max = min(x_true), max(x_true)  
    
    if y_true is None:
        y_true = np.sin(x_true)

    # Prepare figure
    _, axes = plt.subplots(2, 2, figsize=(9, 6),
                         constrained_layout=True,
                         sharex=True, sharey=True)
    axes = np.ravel(axes)

    # Plot the knots and true function
    for ax in axes:
        ax.vlines(knots, -1, 1, color='grey', ls='--')
        ax.plot(x_true, y_true, 'C4.', lw=4, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    labels = ["Piecewise Constant", "Piecewise Linear", "Piecewise Quadratic", "Piecewise Cubic"]
    for order, ax, label in zip(range(0, 4), axes, labels):
        B = basis(x_true, order, knots)
        y_hat = ols(B, y_true)
        ax.plot(x_true, y_hat, c='k')
        ax.set_title(label)

def basis(x_true, order, knots):
    """compute basis functions"""
    B = []
    for i in range(0, order+1):
        B.append(x_true**i)
    for k in knots:
        B.append(np.where(x_true < k, 0, (x_true - k) ** order))
    B = np.array(B).T
    return B


def ols(X, y):
    """Compute ordinary least squares in closed-form"""
    β = np.linalg.solve(X.T @ X, X.T @ y)
    return X @ β

