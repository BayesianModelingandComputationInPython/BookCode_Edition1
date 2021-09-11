from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def plot_pdp(bart, X=None, Y=None, style="pdp", kind="linear", xs_values=None, var_idx=None,
             subsample=False, sharey=True, rug=True, smooth=True, indices=None, grid="long",
             color="C0", color_mean="C0", alpha=0.1, figsize=None, smooth_kwargs=None, ax=None):

    if isinstance(X, pd.DataFrame):
        X_names = list(X.columns)
        X = X.values
    else:
        X_names = []
    
    if isinstance(Y, pd.DataFrame):
        Y_label = f"Predicted {Y.name}"
    else:
        Y_label = "Predicted Y"
        
    barto = bart.distribution
    num_observations = X.shape[0]
    num_covariates = X.shape[1]

    indices = list(range(num_covariates))
    
    if var_idx is None:
        var_idx = indices

    if X_names:
        X_labels = [X_names[idx] for idx in var_idx]
    else:
        X_labels = [f"X_{idx}" for idx in var_idx]


    if kind == "linear" and xs_values is None:
        xs_values = 10
    
    if kind == "quantiles" and xs_values is None:
        xs_values = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
    
    new_Y = []
    new_X_target = []
    if subsample:
        idx_s = np.random.choice(range(X.shape[0]), size=subsample, replace=False)
        new_X = X[idx_s]#np.zeros((subsample, X.shape[1]))
    else:
        new_X = np.zeros_like(X)
        idx_s = list(range(X.shape[0]))
    for i in var_idx: 
        indices_mi = indices[:]
        indices_mi.pop(i)
        preds = []
        if style == "pdp":
            if kind == "linear":
                X_is = np.linspace(X[:,i].min(), X[:,i].max(),  xs_values) 
            elif kind == "quantiles":
                X_is = np.quantile(X[:,i], q=xs_values)
            elif kind == "insample":
                X_is = X[:,i]

            for x_i in X_is:
                if subsample:
                    new_X[:,indices_mi] = X[idx_s][:,indices_mi]
                else:
                    new_X[:,indices_mi] = X[:,indices_mi]
                new_X[:,i] =  x_i
                preds.append(barto.predict(new_X).mean(1)) # average over the X-i variables
            new_X_target.append(X_is)
        else:
            if subsample:
                num_observations = subsample
            tmp = []
            for i_ in range(num_observations):
                new_X = X[idx_s]
                new_X[:,indices_mi] = X[:,indices_mi][i_]
                tmp.append(new_X)
            new_X_ = np.vstack(tmp)
            count = 0
            predicted = barto.predict(new_X_)
            for _ in range(subsample):
                xx = predicted[:,count:count+subsample]
                preds.append(xx.mean(0))
                count += subsample
            new_X_target.append(new_X[:,i])
        new_Y.append(np.array(preds).T)

    
    if ax is None:
        if grid == "long":
            fig, axes = plt.subplots(len(var_idx), sharey=sharey, figsize=figsize)
        elif grid == "wide":
            fig, axes = plt.subplots(1, len(var_idx), sharey=sharey, figsize=figsize)        
        elif isinstance(grid, tuple):
            fig, axes = plt.subplots(grid[0], grid[0], sharey=sharey, figsize=figsize)
        axes = np.ravel(axes)
        fig.text(-0.05, 0.5, Y_label, va='center', rotation='vertical', fontsize=15)

    else:
        axes = [ax]

    if rug:
        lb = np.min(new_Y)
    
    for i, ax in enumerate(axes):
        if smooth:
            if smooth_kwargs is None:
                smooth_kwargs = {}
            smooth_kwargs.setdefault("window_length", 55)
            smooth_kwargs.setdefault("polyorder", 2)
            x_data = np.linspace(new_X_target[i].min(), new_X_target[i].max(), 200)
            x_data[0] = (x_data[0] + x_data[1]) / 2
            if style == "pdp":
                interp = griddata(new_X_target[i], new_Y[i].mean(0), x_data)
            else:
                interp = griddata(new_X_target[i], new_Y[i], x_data) 

            y_data = savgol_filter(interp, axis=0, **smooth_kwargs)

            if style == "pdp":
                az.plot_hdi(new_X_target[i], new_Y[i][None,:], ax=ax)
                ax.plot(x_data, y_data, color=color)
            else:
                ax.plot(x_data, y_data.mean(1), color=color_mean)
                ax.plot(x_data, y_data, color=color, alpha=alpha)


        else:
            idx = np.argsort(new_X_target[i])
            if style == "pdp":
                ax.plot(new_X_target[i][idx], new_Y[i][idx].mean(0), color=color)
                az.plot_hdi(new_X_target[i], new_Y[i], smooth=smooth, ax=ax)
            else:
                ax.plot(new_X_target[i][idx], new_Y[i][idx], color=color, alpha=alpha)
                ax.plot(new_X_target[i][idx], new_Y[i][idx].mean(1), color=color_mean)
            
        if rug:
            ax.plot(X[:,i], np.full_like(X[:,i], lb), 'k|')
        
        ax.set_xlabel(X_labels[i])
    
    return axes
