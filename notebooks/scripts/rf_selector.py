import numpy as np
import pymc3 as pm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def select_model(models, statistics, observations, n_samples=1000, size=None,
                 n_trees=100, f_max_features=0.5, random_seed=None):
    """
    Parameters
    ----------
    models : list of tuple
        A list of PyMC3 models and traces
    statistics: list
        List of summary statistics
    observations : array-like
        observed data
    n_samples : int
        Number of posterior predictive samples to generate. Defaults to 1000.
    size : int
        The number of random draws from the distribution specified by the parameters in each
        sample of the trace. Defaults to None, i.e. the same size of observations
    n_trees : int
        Number of trees in the random forest. Defaults to 100.
    f_max_features : float
        The maximun number of features (summary statistics) to consider when looking for the best split.
        This parameter is expresed as a fraction of the total number of features. Defaults to 0.5.
    random_seed : int
        Seed for the random number generator.
    """
    n_models = len(models)
    ref_table = []
    for model, trace in models:
        obs_name = model.observed_RVs[0].name
        pps = pm.sample_posterior_predictive(trace, samples=n_samples, size=size, model=model,
                                             progressbar=False, random_seed=random_seed)

        pps_sum = []
        for stat in statistics:
            val = np.apply_along_axis(stat, 1, pps[obs_name])
            if val.ndim > 1:
                for v in val.T:
                    pps_sum.append(v)
            else:
                pps_sum.append(val)

        ref_table.append(np.array(pps_sum).T)
    ref_table = np.concatenate(ref_table)

    obs_sum = []
    for stat in statistics:
        val = stat(observations)
        if val.ndim > 1:
            for v in val.T:
                obs_sum.append(v)
        else:
            obs_sum.append(val)

    obs_sum = np.hstack(obs_sum)
    labels = np.repeat(np.arange(n_models), n_samples)

    # Define the Random Forest classifier
    max_features = int(f_max_features*ref_table.shape[1])
    classifier = RandomForestClassifier(n_estimators=n_trees,
                                        max_features=max_features,
                                        bootstrap=True,
                                        random_state=random_seed)

    classifier.fit(ref_table, labels)

    best_model = int(classifier.predict([obs_sum]))

    # Compute missclassification error rate
    pred_prob = classifier.predict_proba(ref_table)
    pred_error = 1 - np.take(pred_prob.T, labels)

    # Estimate a regression function with prediction error as response 
    # on summary statitistics of the reference table
    regressor = RandomForestRegressor(n_estimators=n_trees)
    regressor.fit(ref_table, pred_error)

    prob_best_model = 1 - regressor.predict([obs_sum])

    return best_model, prob_best_model.item()
