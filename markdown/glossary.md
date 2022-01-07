(glossary)=

# Glossary

**Autocorrelation**: Autocorrelation is the correlation of a signal
with a lagged copy of itself. Conceptually, you can think of it as how
similar observations are as a function of the time lag between them.
Large autocorrelation is a concern in MCMC samples as it reduces the
effective sample size.

**Aleatoric Uncertainty**: Aleatoric uncertainty is related to the
notion that there are some quantities that affect a measurement or
observation that are intrinsically unknowable or random. For example,
even if we were able to exactly replicate condition such as direction,
altitude and force when shooting an arrow with a bow. The arrow will
still not hit the same point, because there are other conditions that we
do not control like fluctuations of the atmosphere or vibrations of the
arrow shaft, that are random.

**Bayesian Inference**: Bayesian Inference is a particular form of
statistical inference based on combining probability distributions in
order to obtain other probability distributions. In other words is the
formulation and computation of conditional probability or probability
densities,
$p(\boldsymbol{\theta} \mid \boldsymbol{Y}) \propto p(\boldsymbol{Y} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})$.

**Bayesian workflow**: Designing a good enough model for a given problem
requires significant statistical and domain knowledge expertise. Such
design is typically carried out through an iterative process called
Bayesian workflow. This process includes the three steps of model
building {cite:p}`Gelman2020`: inference, model checking/improvement, and model
comparison. In this context the purpose of model comparison is not
necessarily restricted to pick the *best* model, but more importantly to
better understand the models.

**Causal inference**: or Observational causal inference. The procedures
and tools used to estimate the impact of a treatment (or intervention)
in some system without testing the intervention. That is from
observational data instead of experimental data.

**Covariance Matrix and Precision Matrix**: The covariance matrix is a
square matrix that contains the covariance between each pair of elements
of a collection of random variable. The diagonal of the covariance
matrix is the variance of the random variable. The precision matrix is
the matrix inverse of the covariance matrix.

**Design Matrix**: In the context of regression analysis a design matrix
is a matrix of values of the explanatory variables. Each row represents
an individual object, with the successive columns corresponding to the
variables and their specific values for that observation. It can contain
indicator variables (ones and zeros) indicating group membership, or it
can contain continuous values.

**Decision tree**: A decision tree is a flowchart-like structure in
which each internal node represents a "test\" on an attribute (e.g.
whether a coin flip comes up heads or tails), each branch represents the
outcome of the test, and each leaf node represents a class label
(decision taken after computing all attributes). The paths from root to
leaf represent classification rules. The values at the leaf nodes can be
continuous if the tree is used for regression.

**dse**: The standard error of component-wise differences of `elpd_loo`
between two models. This error is smaller than the standard error (`se`
in `az.compare`) for individual models. The reason being that generally
some observations are as easy/hard to predict for all models and thus
this introduce correlations.

**d_loo**: The difference in `elpd_loo` for two models. If more than
two models are compared, the difference is computed relative to the
model with highest `elpd_loo`).

**Epistimic Uncertainty**: Epistemic uncertainty is related to the lack
of knowledge of the states of a system by some observer. It is related
to the knowledge that we could have in principle but not in practice and
not about the intrinsic unknowable quantity of nature (contrast with
aleatory uncertainty). For example, we may be uncertain of the weight of
an item because we do not have an scale at hand, so we estimate the
weight by lifting it, or we may have one scale but with a precision
limited to the kilogram. We could also have epistemic uncertainty if we
design an experiment or perform a computation ignoring factors. For
example, to estimate how much time we will have to drive to another
city, we may omit the time spent at tolls, or we may assume excellent
weather or road conditions etc. In other words, epistemic uncertainty is
about ignorance and in opposition to aleatoric, uncertainty, we can in
principle reduce it by obtaining more information.

**Statistic**: A statistic (not plural) or sample statistic is any
quantity computed from a sample. Sample statistics are computed for
several reasons including estimating a population (or data generating
process) parameter, describing a sample, or evaluating a hypothesis. The
sample mean (also known as empirical mean) is a statistic, the sample
variance (or empirical variance) is another example. When a statistic is
used to estimate a population (or data generating process) parameter,
the statistic is called an estimator. Thus, the sample mean can be an
estimator and the posterior mean can be another estimator.

**ELPD**: Expected Log-pointwise Predictive Density (or expected log
pointwise predictive probabilities for discrete model). This quantity is
generally estimated by cross-validation or using methods such as WAIC
(`elpd_waic`) or LOO (`elpd_loo`). As probability densities can be
smaller or larger than 1, the ELPD can be negative or positive for
continuous variables and non-negative for discrete variables.

**Exchangeability**: A sequence of Random variables is exchangeable if
their joint probability distribution does not change when the positions
in the sequence is altered. Exchangeable random variables are not
necessarily iid, but iid are exchangeable.

**Exploratory Analysis of Bayesian Models**: The collection of tasks
necessary to perform a successful Bayesian data analysis that are not
the inference itself. This includes. Diagnosing the quality of the
inference results obtained using numerical methods. Model criticism,
including evaluations of both model assumptions and model predictions.
Comparison of models, including model selection or model averaging.
Preparation of the results for a particular audience.

**Hamiltonian Monte Carlo** Hamiltonian Monte Carlo (HMC) is a Markov
chain Monte Carlo (MCMC) method that uses the gradient to efficiently
explore a probability distribution function. In Bayesian statistics this
is most commonly used to obtain samples from the posterior distribution.
HMC methods are instances of the Metropolis--Hastings algorithm, where
the proposed new points are computed from a Hamiltonian, this allows the
methods to proposed new states to be far from the current one with high
acceptance probability. The evolution of the system is simulated using a
time-reversible and volume-preserving numerical integrator (most
commonly the leapfrog integrator). The efficiency of the HMC method is
highly dependant on certain hyperparameters of the method. Thus, the
most useful methods in Bayesian statistics are adaptive dynamics
versions of HMC that can adjust those hyperparameters automatically
during the warm-up or tuning phase.

**Heteroscedasticity**: A sequence of random variables is
heteroscedastic if its random variables do not have the same variance,
i.e. if they are not homoscedastic. This is also known as heterogeneity
of variance.

**Homoscedasticity**: A sequence of random variables is homoscedastic if
all its random variables have the same finite variance. This is also
known as homogeneity of variance. The complementary notion is called
heteroscedasticity.

**iid**: Independent and identically distributed. A collection of random
variables is independent and identically distributed if each random
variable has the same probability distribution as the others and all are
mutually independent. If a collection of random variables is iid it is
also exchangeable, but the converse is not necessarily true.

**Individual Conditional Expectation** ICE: An ICE shows the dependence
between the response variable and a covariate of interest. This is done
for each sample separately with one line per sample. This contrast to
PDPs where the average effect of the covariate is represented.

**Inference**: Colloquially, inference is reaching a conclusion based on
evidence and reasoning. In this book refer to inference we generally
mean about Bayesian Inference, which has a more restricted and precise
definition. Bayesian Inference is the process of conditioning models to
the available data and obtaining posterior distributions. Thus, in order
to reach a conclusion based on evidence and reasoning, we need to
perform more steps that mere Bayesian inference. Hence the importance of
discussing Bayesian analysis in terms of exploratory analysis of
Bayesian models or more generally in term of Bayesian workflows.

**Imputation**: Replacing missing data values through a method of
choice. Common methods may include most common occurrence or
interpolation based on other (present) observed data.

**KDE**: Kernel Density Estimation. A non-parametric method to estimate
the probability density function of a random variable from a finite set
of samples. We often use the term KDE to talk about the estimated
density and not the method.

**LOO**: Short for Pareto smoothed importance sampling leave one out
cross-validation (PSIS-LOO-CV). In the literature "LOO\" may be
restricted to leave one out cross-validation.

**Maximum a Posteriori (MAP)** An estimator of an unknown quantity, that
equals the mode of the posterior distribution. The MAP estimator
requires optimization of the posterior, unlike the posterior mean which
requires integration. If the priors are flat, or in the limit of
infinite sample size, the MAP estimator is equivalent to the Maximum
Likelihood estimator.

**Odds** A measure of the likelihood of a particular outcome. They are
calculated as the ratio of the number of events that produce that
outcome to the number that do not. Odds are commonly used in gambling.

**Overfitting**: A model overfits when produces predictions too closely
to the dataset used for fitting the model failing to fit new datasets.
In terms of the number of parameters an overfitted model contains more
parameters than can be justified by the data. An arbitrary
over-complex model will fit not only the data but also the noise,
leading to poor predictions.

**Partial Dependence Plots** PDP: A PDP shows the dependence between the
response variable and a set of covariates of interest, this is done by
marginalizing over the values of all other covariates. Intuitively, we
can interpret the partial dependence as the expected value of the
response variable as function of the covariates of interest.

**Pareto k estimates** $\hat k$: A diagnostic for Pareto smoothed
importance sampling (PSIS), which is used by LOO. The Pareto k
diagnostic estimates how far an individual leave-one-out observation is
from the full distribution. If leaving out an observation changes the
posterior too much then importance sampling is not able to give reliable
estimates. If $\hat \kappa < 0.5$, then the corresponding component of
`elpd_loo` is estimated with high accuracy. If $0.5< \hat \kappa <0.7$
the accuracy is lower, but still useful in practice. If
$\hat \kappa > 0.7$, then importance sampling is not able to provide a
useful estimate for that observation. The $\hat \kappa$ values are also
useful as a measure of influence of an observation. Highly influential
observations have high $\hat \kappa$ values. Very high $\hat \kappa$
values often indicate model misspecification, outliers, or mistakes in
the data processing.

**Point estimate** A single value, generally but not necessarily in
parameter space, used as a summary of *best estimate* of an unknown
quantity. A point estimate can be contrasted with an interval estimate
like highest density intervals, which provides a range or interval of
values describing the unknown quantity. We can also contrast a point
estimate with distributional estimates, like the posterior distribution
or its marginals.

**p_loo**: The difference between `elpd_loo`: and the
non-cross-validated log posterior predictive density. It describes how
much more difficult it is to predict future data than the observed data.
Asymptotically under certain regularity conditions, `p_loo` can be
interpreted as the effective number of parameters. In well behaving
cases `p_loo` should be lower than the number of parameters in the model
and smaller than the number observations in the data. If not, this is an
indication that the model has very weak predictive capability and may
thus indicate a severe model misspecification. See high Pareto k
diagnostic values.

**Probabilistic Programming Language**: A programming syntax composed of
primitives that allows one to define Bayesian models and perform
inference automatically. Typically a Probabilistic Programming Language
also includes functionality to generate prior or posterior predictive
samples or even to analysis result from inference.

**Prior predictive distribution**: The expected distribution of the data
according to the model (prior and likelihood). That is, the data the
model is expecting to see before seeing any data. See Equation
[\[eq:prior_pred_dist\]](eq:prior_pred_dist). The prior predictive
distribution can be used for prior elicitation, as it is generally
easier to think in terms of the observed data, than to think in terms of
model parameters.

**Posterior predictive distribution**: This is the distribution of
(future) data according to the posterior, which in turn is a consequence
of the model (prior and likelihood) and observed data. In other words,
these are the model's predictions. See Equation
[\[eq:post_pred_dist\]](eq:post_pred_dist). Besides generating
predictions, the posterior predictive distribution can be used to asses
the model fit, by comparing it with the observed data.

**Residuals**: The difference between an observed value and the
estimated value of the quantity of interest. If a model assumes that the
variance is finite and the same for all residuals, we say we have
homoscedasticity. If instead the variance can change, we say we have
heteroscedasticity.

**Sufficient statistics**: A statistic is sufficient with respect to a
model parameter if no other statistic computed from the same sample
provides any additional information about that sample. In other words,
that statistic is *sufficient* to summarize your samples without losing
information. For example, given a sample of independent values from a
normal distribution with expected value $\mu$ and known finite variance
the sample mean is sufficient statistics for $\mu$. Notice that the mean
says nothing about the dispersion, thus it is only sufficient with
respect to the parameter $\mu$. It is known that for iid data the only
distributions with a sufficient statistic with dimension equal to the
dimension of $\theta$ are the distributions from the exponential family.
For other distribution, the dimension of the sufficient statistic
increases with the sample size.

**Synthetic data**: Also known as fake data it refers to data generated
from a model instead of being gathered from experimentation or
observation. Samples from the posterior/prior predictive distributions
are examples of synthetic data.

**Timestamp**: A timestamp is an encoded information to identify when a
certain event happens. Usually a timestamp is written in the format of
date and time of day, with more accurate fraction of a second when
necessary.

**Turing-complete** In colloquial usage, is used to mean that any
real-world general-purpose computer or computer language can
approximately simulate the computational aspects of any other real-world
general-purpose computer or computer language.
