---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Code 10: Probabilistic Programming Languages

```python
from scipy import stats
import pymc as pm
import pytensor
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import datetime
print(f"Last Run {datetime.datetime.now()}")
```

```python
az.style.use("arviz-grayscale")
plt.rcParams["figure.dpi"] = 300
```

## Posterior Computation


### Code 10.1

```python
from jax import grad

simple_grad = grad(lambda x: x**2)
print(simple_grad(4.0))
```

### Code 10.2

```python
from jax import grad
from jax.scipy.stats import norm

def model(test_point, observed):
    z_pdf = norm.logpdf(test_point, loc=0, scale=5)
    x_pdf = norm.logpdf(observed, loc=test_point, scale=1)
    logpdf = z_pdf + x_pdf
    return logpdf

model_grad = grad(model)

observed, test_point = 5.0, 2.5 
logp_val = model(test_point, observed)
grad = model_grad(test_point, observed)
print(f"log_p_val: {logp_val}")
print(f"grad: {grad}")
```

### Code 10.3

```python
with pm.Model() as model:
    z = pm.Normal("z", 0., 5.)
    x = pm.Normal("x", mu=z, sigma=1., observed=observed)

func = model.logp_dlogp_function()
func.set_extra_values({})
print(func(np.array([test_point])))
```

### Code 10.4

```python
def fraud_detector(fraud_observations, non_fraud_observations, fraud_prior=8, non_fraud_prior=6):
    """Conjugate Beta Binomial model for fraud detection"""
    expectation = (fraud_prior + fraud_observations) / (
        fraud_prior + fraud_observations + non_fraud_prior + non_fraud_observations)
    
    if expectation > .5:
        return {"suspend_card":True}

%timeit fraud_detector(2, 0)
```

## PPL Driven Transformations


### Code 10.9

```python
observed = np.repeat(2, 2)
pdf = stats.norm(0, 1).pdf(observed)
np.prod(pdf, axis=0)
```

### Code 10.10

```python
observed = np.repeat(2, 1000)
pdf = stats.norm(0, 1).pdf(observed)
np.prod(pdf, axis=0)
```

### Code 10.11

```python
1.2 - 1
```

```python
pdf[0], np.prod(pdf, axis=0)
```

### Code 10.12

```python
logpdf = stats.norm(0, 1).logpdf(observed)
np.log(pdf[0]), logpdf[0], logpdf.sum()
```

```python
np.log(pdf[0])
```

### Distribution Transforms


### Code 10.13

```python
lower, upper = -1, 2
domain = np.linspace(lower, upper, 5)
transform = np.log(domain - lower) - np.log(upper - domain)
print(f"Original domain: {domain}")
print(f"Transformed domain: {transform}")
```

### Code 10.14

```python
with pm.Model() as model:
    x = pm.Uniform("x", -1., 2.)
    
model.values_to_rvs
```

```python
model.varlogp_nojac.eval
```

```python
print(model.varlogp.eval({"x_interval__":-2}),
      model.varlogp_nojac.eval({"x_interval__":-2}))
print(model.varlogp.eval({"x_interval__":1}),
      model.varlogp_nojac.eval({"x_interval__":1}))
```

### Code 10.16

```python
import tensorflow_probability as tfp
tfd = tfp.distributions
```

```python
tfb = tfp.bijectors

lognormal0 = tfd.LogNormal(0., 1.)
lognormal1 = tfd.TransformedDistribution(tfd.Normal(0., 1.), tfb.Exp())
x = lognormal0.sample(100)

np.testing.assert_array_equal(lognormal0.log_prob(x), lognormal1.log_prob(x))
```

### Code 10.17

```python
y_observed = stats.norm(0, .01).rvs(20)

with pm.Model() as model_transform:
    sd = pm.HalfNormal("sd", 5)
    y = pm.Normal("y", mu=0, sigma=sd, observed=y_observed)
    idata_transform = pm.sample(chains=1, draws=100000)

print(model_transform.values_to_rvs)
print(f"Diverging: {idata_transform.sample_stats['diverging'].sum().item()}")
```

### Code 10.18

```python
with pm.Model() as model_no_transform:
    sd = pm.HalfNormal("sd", 5, transform=None)
    y = pm.Normal("y", mu=0, sigma=sd, observed=y_observed)
    idata_no_transform = pm.sample(chains=1, draws=100000)

print(model_no_transform.values_to_rvs)
print(f"Diverging: {idata_no_transform.sample_stats['diverging'].sum().item()}")
```

## Operation Graphs and Automatic Reparameterization

```python
x = 3
y = 1
x * y / x + 2
```

### Code 10.19

```python
pytensor.config.compute_test_value = 'ignore'
```

```python
x = pytensor.tensor.vector("x")
y = pytensor.tensor.vector("y")
out = x*(y/x) + 0
pytensor.printing.debugprint(out)
```

### Code 10.20

```python
fgraph = pytensor.function([x,y], [out])
pytensor.printing.debugprint(fgraph)
```

### Code 10.21

```python
fgraph([1],[3])
```

### Figure 10.1 and Figure 10.2

```python
pytensor.printing.pydotprint(
    out, outfile="img/chp10/symbolic_graph_unopt.png",
    var_with_name_simple=False, high_contrast=False, with_ids=True)
pytensor.printing.pydotprint(
    fgraph, 
    outfile="img/chp10/symbolic_graph_opt.png", 
    var_with_name_simple=False, high_contrast=False, with_ids=True)
```

### Code 10.22

```python
with pm.Model() as model_normal:
    x = pm.Normal("x", 0., 1.)
    
pytensor.printing.debugprint(model_normal.logp())
```

## Effect handling


### Code 10.23

```python
import jax
import numpyro
from tensorflow_probability.substrates import jax as tfp_jax

tfp_dist = tfp_jax.distributions
numpyro_dist = numpyro.distributions

root = tfp_dist.JointDistributionCoroutine.Root
def tfp_model():
    x = yield root(tfp_dist.Normal(loc=1.0, scale=2.0, name="x"))
    z = yield root(tfp_dist.HalfNormal(scale=1., name="z"))
    y = yield tfp_dist.Normal(loc=x, scale=z, name="y")
    
def numpyro_model():
    x = numpyro.sample("x", numpyro_dist.Normal(loc=1.0, scale=2.0))
    z = numpyro.sample("z", numpyro_dist.HalfNormal(scale=1.0))
    y = numpyro.sample("y", numpyro_dist.Normal(loc=x, scale=z))
```

```python
try:
    print(tfp_model())
except:
    pass
```

```python
try:
    print(numpyro_model())
except:
    pass
```

### Code 10.24

```python
sample_key = jax.random.PRNGKey(52346)

# Draw samples
jd = tfp_dist.JointDistributionCoroutine(tfp_model)
tfp_sample = jd.sample(1, seed=sample_key)

predictive = numpyro.infer.Predictive(numpyro_model, num_samples=1)
numpyro_sample = predictive(sample_key)

# Evaluate log prob
log_likelihood_tfp = jd.log_prob(tfp_sample)
log_likelihood_numpyro = numpyro.infer.util.log_density(
    numpyro_model, [], {},
    # Samples returning from JointDistributionCoroutine is a
    # Namedtuple like Python object, we convert it to a dictionary
    # so that numpyro can recognize it.
    params=tfp_sample._asdict())

# Validate that we get the same log prob
np.testing.assert_allclose(log_likelihood_tfp, log_likelihood_numpyro[0], rtol=1e-6)
```

### Code 10.25

```python
# Condition z to .01 in TFP and sample
jd.sample(z=.01, seed=sample_key)
```

```python
# Condition z to .01 in NumPyro and sample
predictive = numpyro.infer.Predictive(
    numpyro_model, num_samples=1, params={"z": np.asarray(.01)})
predictive(sample_key)
```

### Code 10.26

```python
# Conditioned z to .01 in TFP and construct conditional distributions
dist, value = jd.sample_distributions(z=.01, seed=sample_key)
assert dist.y.loc == value.x
assert dist.y.scale == value.z

# Conditioned z to .01 in NumPyro and construct conditional distributions
model = numpyro.handlers.substitute(numpyro_model, data={"z": .01})
with numpyro.handlers.seed(rng_seed=sample_key):
    # Under the seed context, the default behavior of a NumPyro model is the
    # same as in Pyro: drawing prior sample.
    model_trace = numpyro.handlers.trace(numpyro_model).get_trace()
assert model_trace["y"]["fn"].loc == model_trace["x"]["value"]
assert model_trace["y"]["fn"].scale == model_trace["z"]["value"]
```

## Designing a PPL


### Code 10.27

```python
import numpy as np
from scipy import stats

# Draw 2 samples from a Normal(1., 2.) distribution
x = stats.norm.rvs(loc=1.0, scale=2.0, size=2, random_state=1234)
# Evaluate the log probability of the samples 
logp = stats.norm.logpdf(x, loc=1.0, scale=2.0)
```

### Code 10.28

```python
random_variable_x = stats.norm(loc=1.0, scale=2.0)

x = random_variable_x.rvs(size=2, random_state=1234)
logp = random_variable_x.logpdf(x)
```

### Code 10.29

```python
x = stats.norm(loc=1.0, scale=2.0)
y = stats.norm(loc=x, scale=0.1)
y.rvs()
```

### Code 10.30

```python
class RandomVariable:
    def __init__(self, distribution):
        self.distribution = distribution

    def __array__(self):
        return np.asarray(self.distribution.rvs())

x = RandomVariable(stats.norm(loc=1.0, scale=2.0))
z = RandomVariable(stats.halfnorm(loc=0., scale=1.))
y = RandomVariable(stats.norm(loc=x, scale=z))

for i in range(5):
    print(np.asarray(y))
```

### Code 10.31

```python
class RandomVariable:
    def __init__(self, distribution, value=None):
        self.distribution = distribution
        self.set_value(value)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.__array__()})"

    def __array__(self, dtype=None):
        if self.value is None:
            return np.asarray(self.distribution.rvs(), dtype=dtype)
        return self.value

    def set_value(self, value=None):
        self.value = value
    
    def log_prob(self, value=None):
        if value is not None:
            self.set_value(value)
        return self.distribution.logpdf(np.array(self))

x = RandomVariable(stats.norm(loc=1.0, scale=2.0))
z = RandomVariable(stats.halfnorm(loc=0., scale=1.))
y = RandomVariable(stats.norm(loc=x, scale=z))
```

### Code 10.32

```python
for i in range(3):
    print(y)

print(f"  Set x=5 and z=0.1")
x.set_value(np.asarray(5))
z.set_value(np.asarray(0.05))
for i in range(3):
    print(y)

print(f"  Reset z")
z.set_value(None)
for i in range(3):
    print(y)
```

### Code 10.33

```python
# Observed y = 5.
y.set_value(np.array(5.))

posterior_density = lambda xval, zval: x.log_prob(xval) + z.log_prob(zval) + \
                y.log_prob()
posterior_density(np.array(0.), np.array(1.))
```

### Code 10.34

```python
def log_prob(xval, zval, yval=5):
    x_dist = stats.norm(loc=1.0, scale=2.0)
    z_dist = stats.halfnorm(loc=0., scale=1.)
    y_dist = stats.norm(loc=xval, scale=zval)
    return x_dist.logpdf(xval) + z_dist.logpdf(zval) + y_dist.logpdf(yval)

log_prob(0, 1)
```

### Code 10.35

```python
def prior_sample():
    x = stats.norm(loc=1.0, scale=2.0).rvs()
    z = stats.halfnorm(loc=0., scale=1.).rvs()
    y = stats.norm(loc=x, scale=z).rvs()
    return x, z, y

prior_sample()
```

### Shape handling


### Code 10.36

```python
def prior_sample(size):
    x = stats.norm(loc=1.0, scale=2.0).rvs(size=size)
    z = stats.halfnorm(loc=0., scale=1.).rvs(size=size)
    y = stats.norm(loc=x, scale=z).rvs()
    return x, z, y

print([x.shape for x in prior_sample(size=(2))])
print([x.shape for x in prior_sample(size=(2, 3, 5))])
```

### Code 10.37

```python
n_row, n_feature = 1000, 5
X = np.random.randn(n_row, n_feature)

def lm_prior_sample0():
    intercept = stats.norm(loc=0, scale=10.0).rvs()
    beta = stats.norm(loc=np.zeros(n_feature), scale=10.0).rvs()
    sigma = stats.halfnorm(loc=0., scale=1.).rvs()
    y_hat = X @ beta + intercept
    y = stats.norm(loc=y_hat, scale=sigma).rvs()
    return intercept, beta, sigma, y

def lm_prior_sample(size=10):
    if isinstance(size, int):
        size = (size,)
    else:
        size = tuple(size)
    intercept = stats.norm(loc=0, scale=10.0).rvs(size=size)
    beta = stats.norm(loc=np.zeros(n_feature), scale=10.0).rvs(
        size=size + (n_feature,))
    sigma = stats.halfnorm(loc=0., scale=1.).rvs(size=size)
    y_hat = np.squeeze(X @ beta[..., None]) + intercept[..., None]
    y = stats.norm(loc=y_hat, scale=sigma[..., None]).rvs()
    return intercept, beta, sigma, y
```

```python
print([x.shape for x in lm_prior_sample0()])
```

```python
print([x.shape for x in lm_prior_sample(size=())])
print([x.shape for x in lm_prior_sample(size=10)])
print([x.shape for x in lm_prior_sample(size=[10, 3])])
```

```python
# def lm_prior_sample2(size=10):
#     intercept = stats.norm(loc=0, scale=10.0).rvs(size=size)
#     beta = stats.multivariate_normal(
#         mean=np.zeros(n_feature), cov=10.0).rvs(size=size)
#     sigma = stats.halfnorm(loc=0., scale=1.).rvs(size=size)
#     y_hat = np.einsum('ij,...j->...i', X, beta) + intercept[..., None]
#     y = stats.norm(loc=y_hat, scale=sigma[..., None]).rvs()
#     return intercept, beta, sigma, y

# print([x.shape for x in lm_prior_sample2(size=())])
# print([x.shape for x in lm_prior_sample2(size=10)])
# print([x.shape for x in lm_prior_sample2(size=(10, 3))])
```

### Code 10.38

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

jd = tfd.JointDistributionSequential([
    tfd.Normal(0, 10),
    tfd.Sample(tfd.Normal(0, 10), n_feature),
    tfd.HalfNormal(1),
    lambda sigma, beta, intercept: tfd.Independent(
        tfd.Normal(
            loc=tf.einsum("ij,...j->...i", X, beta) + intercept[..., None],
            scale=sigma[..., None]),
        reinterpreted_batch_ndims=1,
        name="y")
])

print(jd)

n_sample = [3, 2]
for log_prob_part in jd.log_prob_parts(jd.sample(n_sample)):
    assert log_prob_part.shape == n_sample
```
