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

# Code 8: Approximate Bayesian Computation


```{admonition} This is a reference notebook for the book Bayesian Modeling and Computation in Python
:class: tip, dropdown
The textbook is not needed to use or run this code, though the context and explanation is missing from this notebook.

If you'd like a copy it's available
[from the CRC Press](https://www.routledge.com/Bayesian-Modeling-and-Computation-in-Python/Martin-Kumar-Lao/p/book/9780367894368)
or from [Amazon](https://www.routledge.com/Bayesian-Modeling-and-Computation-in-Python/Martin-Kumar-Lao/p/book/9780367894368).
``

```python
%matplotlib inline
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

from scripts.rf_selector import select_model
```

```python
az.style.use("arviz-grayscale")
plt.rcParams['figure.dpi'] = 300
np.random.seed(1346)
```

## Fitting a Gaussian the ABC-way


### Figure 8.2

```python
a = stats.norm(-2.5, 0.5)
b = stats.norm(2.5, 1)
c = stats.norm(0, 3)
x = np.linspace(-6, 6, 500)

lpdf = 0.65 * a.pdf(x) + 0.35* b.pdf(x)
ppdf = c.pdf(x)
_, ax = plt.subplots(figsize=(10, 4))
for c, β in zip(["#A8A8A8", "#585858", "#000000", "#2a2eec"],
                [0, 0.2, 0.5, 1]):
    post = ppdf * lpdf**β
    post /= post.sum()
    ax.plot(x, post, lw=3, label=f"β={β}", color=c)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlabel("θ")
ax.legend()
plt.savefig("img/chp08/smc_tempering.png")
```

## Fitting a Gaussian the ABC-way

```python
data = np.random.normal(loc=0, scale=1, size=1000)

def normal_sim(rng, a, b, size=1000):
    return rng.normal(a, b, size=size)
```

### Code 8.2 and Figure 8.3

```python
with pm.Model() as gauss:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Simulator('s', normal_sim, params=[μ, σ],
                     distance="gaussian",
                     sum_stat="sort",          
                     epsilon=1,
                     observed=data)
    trace_g = pm.sample_smc()
```

```python
az.summary(trace_g)
```

```python
az.plot_trace(trace_g, kind="rank_bars", figsize=(10, 4));
plt.savefig('img/chp08/trace_g.png')
```

## Choosing the Distance Function, $\epsilon$ and the Summary Statistics


### Codes 8.4, 8.5, 8.6, 8.7, and 8.8

```python
with pm.Model() as gauss_001:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Simulator('s', normal_sim, params=[μ, σ],
                     sum_stat="sort",
                     epsilon=0.1,
                     observed=data)
    idata_g_001 = pm.sample_smc()
    idata_g_001.extend(pm.sample_posterior_predictive(idata_g_001))

with pm.Model() as gauss_01:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Simulator('s', normal_sim, params=[μ, σ],
                     sum_stat="sort",
                     epsilon=1,
                     observed=data)
    idata_g_01 = pm.sample_smc()
    idata_g_01.extend(pm.sample_posterior_predictive(idata_g_01))

    
with pm.Model() as gauss_02:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Simulator('s', normal_sim, params=[μ, σ],
                     sum_stat="sort",
                     epsilon=2,
                     observed=data)
    idata_g_02 = pm.sample_smc()
    idata_g_02.extend(pm.sample_posterior_predictive(idata_g_02))

with pm.Model() as gauss_05:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Simulator('s', normal_sim, params=[μ, σ],
                     sum_stat="sort",
                     epsilon=5,
                     observed=data)
    idata_g_05 = pm.sample_smc()
    idata_g_05.extend(pm.sample_posterior_predictive(idata_g_05))
    
with pm.Model() as gauss_10:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Simulator('s', normal_sim, params=[μ, σ],
                     sum_stat="sort",
                     epsilon=10,
                     observed=data)
    idata_g_10 = pm.sample_smc()
    idata_g_10.extend(pm.sample_posterior_predictive(idata_g_10))


with pm.Model() as gauss_NUTS:
    μ = pm.Normal('μ', mu=0, sigma=1)
    σ = pm.HalfNormal('σ', sigma=1)
    s = pm.Normal('s', μ, σ, observed=data)
    idata_g_nuts = pm.sample()
```

```python
idatas = [idata_g_nuts, idata_g_01, idata_g_05, idata_g_10]
az.plot_forest(idatas, model_names=["NUTS", "ϵ 1", "ϵ 5", "ϵ 10"],
               colors=["#2a2eec", "#000000", "#585858", "#A8A8A8"],
               figsize=(8, 3));
plt.savefig("img/chp08/trace_g_many_eps.png")
```

```python
az.plot_trace(idata_g_001, kind="rank_bars", figsize=(10, 4));
plt.savefig("img/chp08/trace_g_eps_too_low.png")
```

```python
idatas_ = [idata_g_001, idata_g_01, idata_g_05, idata_g_10]
epsilons = [0.1, 1, 5, 10]

_, axes = plt.subplots(2, 2, figsize=(10,5))

for i, ax in enumerate(axes.ravel()):
    pp_vals = idatas_[i].posterior_predictive["s"].values.reshape(8000, -1)
    tstat_pit = np.mean(pp_vals <= data, axis=0)
    _, tstat_pit_dens = az.kde(tstat_pit)

    ax.axhline(1, color="w")
    az.plot_bpv(idatas_[i], kind="u_value", ax=ax, reference="analytical")
    ax.tick_params(axis='both', pad=7)
    ax.set_title(f"ϵ={epsilons[i]}, mse={np.mean((1 - tstat_pit_dens)**2) * 100:.2f}")

plt.savefig("img/chp08/bpv_g_many_eps_00.png")
```

```python
_, ax = plt.subplots(2, 2, figsize=(10,5))

ax = ax.ravel()
for i in range(4):
    az.plot_bpv(idatas_[i], kind="p_value", reference='samples', color="C4", ax=ax[i],
               plot_ref_kwargs={"color":"C2"})
    ax[i].set_title(f"ϵ={epsilons[i]}")
plt.savefig("img/chp08/bpv_g_many_eps_01.png")
```

```python
_, axes = plt.subplots(2, 2, figsize=(10,5))

for i, ax in enumerate(axes.ravel()):
    az.plot_ppc(idatas_[i], num_pp_samples=100, ax=ax, color="C2",
                mean=False, legend=False, observed=False)
    az.plot_kde(idatas_[i].observed_data["s"].values, plot_kwargs={"color":"C4"}, ax=ax)
    ax.set_xlabel("s")
    ax.set_title(f"ϵ={epsilons[i]}")
plt.savefig("img/chp08/ppc_g_many_eps.png")
```

## g-and-k distributions


### Figure 8.9

```python
data = pd.read_csv("../data/air_pollution_bsas.csv")
bsas_co = data["co"].dropna().values
```

```python
_, axes = plt.subplots(2,1,  figsize=(10,4), sharey=True)
axes[0].hist(bsas_co, bins="auto", color="C1", density=True)
axes[0].set_yticks([])
axes[1].hist(bsas_co[bsas_co < 3], bins="auto", color="C1", density=True)
axes[1].set_yticks([])
axes[1].set_xlabel("CO levels (ppm)")
plt.savefig("img/chp08/co_ppm_bsas.png")
f"We have {sum(bsas_co > 3)} observations larger than 3 ppm"
```

### Code 8.4 and Figure 8.10

```python
class g_and_k_quantile:
    def __init__(self):
        self.quantile_normal = stats.norm(0, 1).ppf
        self.pdf_normal = stats.norm(0, 1).pdf

    def ppf(self, x, a, b, g, k):
        z = self.quantile_normal(x)
        return a + b * (1 + 0.8 * np.tanh(g*z/2)) * ((1 + z**2)**k) * z

    
    def rvs(self, rng, a, b, g, k, size):
        x = rng.uniform(0, 1, size)
        return self.ppf(x, a, b, g, k)

    def cdf(self, x, a, b, g, k, zscale=False):   
        optimize.fminbound(f, -5, 5)

    def pdf(self, x, a, b, g, k):
        #z = cdf(x, a, b, g, k)
        z = x
        z_sq = z**2
        term1 = (1+z_sq)**k
        term2 = 1+0.8*np.tanh(g*x/2)
        term3 = (1+(2*k+1)*z_sq)/(1+z_sq)
        term4 = 0.8*g*z/(2*np.cosh(g*z/2)**2)

        deriv = b*term1*(term2*term3+term4)
        return self.pdf_normal(x) / deriv
```

```python
gk = g_and_k_quantile()
u = np.linspace(1E-14, 1-1E-14, 10000)

params = ((0, 1, 0, 0), 
 (0, 1, .4, 0),
 (0, 1,-.4, 0),
 (0, 1, 0, 0.25))

_, ax = plt.subplots(2, 4, sharey="row", figsize=(10, 5))
for i, p in enumerate(params):
    a, b, g, k = p
    ppf = gk.ppf(u, a, b, g, k)
    ax[0, i].plot(u, ppf)
    ax[0, i].set_title(f"a={a}, b={b},\ng={g}, k={k}")
    #ax[1, i].plot(x, gk.pdf(x, a, b, g, k))
    az.plot_kde(ppf, ax=ax[1, i], bw=0.5)
plt.savefig("img/chp08/gk_quantile.png")
```

### Code 8.5

```python
def octo_summary(x):
    e1, e2, e3, e4, e5, e6, e7 = np.quantile(x, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    sa = e4
    sb = e6 - e2
    sg = (e6 + e2 - 2*e4)/sb
    sk = (e7 - e5 + e3 - e1)/sb
    return np.array([sa, sb, sg, sk])
```

### Code 8.6

```python
gk = g_and_k_quantile()
def gk_simulator(rng, a, b, g, k, size=None):
    return gk.rvs(rng, a, b, g, k, len(bsas_co))
```

### Code 8.7 and Figure 8.11

```python
with pm.Model() as gkm:
    a = pm.HalfNormal('a', sigma=1)
    b = pm.HalfNormal('b', sigma=1)
    g = pm.HalfNormal('g', sigma=1)
    k = pm.HalfNormal('k', sigma=1)
    
    s = pm.Simulator('s', gk_simulator, params=[a, b, g, k],        
                     sum_stat=octo_summary,
                     epsilon=0.1,
                     observed=bsas_co)
    
    idata_gk = pm.sample_smc()
    idata_gk.extend(pm.sample_posterior_predictive(idata_gk))
```

```python
az.summary(idata_gk)
```

```python
az.plot_trace(idata_gk, kind="rank_bars")
plt.savefig("img/chp08/trace_gk.png")
```

```python
axes = az.plot_pair(idata_gk, 
                    kind="kde", 
                    marginals=True,
                    textsize=45,
                    kde_kwargs={"contourf_kwargs":{"cmap":plt.cm.viridis}},
                    )

for ax, pad in zip(axes[:,0], (70, 30, 30, 30)):
    ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=pad)

plt.savefig("img/chp08/pair_gk.png")
```

## Approximating moving averages


### Code 8.8 and Figure 8.12

```python
def moving_average_2(θ1, θ2, n_obs=500):
    λ = np.random.normal(0, 1, n_obs+2)
    y = λ[2:] + θ1*λ[1:-1] + θ2*λ[:-2]
    return y
```

We are calling the simulator one more time to generate "observed data".

```python
θ1_true = 0.6
θ2_true = 0.2
y_obs = moving_average_2(θ1_true, θ2_true)
```

```python
az.plot_trace({'one sample':moving_average_2(θ1_true, θ2_true),
               'another sample':moving_average_2(θ1_true, θ2_true)},
              trace_kwargs={'alpha':1},
              figsize=(10, 4)
             )
plt.savefig("img/chp08/ma2_simulator_abc.png")
```

### Code 8.9

```python
def autocov(x):
    a = np.mean(x[1:] * x[:-1])
    b = np.mean(x[2:] * x[:-2])
    return np.array((a, b))


def moving_average_2(rng, θ1, θ2, size=500):
    λ = rng.normal(0, 1, size[0]+2)
    y = λ[2:] + θ1*λ[1:-1] + θ2*λ[:-2]
    return y
```

### Code 8.10 and Figure 8.13

```python
with pm.Model() as model_ma2:
    θ1 = pm.Uniform('θ1', -2, 2)
    θ2 = pm.Uniform('θ2', -1, 1)
    p1 = pm.Potential("p1", pm.math.switch(θ1+θ2 > -1, 0, -np.inf))
    p2 = pm.Potential("p2", pm.math.switch(θ1-θ2 < 1, 0, -np.inf))

    y = pm.Simulator('y', moving_average_2,
                     params=[θ1, θ2],
                     sum_stat=autocov,
                     epsilon=0.1,
                     observed=y_obs)

    trace_ma2 = pm.sample_smc(3000)
```

```python
az.summary(trace_ma2)
```

```python
az.plot_trace(trace_ma2, kind="rank_bars", figsize=(10, 4))
plt.savefig("img/chp08/ma2_trace.png")
```

```python
axes = az.plot_pair(trace_ma2, kind="kde", var_names=["θ1", "θ2"],
                    marginals=True, figsize=(10,5),
                    kde_kwargs={"contourf_kwargs":{"cmap":plt.cm.viridis}},
                    point_estimate="mean",
                    point_estimate_kwargs={"ls":"none"},
                    point_estimate_marker_kwargs={"marker":".",
                                                  "facecolor":"k",
                                                  "zorder":2})

axes[1,0].set_xlim(-2.1, 2.1)
axes[1,0].set_ylim(-1.1, 1.1)
axes[1,0].set_ylabel(axes[1,0].get_ylabel(), rotation=0)
axes[1,0].plot([0, 2, -2, 0], [-1, 1, 1, -1], "C2", lw=2)
plt.savefig("img/chp08/ma2_triangle.png")
```

## Model Comparison in the ABC context

To reproduce the figures in the book, run `loo_abc.py`


##  Model choice via random forest

```python
def moving_average_1(rng, θ1, size=(500,)):
    λ = rng.normal(0, 1, size[0]+1)
    y = λ[2:] + θ1*λ[1:-1]
    return y

def moving_average_2(rng, θ1, θ2, size=(500,)):
    λ = rng.normal(0, 1, size[0]+2)
    y = λ[2:] + θ1*λ[1:-1] + θ2*λ[1:-1]
    return y

rng = np.random.default_rng(1346)
θ1_true = 0.7
θ2_true = 0.3
y_obs = moving_average_2(rng, θ1_true, θ2_true)
```

```python
def autocov(x, n=2):
    return np.array([np.mean(x[i:] * x[:-i]) for i in range(1, n+1)])
```

### Code 8.12

```python
with pm.Model() as model_ma1:
    θ1 = pm.Uniform('θ1', -2, 2)
    y = pm.Simulator('y', moving_average_1,
                     params=[θ1], sum_stat=autocov, epsilon=0.2, observed=y_obs)
    idata_ma1 = pm.sample_smc(3000, idata_kwargs={"log_likelihood":True})
```

```python
with pm.Model() as model_ma2:
    θ1 = pm.Uniform('θ1', -2, 2)
    θ2 = pm.Uniform('θ2', -1, 1)
    p1 = pm.Potential("p1", pm.math.switch(θ1+θ2 > -1, 0, -np.inf))
    p2 = pm.Potential("p2", pm.math.switch(θ1-θ2 < 1, 0, -np.inf))

    y = pm.Simulator('y', moving_average_2,
                     params=[θ1, θ2],
                     sum_stat=autocov,
                     epsilon=0.1,
                     observed=y_obs)

    idata_ma2 = pm.sample_smc(3000, idata_kwargs={"log_likelihood":True})
```

```python
mll_ma2 = np.nanmean(np.concatenate([np.hstack(v) for v in idata_ma2.sample_stats.log_marginal_likelihood.values]))
mll_ma1 = np.nanmean(np.concatenate([np.hstack(v) for v in idata_ma1.sample_stats.log_marginal_likelihood.values]))

mll_ma2/mll_ma1
```

### Code 8.13

```python
cmp = az.compare({"model_ma1":idata_ma1, "model_ma2":idata_ma2})
cmp
```

### Code 8.14

```python
from functools import partial
select_model([(model_ma1, idata_ma1), (model_ma2, idata_ma2)], 
             statistics=[partial(autocov, n=6)],
             n_samples=10000,
             observations=y_obs)
```
