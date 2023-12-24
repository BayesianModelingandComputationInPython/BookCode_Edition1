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

# Code 4: Extending Linear Models


```{admonition} This is a reference notebook for the book Bayesian Modeling and Computation in Python
:class: tip, dropdown
The textbook is not needed to use or run this code, though the context and explanation is missing from this notebook.

If you'd like a copy it's available
[from the CRC Press](https://www.routledge.com/Bayesian-Modeling-and-Computation-in-Python/Martin-Kumar-Lao/p/book/9780367894368)
or from [Amazon](https://www.routledge.com/Bayesian-Modeling-and-Computation-in-Python/Martin-Kumar-Lao/p/book/9780367894368).
``

```python
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
```

```python
az.style.use("arviz-grayscale")
plt.rcParams['figure.dpi'] = 300 
```

## Transforming Covariates


### Code 4.1

```python
babies = pd.read_csv('../data/babies.csv')

# Add a constant term so we can use a the dot product approach
babies["Intercept"] = 1

babies.head()
```

### Figure 4.1

```python
fig, ax = plt.subplots()

ax.plot(babies["Month"], babies["Length"], 'C0.', alpha=0.1)
ax.set_ylabel("Length")
ax.set_xlabel("Month");
plt.savefig('img/chp04/baby_length_scatter.png', dpi=300)
```

### Code 4.2

```python
with pm.Model() as model_baby_linear:
    β = pm.Normal('β', sigma=10, shape=2)
    
    # Use dot product instead of expanded multiplication
    μ = pm.Deterministic("μ", pm.math.dot(babies[["Intercept", "Month"]], β))
    ϵ = pm.HalfNormal("ϵ", sigma=10)

    length = pm.Normal("length", mu=μ, sigma=ϵ, observed=babies["Length"])

    idata_linear = pm.sample(draws=2000, tune=4000, idata_kwargs={"log_likelihood": True})
    idata_linear.extend(pm.sample_posterior_predictive(idata_linear))
```

### Figure 4.2

```python
fig, ax = plt.subplots()

ax.set_ylabel("Length")
ax.set_xlabel("Month");

μ_m = idata_linear.posterior["μ"].mean(("chain", "draw"))

ax.plot(babies["Month"], μ_m, c='C4')
az.plot_hdi(babies["Month"], idata_linear.posterior_predictive["length"], hdi_prob=.50, ax=ax)
az.plot_hdi(babies["Month"], idata_linear.posterior_predictive["length"], hdi_prob=.94, ax=ax)

ax.plot(babies["Month"], babies["Length"], 'C0.', alpha=0.1)

plt.savefig('img/chp04/baby_length_linear_fit.png', dpi=300)
```

```python
az.loo(idata_linear)
```

### Code 4.3

```python
with pm.Model() as model_baby_sqrt:
    β = pm.Normal("β", sigma=10, shape=2)

    μ = pm.Deterministic("μ", β[0] + β[1] * np.sqrt(babies["Month"]))
    σ = pm.HalfNormal("σ", sigma=10)

    length = pm.Normal("length", mu=μ, sigma=σ, observed=babies["Length"])
    idata_sqrt = pm.sample(draws=2000, tune=4000, idata_kwargs={"log_likelihood": True})
    idata_sqrt.extend(pm.sample_posterior_predictive(idata_sqrt))

```

```python
fig, ax = plt.subplots()

ax.plot(babies["Month"], babies["Length"], 'C0.', alpha=0.1)

ax.set_ylabel("Length")
ax.set_xlabel("Month");

μ_m = idata_sqrt.posterior["μ"].mean(("chain", "draw"))

az.plot_hdi(babies["Month"], idata_sqrt.posterior_predictive["length"], hdi_prob=.50, ax=ax)
az.plot_hdi(babies["Month"], idata_sqrt.posterior_predictive["length"], hdi_prob=.94, ax=ax)

ax.plot(babies["Month"], μ_m, c='C4')

plt.savefig('img/chp04/baby_length_sqrt_fit.png', dpi=300)
```

### Figure 4.3

```python
fig, axes = plt.subplots(1,2)
axes[0].plot(babies["Month"], babies["Length"], 'C0.', alpha=0.1)

μ_m = idata_sqrt.posterior["μ"].mean(("chain", "draw"))

axes[0].plot(babies["Month"], μ_m, c='C4')
az.plot_hdi(babies["Month"], idata_sqrt.posterior_predictive["length"], hdi_prob=.50, ax=axes[0])
az.plot_hdi(babies["Month"], idata_sqrt.posterior_predictive["length"], hdi_prob=.94, ax=axes[0])

axes[0].set_ylabel("Length")
axes[0].set_xlabel("Month");

axes[1].plot(np.sqrt(babies["Month"]), babies["Length"], 'C0.', alpha=0.1)
axes[1].set_xlabel("Square Root of Month");

az.plot_hdi(np.sqrt(babies["Month"]), idata_sqrt.posterior_predictive["length"], hdi_prob=.50, ax=axes[1])
az.plot_hdi(np.sqrt(babies["Month"]), idata_sqrt.posterior_predictive["length"], hdi_prob=.94, ax=axes[1])
axes[1].plot(np.sqrt(babies["Month"]), μ_m, c='C4')

axes[1].set_yticks([])
axes[1]

plt.savefig('img/chp04/baby_length_sqrt_fit.png', dpi=300)
```

```python
az.compare({"Linear Model":idata_linear,
            "Non Linear Model":idata_sqrt})
```

## Varying Uncertainty


### Code 4.4

```python
with pm.Model() as model_baby_vv:
    β = pm.Normal("β", sigma=10, shape=2)

    # Additional variance terms
    δ = pm.HalfNormal("δ", sigma=10, shape=2)

    μ = pm.Deterministic("μ", β[0] + β[1] * np.sqrt(babies["Month"]))
    σ = pm.Deterministic("σ", δ[0] + δ[1] * babies["Month"])

    length = pm.Normal("length", mu=μ, sigma=σ, observed=babies["Length"])

    idata_baby_vv = pm.sample(2000, target_accept=.95)
    idata_baby_vv.extend(pm.sample_posterior_predictive(idata_baby_vv))
```

```python
az.summary(idata_baby_vv, var_names=["δ"])
```

```python
fig, ax = plt.subplots()

ax.set_ylabel("Length")
ax.set_xlabel("Month");

ax.plot(babies["Month"], babies["Length"], 'C0.', alpha=0.1)

μ_m = idata_baby_vv.posterior["μ"].mean(("chain", "draw"))

ax.plot(babies["Month"], μ_m, c='C4')

az.plot_hdi(babies["Month"], idata_baby_vv.posterior_predictive["length"], hdi_prob=.50, ax=ax)
az.plot_hdi(babies["Month"], idata_baby_vv.posterior_predictive["length"], hdi_prob=.94, ax=ax)

plt.savefig('img/chp04/baby_length_sqrt_vv_fit.png', dpi=300)
```

### Figure 4.4

```python
fig, axes = plt.subplots(2,1)

axes[0].plot(babies["Month"], babies["Length"], 'C0.', alpha=0.1)

μ_m = idata_baby_vv.posterior["μ"].mean(("chain", "draw"))

axes[0].plot(babies["Month"], μ_m, c='C4')

az.plot_hdi(babies["Month"], idata_baby_vv.posterior_predictive["length"], hdi_prob=.50, ax=axes[0])
az.plot_hdi(babies["Month"], idata_baby_vv.posterior_predictive["length"], hdi_prob=.94, ax=axes[0])
axes[0].set_ylabel("Length")

σ_m = idata_baby_vv.posterior["σ"].mean(("chain", "draw"))

axes[1].plot(babies["Month"], σ_m, c='C1')

axes[1].set_ylabel("σ")
axes[1].set_xlabel("Month")

axes[0].set_xlim(0,24)
axes[1].set_xlim(0,24)

plt.savefig('img/chp04/baby_length_sqrt_vv_fit_include_error.png', dpi=300)
```

## Interaction effects


### Code 4.5

```python
tips_df = pd.read_csv('../data/tips.csv')
tips_df.head()
```

```python
tips = tips_df["tip"]
total_bill_c = (tips_df["total_bill"] - tips_df["total_bill"].mean())  
smoker = pd.Categorical(tips_df["smoker"]).codes

with pm.Model() as model_no_interaction:
    β = pm.Normal("β", mu=0, sigma=1, shape=3)
    σ = pm.HalfNormal("σ", 1)

    μ = (β[0] +
         β[1] * total_bill_c + 
         β[2] * smoker)

    obs = pm.Normal("obs", μ, σ, observed=tips)
    idata_no_interaction = pm.sample(1000, tune=1000)
```

```python
idata_no_interaction.posterior
```

```python
_, ax = plt.subplots(figsize=(8, 4.5))

total_bill_c_da = xr.DataArray(total_bill_c)

posterior_no_interaction = az.extract(idata_no_interaction, var_names=["β"])

β0_nonint = posterior_no_interaction.sel(β_dim_0=0)
β1_nonint = posterior_no_interaction.sel(β_dim_0=1)
β2_nonint = posterior_no_interaction.sel(β_dim_0=2)

pred_y_non_smokers = β0_nonint + β1_nonint * total_bill_c_da
pred_y_smokers = β0_nonint + β1_nonint * total_bill_c_da + β2_nonint

ax.scatter(total_bill_c[smoker==0], tips[smoker==0], label='non-smokers', marker='.')
ax.scatter(total_bill_c[smoker==1], tips[smoker==1], label='smokers', marker='.', c="C4")
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
ax.legend()

ax.plot(total_bill_c, pred_y_non_smokers.mean("sample"), lw=2)
ax.plot(total_bill_c, pred_y_smokers.mean("sample"), lw=2, c="C4");
```

### Code 4.6

```python
with pm.Model() as model_interaction:
    β = pm.Normal('β', mu=0, sigma=1, shape=4)
    σ = pm.HalfNormal('σ', 1)

    μ = (β[0] +
         β[1] * total_bill_c + 
         β[2] * smoker +
         β[3] * smoker * total_bill_c
        )

    obs = pm.Normal('obs', μ, σ, observed=tips)
    idata_interaction = pm.sample(1000, tune=1000)
```

### Figure 4.5

```python
_, ax = plt.subplots(1, 2, figsize=(8, 4.5))

posterior_no_interaction = az.extract(idata_no_interaction, var_names=["β"])

β0_nonint = posterior_no_interaction.sel(β_dim_0=0)
β1_nonint = posterior_no_interaction.sel(β_dim_0=1)
β2_nonint = posterior_no_interaction.sel(β_dim_0=2)

pred_y_non_smokers = β0_nonint + β1_nonint * total_bill_c_da
pred_y_smokers = β0_nonint + β1_nonint * total_bill_c_da + β2_nonint

ax[0].scatter(total_bill_c[smoker==0], tips[smoker==0], label='non-smokers', marker='.')
ax[0].scatter(total_bill_c[smoker==1], tips[smoker==1], label='smokers', marker='.', c="C4")
ax[0].set_xlabel('Total Bill (Centered)')
ax[0].set_ylabel('Tip')
ax[0].legend(frameon=True)

ax[0].plot(total_bill_c, pred_y_non_smokers.mean("sample"), lw=2)
ax[0].plot(total_bill_c, pred_y_smokers.mean("sample"), lw=2, c="C4")
ax[0].set_title('No Interaction')


az.plot_hdi(total_bill_c, pred_y_non_smokers, color='C0', ax=ax[0])
az.plot_hdi(total_bill_c, pred_y_smokers, ax=ax[0], color="C4");


posterior_interaction = az.extract(idata_interaction, var_names=["β"])

β0_int = posterior_interaction.sel(β_dim_0=0)
β1_int = posterior_interaction.sel(β_dim_0=1)
β2_int = posterior_interaction.sel(β_dim_0=2)
β3_int = posterior_interaction.sel(β_dim_0=3)


# Because smoker=0 I am omitting the terms including the smoker covariate
pred_y_non_smokers = (β0_int +
                      β1_int * total_bill_c_da)

# Because x1=1 I am ommiting x1
pred_y_smokers = (β0_int +
                  β1_int * total_bill_c_da +
                  β2_int +
                  β3_int * total_bill_c_da)


ax[1].scatter(total_bill_c[smoker==0], tips[smoker==0], label='non-smokers', marker='.')
ax[1].scatter(total_bill_c[smoker==1], tips[smoker==1], label='smokers', marker='.', c="C4")
ax[1].set_xlabel('Total Bill (Centered)')
ax[1].set_yticks([])

ax[1].set_title('Interaction')

ax[1].plot(total_bill_c, pred_y_non_smokers.mean("sample"), lw=2)
ax[1].plot(total_bill_c, pred_y_smokers.mean("sample"), lw=2)
az.plot_hdi(total_bill_c, pred_y_non_smokers, color='C0', ax=ax[1])
az.plot_hdi(total_bill_c, pred_y_smokers, ax=ax[1], color="C4");

plt.savefig('img/chp04/smoker_tip_interaction.png', dpi=300)
```

## Robust Regression


### Figure 4.6

```python
mean = 5
sigma = 2

x = np.linspace(-5, 15, 1000)
fig, ax = plt.subplots()

ax.plot(x, stats.norm(5,2).pdf(x), label=f"Normal μ={mean}, σ={sigma}", color="C4")

for i, nu in enumerate([1, 2, 20],1):
    ax.plot(x, stats.t(loc=5, scale=2, df=nu).pdf(x), label=f"Student T μ={mean}, σ={sigma}, ν={nu}", color=f"C{i}")

ax.set_xlim(-5, 18)
ax.legend(loc="upper right", frameon=False)
ax.set_yticks([])
plt.savefig('img/chp04/studentt_normal_comparison.png', dpi=300)
```

### Figure 4.7

```python
def generate_sales(*, days, mean, std, label):
    np.random.seed(0)
    df = pd.DataFrame(index=range(1, days+1), columns=["customers", "sales"])
    for day in range(1, days+1):
        num_customers = stats.randint(30, 100).rvs()+1
        
        # This is correct as there is an independent draw for each customers orders
        dollar_sales = stats.norm(mean, std).rvs(num_customers).sum()
        
        df.loc[day, "customers"] = num_customers
        df.loc[day, "sales"] = dollar_sales
        
    # Fix the types as not to cause Theano errors
    df = df.astype({'customers': 'int32', 'sales': 'float32'})
    
    # Sorting will make plotting the posterior predictive easier later
    df["Food_Category"] = label
    df = df.sort_values("customers")
    return df

```

```python
fig, ax = plt.subplots()

empanadas =  generate_sales(days=200, mean=180, std=30, label="Empanada")
empanadas.iloc[0] = [50, 92000, "Empanada"]
empanadas.iloc[1] = [60, 90000, "Empanada"]
empanadas.iloc[2] = [70, 96000, "Empanada"]
empanadas.iloc[3] = [80, 91000, "Empanada"]
empanadas.iloc[4] = [90, 99000, "Empanada"]

empanadas = empanadas.sort_values("customers")

empanadas.sort_values("sales")[:-5].plot(x="customers", y="sales", kind="scatter", ax=ax);
empanadas.sort_values("sales")[-5:].plot(x="customers", y="sales", kind="scatter", c="C4", ax=ax);

ax.set_ylabel("Argentine Peso")
ax.set_xlabel("Customer Count")
ax.set_title("Empanada Sales")
plt.savefig('img/chp04/empanada_scatter_plot.png', dpi=300)
```

### Code 4.7

```python
with pm.Model() as model_non_robust:
    σ = pm.HalfNormal("σ", 50)
    β = pm.Normal('β', mu=150, sigma=20)

    μ = pm.Deterministic("μ", β * empanadas["customers"])
    
    sales = pm.Normal("sales", mu=μ, sigma=σ, observed=empanadas["sales"])
    
    idata_non_robust = pm.sample(random_seed=1, idata_kwargs={"log_likelihood": True})
    idata_non_robust.extend(pm.sample_posterior_predictive(idata_non_robust))
```

### Figure 4.8

```python
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
μ_m = idata_non_robust.posterior["μ"].mean(("chain", "draw"))

for i in range(2):
    empanadas.sort_values("sales")[:-5].plot(x="customers", y="sales", kind="scatter", ax=axes[i]);
    empanadas.sort_values("sales")[-5:].plot(x="customers", y="sales", kind="scatter", c="C4", ax=axes[i]);
    axes[i].plot(empanadas.customers, μ_m, c='C4')
    az.plot_hdi(empanadas.customers, idata_non_robust.posterior_predictive["sales"], hdi_prob=.95, ax=axes[i])

    axes[1].set_ylabel("Argentine Peso")

axes[0].set_ylabel("")
axes[1].set_xlabel("Customer Count")
axes[1].set_ylim(400, 25000);
plt.savefig('img/chp04/empanada_scatter_non_robust.png', dpi=300)
```

### Table 4.1

```python
az.summary(idata_non_robust, kind="stats", var_names=["β", "σ"]).round(1)
```

```python
with pm.Model() as model_robust:
    σ = pm.HalfNormal("σ", 50)
    β = pm.Normal("β", mu=150, sigma=20)
    ν = pm.HalfNormal("ν", 20)

    μ = pm.Deterministic("μ", β * empanadas["customers"])
    
    sales = pm.StudentT("sales", mu=μ, sigma=σ, nu=ν,
                        observed=empanadas["sales"])
        
    idata_robust  = pm.sample(random_seed=0, idata_kwargs={"log_likelihood": True})
    idata_robust.extend(pm.sample_posterior_predictive(idata_robust))
```

### Table 4.2

```python
az.summary(idata_robust, var_names=["β", "σ", "ν"], kind="stats").round(1)
```

### Figure 4.9

```python
fig, ax = plt.subplots(figsize=(10, 6))
μ_m = idata_robust.posterior["μ"].mean(("chain", "draw"))
    
ax.plot(empanadas.customers, μ_m, c='C4')
az.plot_hdi(empanadas.customers, idata_robust.posterior_predictive["sales"], hdi_prob=.95, ax=ax)

empanadas.plot(x="customers", y="sales", kind="scatter", ax=ax)
ax.set_ylim(4000, 20000);
ax.set_ylabel("Argentine Peso")
ax.set_xlabel("Customer Count")
ax.set_title("Empanada Sales with Robust Regression Fit")
plt.savefig('img/chp04/empanada_scatter_robust.png', dpi=300)
```

```python
az.compare({"Non robust": idata_non_robust,
            "Robust":idata_robust})
```

## Pooling, Multilevel Models, and Mixed Effects

```python
def generate_sales(*, days, mean, std, label):
    np.random.seed(0)
    df = pd.DataFrame(index=range(1, days+1), columns=["customers", "sales"])
    for day in range(1, days+1):
        num_customers = stats.randint(30, 100).rvs()+1
        
        # This is correct as there is an independent draw for each customers orders
        dollar_sales = stats.norm(mean, std).rvs(num_customers).sum()
        
        df.loc[day, "customers"] = num_customers
        df.loc[day, "sales"] = dollar_sales
        
    # Fix the types as not to cause Theano errors
    df = df.astype({'customers': 'int32', 'sales': 'float32'})
    
    # Sorting will make plotting the posterior predictive easier later
    df["Food_Category"] = label
    df = df.sort_values("customers")
    return df

```

```python
pizza_df = generate_sales(days=365, mean=13, std=5, label="Pizza")
sandwich_df = generate_sales(days=100, mean=6, std=5, label="Sandwich")

salad_days = 3
salad_df = generate_sales(days=salad_days, mean=8 ,std=3, label="Salad")

salad_df.plot(x="customers", y="sales", kind="scatter");
```

```python
sales_df = pd.concat([pizza_df, sandwich_df, salad_df]).reset_index(drop=True)
sales_df["Food_Category"] = pd.Categorical(sales_df["Food_Category"])
sales_df
```

### Figure 4.10

```python
fig, ax = plt.subplots()
pizza_df.plot(x="customers", y="sales", kind="scatter", ax=ax, c="C1", label="Pizza", marker="^", s=60);
sandwich_df.plot(x="customers", y="sales", kind="scatter", ax=ax,  label="Sandwich", marker="s");
salad_df.plot(x="customers", y="sales", kind="scatter", ax=ax, label="Salad", c="C4");

ax.set_xlabel("Number of Customers")
ax.set_ylabel("Daily Sales Dollars")
ax.set_title("Aggregated Sales Dollars")
ax.legend()

plt.savefig('img/chp04/restaurant_order_scatter.png', dpi=300)
```

### Unpooled Model


### Code 4.9

```python
customers = sales_df.loc[:, "customers"].values
sales_observed = sales_df.loc[:, "sales"].values
food_category = pd.Categorical(sales_df["Food_Category"])

coords = {"meals":food_category.categories}

with pm.Model(coords=coords) as model_sales_unpooled:
    σ = pm.HalfNormal("σ", 20, dims="meals")
    β = pm.Normal("β", mu=10, sigma=10, dims="meals")
    
    μ = pm.Deterministic("μ", β[food_category.codes] *customers)
    
    sales = pm.Normal("sales", mu=μ, sigma=σ[food_category.codes],
                      observed=sales_observed)
    
    idata_sales_unpooled = pm.sample(target_accept=.9)
```

### Figure 4.12

```python
sales_unpooled_diagram = pm.model_to_graphviz(model_sales_unpooled)
sales_unpooled_diagram.render("img/chp04/salad_sales_basic_regression_model_unpooled", format="png", cleanup=True)
sales_unpooled_diagram
```

```python
idata_salads_sales_unpooled = idata_sales_unpooled.posterior.sel(meals="Salad", μ_dim_0=slice(465, 467))
```

```python
az.summary(idata_sales_unpooled, var_names=["β", "σ"])
```

```python
az.plot_trace(idata_sales_unpooled, var_names=["β", "σ"], compact=False);
```

### Figure 4.13

```python
axes = az.plot_forest([idata_sales_unpooled],
                      model_names = ["Unpooled",],
                      var_names=["β"], combined=True, figsize=(7, 1.8));
axes[0].set_title("β parameter estimates 94% HDI")
plt.savefig("img/chp04/salad_sales_basic_regression_forestplot_beta.png")
```

### Figure 4.14

```python
axes = az.plot_forest([idata_sales_unpooled],
                      model_names = ["Unpooled",],
                      var_names=["σ"], combined=True, figsize=(7, 1.8));
axes[0].set_title("σ parameter estimates 94% HDI")
plt.savefig("img/chp04/salad_sales_basic_regression_forestplot_sigma.png")
```

### Pooled Model


### Code 4.10

```python
with pm.Model() as model_sales_pooled:
    σ = pm.HalfNormal("σ", 20)
    β = pm.Normal("β", mu=10, sigma=10)

    μ = pm.Deterministic("μ", β * customers)
    
    sales = pm.Normal("sales", mu=μ, sigma=σ,
                      observed=sales_observed)
                        
    idata_sales_pooled = pm.sample()
```

```python
with model_sales_pooled:
    idata_sales_pooled.extend(pm.sample_posterior_predictive(idata_sales_pooled))
```

### Figure 4.16

```python
pooled_sales_diagram = pm.model_to_graphviz(model_sales_pooled)
pooled_sales_diagram.render("img/chp04/salad_sales_basic_regression_model_pooled", format="png", cleanup=True)
pooled_sales_diagram
```

```python
az.plot_trace(idata_sales_pooled, var_names=["β", "σ"], compact=False);
```

```python
az.summary(idata_sales_pooled, var_names=["β", "σ"])
```

### Figure 4.17

```python
axes = az.plot_forest([idata_sales_pooled, idata_sales_unpooled],
                      model_names = ["Pooled", "Unpooled"], var_names=["σ"], combined=True, figsize=(10, 3));
axes[0].set_title("Comparison of pooled and unpooled models \n 94% HDI")

#plt.subplots_adjust(top=1)
plt.savefig("img/chp04/salad_sales_basic_regression_forestplot_sigma_comparison.png", bbox_inches='tight')
```

### Figure 4.18

```python
fig, ax = plt.subplots(figsize=(10, 6))
μ_m = idata_sales_pooled.posterior["μ"].mean(("chain", "draw"))
#σ_m = idata_sales_pooled.posterior["σ"].mean(("chain", "draw"))

ax.plot(customers, μ_m, c='C4')

az.plot_hdi(customers, idata_sales_pooled.posterior_predictive["sales"], hdi_prob=.50, ax=ax)
az.plot_hdi(customers, idata_sales_pooled.posterior_predictive["sales"], hdi_prob=.94, ax=ax)


pizza_df.plot(x="customers", y="sales", kind="scatter", ax=ax, c="C1", label="Pizza", marker="^", s=60);
sandwich_df.plot(x="customers", y="sales", kind="scatter", ax=ax,  label="Sandwich", marker="s");
salad_df.plot(x="customers", y="sales", kind="scatter", ax=ax, label="Salad", c="C4");


ax.set_xlabel("Number of Customers")
ax.set_ylabel("Daily Sales Dollars")
ax.set_title("Pooled Regression")
plt.savefig("img/chp04/salad_sales_basic_regression_scatter_pooled.png")
```

### Code 4.11

```python
coords = {"meals":food_category.categories, "meals_idx":food_category}

with pm.Model(coords=coords) as model_pooled_sigma_sales:
    σ = pm.HalfNormal("σ", 20)
    β = pm.Normal("β", mu=10, sigma=20, dims="meals")
    
    μ = pm.Deterministic("μ", β[food_category.codes] * customers, dims="meals_idx")
    
    sales = pm.Normal("sales", mu=μ, sigma=σ, observed=sales_observed, dims="meals_idx")
    
    idata_pooled_sigma_sales = pm.sample()
    idata_pooled_sigma_sales.extend(pm.sample_posterior_predictive(idata_pooled_sigma_sales))
```

```python
multilevel_sales_diagram = pm.model_to_graphviz(model_pooled_sigma_sales)
multilevel_sales_diagram.render("img/chp04/salad_sales_basic_regression_model_multilevel", format="png", cleanup=True)
multilevel_sales_diagram
```

```python
az.summary(idata_pooled_sigma_sales, var_names=["β", "σ"])
```

### Figure 4.20

```python
fig, ax = plt.subplots(figsize=(10, 6))
σ_m = idata_sales_pooled.posterior["σ"].mean().values

# Salads

for meal in food_category.categories:
    category_mask = (food_category==meal)
    μ_m_meals = idata_pooled_sigma_sales.posterior["μ"].sel({"meals_idx":meal})
    ax.plot(sales_df.customers[category_mask], μ_m_meals.mean(("chain", "draw")), c='C4')
    az.plot_hdi(sales_df.customers[category_mask], 
                idata_pooled_sigma_sales.posterior_predictive["sales"].sel({"meals_idx":meal}),
                hdi_prob=.50, ax=ax, fill_kwargs={"alpha": .5})


pizza_df.plot(x="customers", y="sales", kind="scatter", ax=ax, c="C1", label="Pizza", marker="^", s=60);
sandwich_df.plot(x="customers", y="sales", kind="scatter", ax=ax,  label="Sandwich", marker="s");
salad_df.plot(x="customers", y="sales", kind="scatter", ax=ax, label="Salad", c="C4");


ax.set_xlabel("Number of Customers")
ax.set_ylabel("Daily Sales Dollars")
ax.set_title("Unpooled Slope Pooled Sigma Regression")
plt.savefig("img/chp04/salad_sales_basic_regression_scatter_sigma_pooled_slope_unpooled.png")
```

### Figure 4.21

```python
axes = az.plot_forest([idata_sales_unpooled,
                       idata_pooled_sigma_sales
                      ],
                      model_names = ["Unpooled",
                                     "Multilevel "
                                    ],
                      var_names=["σ"], combined=True, figsize=(7, 1.8));
axes[0].set_title("Comparison of σ parameters 94% HDI")

plt.savefig("img/chp04/salad_sales_forestplot_sigma_unpooled_multilevel_comparison.png")
```

```python
axes = az.plot_forest([idata_sales_unpooled,
                       idata_pooled_sigma_sales
                      ],
                      model_names = ["Unpooled",
                                     "Multilevel"
                                    ],
                      var_names=["β"], combined=True, figsize=(7, 2.8));
axes[0].set_title("Comparison of β parameters 94% HDI");
```

### Hierarchical


### Code 4.12

```python
with pm.Model(coords=coords) as model_hierarchical_sales:
    σ_hyperprior = pm.HalfNormal("σ_hyperprior", 20)
    σ = pm.HalfNormal("σ", σ_hyperprior,  dims="meals")
    
    β = pm.Normal("β", mu=10, sigma=20, dims="meals")
    μ = pm.Deterministic("μ", β[food_category.codes] * customers)
    
    sales = pm.Normal("sales", mu=μ, sigma=σ[food_category.codes],
                      observed=sales_observed)
    
    idata_hierarchical_sales = pm.sample(target_accept=.9)
```

```python
az.plot_trace(idata_hierarchical_sales, compact=False, var_names=["β", "σ", "σ_hyperprior"]);
```

```python
az.plot_parallel(idata_hierarchical_sales, var_names=["σ", "σ_hyperprior"])
```

### Figure 4.23

```python
hierarchial_sales_diagram = pm.model_to_graphviz(model_hierarchical_sales)
hierarchial_sales_diagram.render("img/chp04/salad_sales_hierarchial_regression_model", format="png", cleanup=True)
hierarchial_sales_diagram
```

```python
az.summary(idata_hierarchical_sales, var_names=["β", "σ"])
```

```python
axes = az.plot_forest(idata_hierarchical_sales, var_names=["β"], combined=True,  figsize=(7, 1.5))
axes[0].set_title("Hierarchical β estimates 94% HDI")
```

```python
axes = az.plot_forest(idata_hierarchical_sales, var_names=["σ", "σ_hyperprior"], combined=True,  figsize=(7, 1.8))
axes[0].set_title("Hierarchical σ estimates 94% HDI")
plt.savefig("img/chp04/salad_sales_forestplot_sigma_hierarchical.png")

```

```python
print(food_category.categories)
```

### Table 4.3

```python
az.summary(idata_sales_unpooled.posterior["σ"], kind="stats").round(1)
```

### Table 4.4

```python
az.summary(idata_hierarchical_sales, var_names=["σ", "σ_hyperprior"], kind="stats").round(1)
```

```python
axes = az.plot_forest([idata_sales_unpooled.posterior["σ"].sel({"meals":"Salad"}),
                       idata_hierarchical_sales
                      ],
                      model_names = ["sales_unpooled",
                                     "sales_hierarchical"
                                    ], combined=True, figsize=(10, 4),
                     var_names=["σ", "σ_hyperprior"]
                     );
axes[0].set_title("Comparison of σ parameters from unpooled \n and hierarchical models \n 94% HDI")

plt.savefig("img/chp04/salad_sales_forestolot_sigma_unpooled_multilevel_comparison.png")
```

```python
fig, ax = plt.subplots()
az.plot_kde(idata_sales_unpooled.posterior['σ'].sel({"meals":"Salad"}).values, 
            label="Unpooled Salad Sigma", ax=ax)
az.plot_kde(idata_hierarchical_sales.posterior["σ"].sel({"meals":"Salad"}).values, 
            label="Hierarchical Salad Sigma", plot_kwargs={"color":"C4"}, ax=ax)

ax.set_title("Comparison of Hierarchical versus Unpooled Variance")
```

### Figure 4.25

```python
nsample = 10000
nd=1
yr = stats.norm.rvs(loc=2., scale=3., size=nsample)
xnr = stats.norm.rvs(loc=0., scale=np.exp(yr/4), size=(nd, nsample))

fig, ax = plt.subplots()
ax.scatter(xnr[0], yr, marker='.', alpha=.05, color="C4")
ax.set_xlim(-20, 20)
ax.set_ylim(-9, 9)
ax.set_xlabel('x')
ax.set_ylabel('y')
```

```python
def salad_generator(hyperprior_beta_mean=5, hyperprior_beta_sigma=.2, sigma=50, days_per_location=[6, 4, 15, 10, 3, 5], sigma_per_location=[50,10,20,80,30,20]):
    """Generate noisy salad data"""
    beta_hyperprior = stats.norm(hyperprior_beta_mean, hyperprior_beta_sigma)
    
    # Generate demands days per restaurant
    df = pd.DataFrame()
    for i, days in enumerate(days_per_location):
        np.random.seed(0)

        num_customers = stats.randint(30, 100).rvs(days)
        sales_location = beta_hyperprior.rvs()*num_customers + stats.norm(0, sigma_per_location[i]).rvs(num_customers.shape)

        location_df = pd.DataFrame({"customers":num_customers, "sales":sales_location})
        location_df["location"] = i
        location_df.sort_values(by="customers", ascending=True)
        df = pd.concat([df, location_df])
        
    df.reset_index(inplace=True, drop=True)
    return df
hierarchical_salad_df = salad_generator()
```

### Figure 4.26

```python
fig, axes, = plt.subplots(2,3, sharex=True, sharey=True)

for i, ax in enumerate(axes.ravel()):
    location_filter = (hierarchical_salad_df["location"] == i)
    hierarchical_salad_df[location_filter].plot(kind="scatter", x="customers", y="sales", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")

    
axes[1,0].set_xlabel("Number of Customers")
axes[1,0].set_ylabel("Sales");
plt.savefig("img/chp04/multiple_salad_sales_scatter.png")
```

### Code 4.13

```python
# import tensorflow as tf
# import tensorflow_probability as tfp

# tfd = tfp.distributions
# root = tfd.JointDistributionCoroutine.Root
```

```python
# run_mcmc = tf.function(
#     tfp.experimental.mcmc.windowed_adaptive_nuts,
#     autograph=False, jit_compile=True)
```

```python
# def gen_hierarchical_salad_sales(input_df, beta_prior_fn, dtype=tf.float32):
#     customers = tf.constant(
#         hierarchical_salad_df["customers"].values, dtype=dtype)
#     location_category = hierarchical_salad_df["location"].values
#     sales = tf.constant(hierarchical_salad_df["sales"].values, dtype=dtype)

#     @tfd.JointDistributionCoroutine
#     def model_hierarchical_salad_sales():
#         β_μ_hyperprior = yield root(tfd.Normal(0, 10, name="beta_mu"))
#         β_σ_hyperprior = yield root(tfd.HalfNormal(.1, name="beta_sigma"))
#         β = yield from beta_prior_fn(β_μ_hyperprior, β_σ_hyperprior)

#         σ_hyperprior = yield root(tfd.HalfNormal(30, name="sigma_prior"))
#         σ = yield tfd.Sample(tfd.HalfNormal(σ_hyperprior), 6, name="sigma")

#         loc = tf.gather(β, location_category, axis=-1) * customers
#         scale = tf.gather(σ, location_category, axis=-1)
#         sales = yield tfd.Independent(tfd.Normal(loc, scale),
#                                       reinterpreted_batch_ndims=1,
#                                       name="sales")

#     return model_hierarchical_salad_sales, sales
```

### Code 4.14 and 4.15

```python
# def centered_beta_prior_fn(hyper_mu, hyper_sigma):
#     β = yield tfd.Sample(tfd.Normal(hyper_mu, hyper_sigma), 6, name="beta")
#     return β

# # hierarchical_salad_df is the generated dataset as pandas.DataFrame
# centered_model, observed = gen_hierarchical_salad_sales(
#     hierarchical_salad_df, centered_beta_prior_fn)

# mcmc_samples_centered, sampler_stats_centered = run_mcmc(
#     1000, centered_model, n_chains=4, num_adaptation_steps=1000,
#     sales=observed)

# divergent_per_chain = np.sum(sampler_stats_centered['diverging'], axis=0)
# print(f"""There were {divergent_per_chain} divergences after tuning per chain.""")
```

```python
# idata_centered_model = az.from_dict(
#     posterior={
#         k:np.swapaxes(v, 1, 0)
#         for k, v in mcmc_samples_centered._asdict().items()},
#     sample_stats={
#         k:np.swapaxes(sampler_stats_centered[k], 1, 0)
#         for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]}
# )

# az.plot_trace(idata_centered_model, compact=True);
```

```python
# az.summary(idata_centered_model)
```

### Figure 4.27

```python
# slope = mcmc_samples_centered.beta[..., 4].numpy().flatten()
# sigma = mcmc_samples_centered.beta_sigma.numpy().flatten()
# divergences = sampler_stats_centered['diverging'].numpy().flatten()

# axes = az.plot_joint({"β[4]": slope, "β_σ_hyperprior": sigma},
#                      joint_kwargs={"alpha": .05}, figsize=(6, 6))
# axes[0].scatter(slope[divergences], sigma[divergences], c="C4", alpha=.3, label='divergent sample')
# axes[0].legend(frameon=True)
# axes[0].set_ylim(0, .3)
# axes[0].set_xlim(4.5, 5.5)

# plt.savefig("img/chp04/Neals_Funnel_Salad_Centered.png")
```

### Code 4.16

```python
# def non_centered_beta_prior_fn(hyper_mu, hyper_sigma):
#     β_offset = yield root(tfd.Sample(tfd.Normal(0, 1), 6, name="beta_offset"))
#     return β_offset * hyper_sigma[..., None] + hyper_mu[..., None]

# # hierarchical_salad_df is the generated dataset as pandas.DataFrame
# non_centered_model, observed = gen_hierarchical_salad_sales(
#     hierarchical_salad_df, non_centered_beta_prior_fn)

# mcmc_samples_noncentered, sampler_stats_noncentered = run_mcmc(
#     1000, non_centered_model, n_chains=4, num_adaptation_steps=1000,
#     sales=observed)

# divergent_per_chain = np.sum(sampler_stats_noncentered['diverging'], axis=0)
# print(f"""There were {divergent_per_chain} divergences after tuning per chain.""")
```

```python
# idata_non_centered_model = az.from_dict(
#     posterior={
#         k:np.swapaxes(v, 1, 0)
#         for k, v in mcmc_samples_noncentered._asdict().items()},
#     sample_stats={
#         k:np.swapaxes(sampler_stats_noncentered[k], 1, 0)
#         for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]}
# )

# az.plot_trace(idata_non_centered_model, compact=True);
```

```python
# az.summary(idata_non_centered_model)
```

### Figure  4.28

```python
# noncentered_beta = (mcmc_samples_noncentered.beta_mu[..., None]
#         + mcmc_samples_noncentered.beta_offset * mcmc_samples_noncentered.beta_sigma[..., None])
# slope = noncentered_beta[..., 4].numpy().flatten()
# sigma = mcmc_samples_noncentered.beta_sigma.numpy().flatten()
# divergences = sampler_stats_noncentered['diverging'].numpy().flatten()

# axes = az.plot_joint({"β[4]": slope, "β_σ_hyperprior": sigma},
#                      joint_kwargs={"alpha": .05}, figsize=(6, 6))
# axes[0].scatter(slope[divergences], sigma[divergences], c="C4", alpha=.3, label='divergent sample')
# axes[0].legend(frameon=True)
# axes[0].set_ylim(0, .3)
# axes[0].set_xlim(4.5, 5.5)

# plt.savefig("img/chp04/Neals_Funnel_Salad_NonCentered.png")
```

### Figure 4.29

```python
# centered_β_sigma = mcmc_samples_centered.beta_sigma.numpy()
# noncentered_β_sigma = mcmc_samples_noncentered.beta_sigma.numpy()
```

```python
# fig, ax = plt.subplots()
# az.plot_kde(centered_β_sigma, label="Centered β_σ_hyperprior", ax=ax)
# az.plot_kde(noncentered_β_sigma, label="Noncentered β_σ_hyperprior", plot_kwargs={"color":"C4"}, ax=ax);

# ax.set_title("Comparison of Centered vs Non Centered Estimates");
# plt.savefig("img/chp04/Salad_Sales_Hierarchical_Comparison.png")
```

### Code 4.17

```python
# out_of_sample_customers = 50.

# @tfd.JointDistributionCoroutine
# def out_of_sample_prediction_model():
#     model = yield root(non_centered_model)
#     β = model.beta_offset * model.beta_sigma[..., None] + model.beta_mu[..., None]
    
#     β_group = yield tfd.Normal(
#         model.beta_mu, model.beta_sigma, name="group_beta_prediction")
#     group_level_prediction = yield tfd.Normal(
#         β_group * out_of_sample_customers,
#         model.sigma_prior,
#         name="group_level_prediction")
#     for l in [2, 4]:
#         yield tfd.Normal(
#             tf.gather(β, l, axis=-1) * out_of_sample_customers,
#             tf.gather(model.sigma, l, axis=-1),
#             name=f"location_{l}_prediction")

# amended_posterior = tf.nest.pack_sequence_as(
#     non_centered_model.sample(),
#     list(mcmc_samples_noncentered) + [observed],
# )
# ppc = out_of_sample_prediction_model.sample(var0=amended_posterior)
```

```python
# fig, ax = plt.subplots()

# az.plot_kde(ppc.group_level_prediction, plot_kwargs={"color":"C0"}, ax=ax, label="All locations")
# az.plot_kde(ppc.location_2_prediction, plot_kwargs={"color":"C2"}, ax=ax, label="Location 2")
# az.plot_kde(ppc.location_4_prediction, plot_kwargs={"color":"C4"}, ax=ax, label="Location 4")

# ax.set_xlabel("Predicted revenue with 50 customers")
# ax.set_xlim([0,600])

# ax.set_yticks([])

# plt.savefig("img/chp04/Salad_Sales_Hierarchical_Predictions.png")
```

### Code 4.18

```python
# out_of_sample_customers2 = np.arange(50, 90)

# @tfd.JointDistributionCoroutine
# def out_of_sample_prediction_model2():
#     model = yield root(non_centered_model)
    
#     β_new_loc = yield tfd.Normal(
#         model.beta_mu, model.beta_sigma, name="beta_new_loc")
#     σ_new_loc = yield tfd.HalfNormal(model.sigma_prior, name="sigma_new_loc")
#     group_level_prediction = yield tfd.Normal(
#         β_new_loc[..., None] * out_of_sample_customers2,
#         σ_new_loc[..., None],
#         name="new_location_prediction")

# ppc = out_of_sample_prediction_model2.sample(var0=amended_posterior)
```

```python
# az.plot_hdi(out_of_sample_customers2, ppc.new_location_prediction, hdi_prob=.95)
```
