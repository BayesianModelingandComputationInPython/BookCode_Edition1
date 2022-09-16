# Errata


# 1st Printing

| Page | Printed text | Correct text | Note |
|---|---|---|---|
|xvi| If you have a good understanding of statistics, either by practice or formal training, but you have never being ... | If you have a good understanding of statistics, either by practice or formal training, but you have never **being**... | Thanks Behrouz B. |
| xvi | ...and may require a couple **read troughs** | ...and may require a couple **read-throughs** | Thanks John M. Shea |
| xvi | For a reference on Python, or how to setup the computation environment needed for this book, go to README.md in Github to understand how to setup a code environment | For a reference on how to setup the computation environment needed for this book, go to README.md in GitHhub. |  |
| 1 | ...we introduce these concepts and methods, many, which... | ...we introduce these concepts and methods, many **of** which... | Thanks  Thomas Ogden |
| 2 | ...though this is not a **guaranteed** of any Bayesian model. | ...though this is not a **guarantee**  of any Bayesian model. | Thanks  Guilherme Costa  |
|7| ...(conceptually it means it is equally likely **are** we are... | ...(conceptually it means it is equally likely we are... | Thanks Behrouz B. |
|8| At line **20**.... | At line **14**... | Thanks Behrouz B. |
| 8 | ...it will **depends** on the result... | ...it will **depend** on the result.. | Thanks Ero Carrera |
|9| Some people make the distinction that a sample is made up by a collection of draws, **other**... | Some people make the distinction that a sample is made up by a collection of draws, **others**... | Thanks Behrouz B. |
| 9 | ...or **simple** the posterior. | ...or **simply** the posterior.  | Thanks Ero Carrera |
| 23 | An absolute value mean... | An absolute deviation to the mean... | Thanks Zhengchen Cai|
| 24 | One is what could called... | One is what could **be** called... | Thanks Sebastian |
| 26 | 1E8. Rerun Code block | 1E8. Rerun Code Block |  |
| 27 | Which can can be used **to to** visualize a Highest Density Interval? | Which can can be used **to** visualize a Highest Density Interval? |  |
| 28 | Build a model that will make **these** estimation. | Build a model that will make **this** estimation. |  |
| 28 | Determine two prior **distribution** that satisfy these constraints using Python. | Determine two prior **distributions** that satisfy these constraints using Python. |  |
| 31 | In this chapter we will discuss some of these tasks including**,** checking ... results and model comparison | In this chapter we will discuss some of these tasks**,** including checking ... results**,** and model comparison  | Thanks Sebastian |
| 42 | Plotting the ESS for specifics quantiles `az.plot_ess(., kind="quantiles"`  | Plotting the ESS for specifics quantiles `az.plot_ess(., kind="quantiles"`) | Thanks Juan Orduz |
| 49 | ...which means `model_0` specified a posterior... | ...which means `model_0` specifies a posterior... | Thanks Sebastian |
| 51 | Thus increasing the **turning** steps can help to increase the ESS... | Thus increasing the **tuning** steps can help to increase the ESS... | Thanks Ero Carrera |
| 52 | where $p_t(\tilde y_i)$ is distribution of the true data-generating process... | where $p_t(\tilde y_i)$ is **the** distribution of the true data-generating process... | Thanks Ero Carrera |
| 52 | ...and it is **use** in both Bayesians and non-Bayesians contexts. | ...and it is **used** in both Bayesians and non-Bayesians contexts. | Thanks Sebastian |
| 53 | It is important to remember we are ~~**are**~~ talking about PSIS-LOO-CV... | It is important to remember we are talking about PSIS-LOO-CV... | Thanks Ero Carrera |
| 54 | 2. `rank`: The ranking **on** the models starting... | 2. `rank`: The ranking **of** the models starting... | Thanks Ero Carrera |
| 54 | 4. `p_loo`: The list values for the... | 4. `p_loo`: The list **of** values for the... | Thanks Ero Carrera |
| 54 | ...the actual number of parameters in model that *has more structure* like hierarchical **models** or can be much higher than the actual.. | ...the actual number of parameters in **a** model that *has more structure* like **a** hierarchical **model** or can be much higher than the actual... | Thanks Ero Carrera |
| 58 | ...we can obtain some additional ~~**additional**~~ information. | ...we can obtain some additional information. | Thanks Ero Carrera |
| 58 | ...comparing p_loo to the number of parameters $p$ can **provides** us with... | ...comparing p_loo to the number of parameters $p$ can **provide** us with... | Thanks Ero Carrera |
| 59 | ...which is transformation in 1D where we can... | ...which is **a** transformation in 1D where we can... | Thanks Ero Carrera |
| 61 | When using a logarithmic scoring rule this is **equivalently** to **compute**: | When using a logarithmic scoring rule this is **equivalent** to **computing**: | Thanks Ero Carrera |
|61| $\max_{n} \frac{1}{n} \sum_{i=1}^{n}log\sum_{j=1}^{k} w_j p(y_i \mid y_{-i}, M_j)$ | $\max_{w} \frac{1}{n} \sum_{i=1}^{n}log\sum_{j=1}^{k} w_j p(y_i \mid y_{-i}, M_j)$ | Thanks Ikaro Silva  |
| 61 | ...the computation of the weights **take** into account all models together. | ...the computation of the weights **takes** into account all models together. | Thanks Ero Carrera |
| 61 | ...the weights computed with `az.compare(., method="stacking")`~~**,**~~ makes a lot of sense. | ...the weights computed with `az.compare(., method="stacking")` makes a lot of sense. | Thanks Ero Carrera |
| 62 | ...Reproduce Figure 2.7, but using **az.plot_loo(ecdf=True)**... | ...Reproduce Figure 2.7, but using **az.plot_loo_pit(ecdf=True)**... | Thanks Alihan Zihna |
| 62 | Use `az.load_arviz_data(.)` to load them... | Use `az.from_netcdf(.)` to load them... | Thanks Ikaro Silva |
| 64 | ...and prior distribution $\mathcal{N}(201)... | ...and prior distribution $\mathcal{N}(20, 1)... | |
| 70 |  | Figure 3.3 updated to include vertical lines of empirical estimate | |
| 71 | Take a moment to compare the estimate of the mean with the summary mean show**s**... | Take a moment to compare the estimate of the mean with the summary mean show**n**... | Thanks Sebastian |
| 73 | ...the **thin** line is the interquartile range **from 25% to 75% of the posterior** and the **thick** line is the 94% Highest Density Interval | the **thick** line is the interquartile range and the **thin** line is the 94% Highest Density Interval | Thanks Jose Roberto Ayala Solares |
| 73 | ... more **compostable** modeling and inference. | ... more **composable** modeling and inference. | |
| 75 | ... is the intercept only regression model **from** | is the intercept only regression model in **Code Block** |  |
| 77 | ...where the coefficients, also referred to as covariates, are represented by the parameter $\beta_i$... | ...where the coefficients, **also referred as parameters, are represented** with $\beta_i$. | Thanks Sebastian |
| 78 | to parse the categorical information into a design matrix `mu = pd.get_dummies(penguins["species"]) @ μ`**.** where  | ...to parse the categorical information into a design matrix, **and then write** `mu = pd.get_dummies(penguins["species"]) @ μ`**,** where... | Thanks Sebastian |
| 81 | ...we would expect the mass of this impossible penguin to somewhere between **-4213** and **-546** grams. | ...we would expect the mass of this impossible penguin to **be** somewhere between **-4151** and **-510** grams. | Thanks Sebastian |
| 83 | ...which takes a set **a value**... | ...which takes a set **of values**... | Thanks Sebastian |
| 86 | ...has dropped a mean of 462 grams ... to a mean value 298 grams... | ...has dropped **from** a mean of 462 grams ... to a mean value **of** 298 grams... | Thanks Sebastian |
| 88 | ...which lower value than estimated...  | ...which is a lower value than the estimated... | Thanks Sebastian |
| 89 | ...which useful for counterfactual analyses. | ...which **is** useful for counterfactual analyses. | Thanks Sebastian |
| 91 | We are still dealing a linear model here... | We are still dealing **with** a linear model here... | Thanks Sebastian |
| 93 | ...we find it reasonable to equally expect a **Gentoo** penguin... | ...we find it reasonable to equally expect a **Chinstrap** penguin... | Thanks Sebastian |
| 97 | ...and Fig. 3.22.A separation... | ...and Fig. 3.22. A separation... | Thanks Sebastian |
| 98 | ...from Adelie or Chinstrap **penguinsthe**... | ...from Adelie or Chinstrap **penguins the**... | Thanks Sebastian |
| 101 | Given these choices we can write our model in Code Block 3.30**)**... | Given these choices we can write our model in Code Block 3.30 | Thanks Sebastian |
| 101 | This is not a fully uninformative **priors**... | This is not a fully uninformative **prior**... | Thanks Sebastian |
| 102 | ...fall into bounds that more reasonable... | ...fall into bounds that **are** more reasonable... | Thanks Sebastian |
| 126 | ... describes the distribution **of** for the parameters of the prior itself... | ... describes the distribution for the parameters of the prior itself... | |
| 130 | ...the estimates of the pizza and salad categories... | ...the estimates of the pizza and **sandwich** categories... | Thanks  @paw-lu |
| 133 | (In Code Block 9.1) inside function `gen_hierarchical_salad_sales` all reference to `hierarchical_salad_df` | should be `input_df` | Thanks Alihan Zihna |
| 156 | Fig 5.7 y_labels is **count_std** | should be **count_normalized** | Thanks Paulo S. Costa |
| 183 | ...a backshift operator, also called Lag operator) | ...a backshift operator **(**also called Lag operator) |  |
| 189 | Equation (6.9) $y_t = \alpha + \sum_{i=1}^{p}\phi_i y_{t-period-i} + \sum_{j=1}^{q}\theta_j \epsilon_{t-period-j} + \epsilon_t$ | $y_t = \alpha + \sum_{i=1}^{p}\phi_i y_{t-period \cdot i} + \sum_{j=1}^{q}\theta_j \epsilon_{t-period \cdot j} + \epsilon_t$ | Thanks Marcin Elantkowski |
| 191 | (footnote) The Stan implementation of SARIMA can be found in https://github.com/asael697/**varstan**. | The Stan implementation of SARIMA can be found in **e.g.,** https://github.com/asael697/**bayesforecast**. |  |
| 197 | we can apply the Kalman filter to **to** obtain**s** the posterior | we can apply the Kalman filter to obtain the posterior |  |
|227| Only the first 2 independent variables are **unrelated**... | Only the first 2 independent variables are **related**... | Thanks  icfly2   |
| 261 | Some **commons** elements to all Bayesian analyses, | Some **common** elements to all Bayesian analyses, | Thanks Ero Carrera |
| 262 | (In Figure 9.1.) Model **Compasion** | Model **Comparison** | Thanks Ben Vincent |
| 262 | ...averaging some **of** all of them, or even presenting all the models and discussing their **strength** and... | ...averaging some **or** all of them, or even presenting all the models and discussing their **strengths** and... | Thanks Ero Carrera |
| 262 | A more detailed version of the Bayesian workflow can be **see** in a paper... | A more detailed version of the Bayesian workflow can be **seen** in a paper... | Thanks Ero Carrera |
| 262 | ...but should not be confused with driving question... | ...but should not be confused with **the** driving question... | Thanks Ero Carrera |
| 264 | ...report regarding the potential financial **affects**. | ...report regarding the potential financial **effects**. | Thanks Ero Carrera |
| 264 | ...risk of making a poor decision far **outweights**... | ...risk of making a poor decision far **outweighs**... | Thanks Ero Carrera |
| 264 | ...in the sub-sections with **title start** with *Applied Example*. | ...in the sub-sections with **titles starting** with *Applied Example*. | Thanks Ero Carrera |
| 264 | Likewise inference is impossible without data. challenging with poor quality data, **and the** best statisticians... | Likewise, inference is impossible without data **and** challenging with poor quality data. **The** best statisticians... | Thanks Ero Carrera |
| 264 | For statistician the equivalent is **Sample** surveys... | For statistician the equivalent is **sample** surveys... | Thanks Ero Carrera |
| 265 | foraging for ingredients are growing by themselves. | foraging for ingredients **that** are growing by themselves. |  |
| 265 | When collecting data be sure not only pay attention to what is present, but consider what may not be present. | When collecting data be sure **to** not only pay attention to what is present, but consider **also** what may not be present. | Thanks Ero Carrera |
| 267 | We do this using numerous tools, diagnostics, and visualizations that we have seen **through out** this book. | We do this using **the** numerous tools, diagnostics, and visualizations that we have seen **throughout** this book. | Thanks Ero Carrera |
| 267 | The fundamentals of Bayes formula **has** no opinion... | The fundamentals of Bayes formula **have** no opinion... | Thanks Ero Carrera |
| 267 | We take a moment to collect **our** detail... | We take a moment to collect **in** detail... | Thanks Ero Carrera |
| 267 | (In Code Block 9.1) `df = pd.read_csv("../data/948363589_T_ONTIME_MARKETING.zip",` | `df = pd.read_csv("../data/948363589_T_ONTIME_MARKETING.zip")` |  |
| 269 | ...or likelihood distributions are preordained, What is printed in this book... | ...or likelihood distributions are preordained. What is printed in this book... | Thanks Ero Carrera |
| 276 | We can also generate a visual check with **9.7which** | We can also generate a visual check with **Code Block 9.7 which** |  |
| 276 | Thus, there still room for improvement... | Thus, there **is** still room for improvement... | Thanks Ero Carrera |
| 276 | What is important at this step is we are sufficiently... | What is important at this step is **that** we are sufficiently... | Thanks Ero Carrera |
| 277 | ...maps how wet or dry a person’s clothes **is** to... | ...maps how wet or dry a person’s clothes **are** to... | Thanks Ero Carrera |
| 277 | ...map those outcomes to expected **reward**... | ...map those outcomes to expected **rewards**... | Thanks Ero Carrera |
| 280 | ...the airline, **If** a flight is between 0 and 10 minutes late, the fee is 1,000 dollars. **if** the flight is... | ...the airline, **if** a flight is between 0 and 10 minutes late, the fee is 1,000 dollars. **If** the flight is... | Thanks Ero Carrera |
| 280 | ...more time than all the previous **one** combined. | ...more time than all the previous **ones** combined. | Thanks Ero Carrera |
| 281 | ...and **legality** but is important to note this. | ...and **legal considerations** but is important to note this. | Thanks Ero Carrera |
| 282 | When environments cannot be replicated one outcome is **code that was working**... | When environments cannot be replicated one **possible** outcome is **that code that used to work**... | Thanks Ero Carrera |
| 282 | This can occur because the **library** may change, or the **algorithm itself**. | This can occur because the **libraries** may change, or the **algorithms themselves**. | Thanks Ero Carrera |
| 282 | workflow should be robust that changing the seed | workflow should be robust **so** that changing the seed | Thanks Ero Carrera |
| 282 | In short reproducible analyses both **helps** you and others build confidence in your prior results, and also **helps** future efforts extend the work. | In short reproducible analyses both **help** you and others build confidence in your prior results, and also **help** future efforts extend the work. | Thanks Ero Carrera |
| 284 | which used a shaking needle gauge to highlight. | highlight **the uncertainty in the estimation of which candidate would ultimately win.** | Thanks ST John |
| 284 | or the randomness of the **plinko** drops in **Matthew** Kay's | or the randomness of the **Plinko** drops in **Matthew** Kay's | |
| 286 | ... a cross section area of .504 inches (**12.8mm**) by .057 inches (**1.27**)... | ... a cross section area of .504 inches (**12.8 mm**) by .057 inches (**1.27 mm**)... | Thanks Juan Orduz |
| 289 | ... In both the plots a value of 0 **seems** is relatively unlikely ... | ... In both the plots a value of 0 is relatively unlikely ... | |
| 297 | | Equation 10.1 and Code 10.4 updated for readability | Thanks ST John |
| 299 | The Zen of Python **detai** the philosophy behind **this** idea of pythonic design... | The Zen of Python **details** the philosophy behind **the** idea of pythonic design ... | |
| 318 | As you can see, there is a lot of **rooms** for... | As you can see, there is a lot of **room** for... |  |
| 344 | ...will make the **skweness** independent... | ...will make the **skewness** independent... | Thanks Alihan Zihna |
| 371 | ...a simple Python implementation in **Code block** | ...simple Python implementation in **Code Block** |  |
| 376 |  We can see that all these trajectories **when** wrong. We call this kind **these divergences and we can used as diagnostics of the** HMC samplers. | We can see that all these trajectories went wrong. We call this kind **of trajectories divergences and can be used as a diagnostic of** HMC samplers | Thanks Alihan Zihna |
| 380 | ... if **you** future lab... | ... if **your** future lab... | Thanks Alihan Zihna |
| 385 | ...more parameters than can be justified by the data.**[2]** | ... more parameters than can be justified by the data. |  |
